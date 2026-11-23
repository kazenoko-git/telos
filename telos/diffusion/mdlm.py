"""
Masked Diffusion Language Modeling (MDLM) loss and masking logic.
Contains implementations for both MLX and PyTorch.
"""

import math
import numpy as np


# =========================================================================
# MLX IMPLEMENTATION
# =========================================================================

try:
    from telos.training.core import MLX_AVAILABLE
    if MLX_AVAILABLE:
        import mlx.core as mx
        import mlx.nn as mx_nn
        from telos.training.mlx_losses import cross_entropy
except ImportError:
    MLX_AVAILABLE = False


if MLX_AVAILABLE:
    def apply_masking_mlx(input_ids, t_values, mask_token_id=1, special_token_lut=None):
        B, T = input_ids.shape
        rand_matrix = mx.random.uniform(0.0, 1.0, (B, T))
        raw_mask = rand_matrix < t_values

        if special_token_lut is not None:
            is_special = special_token_lut[input_ids]
        else:
            is_special = input_ids < 4

        mask_positions = raw_mask & (~is_special)
        masked_input_ids = mx.where(mask_positions, mask_token_id, input_ids)
        return masked_input_ids, mask_positions, t_values

    def mdlm_loss_mlx(model, masked_input_ids, targets, mask_positions, t_values, vocab_size):
        logits = model(masked_input_ids)
        B, T, V = logits.shape
        # The Metal loss reduces in FP32 without upcasting the logits matrix.
        logits_flat = logits.reshape(-1, V)
        targets_flat = targets.reshape(-1)

        ce_per_token = cross_entropy(logits_flat, targets_flat, reduction="none").reshape(B, T)
        masked_ce = ce_per_token * mask_positions

        masked_count = mx.clip(mx.sum(mask_positions, axis=1), 1.0, float(T))
        per_example_ce = mx.sum(masked_ce, axis=1) / masked_count
        unweighted_ce = mx.mean(per_example_ce)

        t_weights = 1.0 / mx.clip(mx.squeeze(t_values, -1), 1e-3, 1.0)
        reweighted_loss = mx.mean(per_example_ce * t_weights)
        return reweighted_loss, unweighted_ce


# =========================================================================
# PYTORCH IMPLEMENTATION
# =========================================================================

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    def apply_masking_pytorch(input_ids: torch.Tensor, t_values: torch.Tensor, mask_token_id: int = 1, special_token_lut: torch.Tensor | None = None):
        """Applies random masking according to t_values (masking probability)."""
        B, T = input_ids.shape
        rand_matrix = torch.rand(B, T, device=input_ids.device)
        raw_mask = rand_matrix < t_values

        if special_token_lut is not None and not str(input_ids.device).startswith("xla"):
            is_special = special_token_lut[input_ids]
        else:
            is_special = input_ids < 4

        mask_positions = raw_mask & (~is_special)
        masked_input_ids = torch.where(mask_positions, mask_token_id, input_ids)
        return masked_input_ids, mask_positions, t_values

    def mdlm_loss_pytorch(
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask_positions: torch.Tensor,
        t_values: torch.Tensor,
        label_smoothing: float = 0.0
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """compute 1/t reweighted MDLM Cross-Entropy loss."""
        batch_size, seq_len, vocab_size = logits.shape

        flat_mask = mask_positions.view(-1)
        flat_logits = logits.view(-1, vocab_size)
        flat_targets = targets.view(-1)

        masked_count_per_example = mask_positions.sum(dim=1).float().clamp(min=1.0)

        ce_loss_per_token = F.cross_entropy(
            flat_logits,
            flat_targets,
            reduction="none",
            label_smoothing=label_smoothing
        ).view(batch_size, seq_len)

        masked_ce_loss = ce_loss_per_token * mask_positions.float()
        per_example_ce = masked_ce_loss.sum(dim=1) / masked_count_per_example

        t_weights = 1.0 / t_values.squeeze(-1).clamp(min=1e-3)
        reweighted_per_example_loss = per_example_ce * t_weights
        total_loss = reweighted_per_example_loss.mean()

        metrics = {
            "loss": total_loss,
            "unweighted_ce": per_example_ce.mean(),
            "masked_tokens_avg": mask_positions.sum().float() / batch_size
        }

        return total_loss, metrics

    def mdlm_loss_from_model_pytorch(
        model: torch.nn.Module,
        masked_input_ids: torch.Tensor,
        targets: torch.Tensor,
        mask_positions: torch.Tensor,
        t_values: torch.Tensor,
        label_smoothing: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute 1/t reweighted MDLM CE projecting only masked positions when supported."""
        raw_model = getattr(model, "_orig_mod", model)
        raw_model = getattr(raw_model, "module", raw_model)
        can_sparse_project = (
            masked_input_ids.device.type != "xla"
            and model is raw_model
            and hasattr(raw_model, "hidden_states")
            and hasattr(raw_model, "output_projection")
        )
        if not can_sparse_project:
            logits = model(masked_input_ids, mask_override=False)
            return mdlm_loss_pytorch(
                logits, targets, mask_positions, t_values, label_smoothing=label_smoothing,
            )

        batch_size, seq_len = masked_input_ids.shape
        hidden = raw_model.hidden_states(masked_input_ids, mask_override=False)
        flat_mask = mask_positions.reshape(-1)
        masked_count_per_example = mask_positions.sum(dim=1).float().clamp(min=1.0)

        if flat_mask.any():
            masked_hidden = hidden.reshape(-1, hidden.shape[-1])[flat_mask]
            masked_logits = raw_model.output_projection(masked_hidden)
            masked_targets = targets.reshape(-1)[flat_mask]
            masked_ce = F.cross_entropy(
                masked_logits,
                masked_targets,
                reduction="none",
                label_smoothing=label_smoothing,
            )
            row_idx = (
                torch.arange(batch_size, device=masked_input_ids.device)
                .unsqueeze(1)
                .expand(batch_size, seq_len)
                .reshape(-1)[flat_mask]
            )
            per_example_sum = torch.zeros(batch_size, dtype=masked_ce.dtype, device=masked_input_ids.device)
            per_example_sum.scatter_add_(0, row_idx, masked_ce)
        else:
            per_example_sum = hidden.sum(dim=(1, 2)).float() * 0.0

        per_example_ce = per_example_sum / masked_count_per_example
        t_weights = 1.0 / t_values.squeeze(-1).clamp(min=1e-3)
        total_loss = (per_example_ce * t_weights).mean()
        metrics = {
            "loss": total_loss,
            "unweighted_ce": per_example_ce.mean(),
            "masked_tokens_avg": mask_positions.sum().float() / batch_size,
        }
        return total_loss, metrics

    def apply_masking(
        input_ids: torch.Tensor,
        t_values: torch.Tensor | None = None,
        mask_token_id: int = 1,
        special_token_ids: set | None = None,
        special_token_lut: torch.Tensor | None = None
    ):
        """High-level apply_masking supporting both explicit t_values or sampled uniform t."""
        if t_values is None:
            B = input_ids.shape[0]
            t_values = torch.rand(B, 1, device=input_ids.device).clamp(min=1e-5, max=1.0)
        
        if special_token_lut is None and special_token_ids is not None:
            max_id = max(input_ids.max().item(), max(special_token_ids)) + 1
            special_token_lut = torch.zeros(max_id, dtype=torch.bool, device=input_ids.device)
            for sid in special_token_ids:
                special_token_lut[sid] = True
                
        return apply_masking_pytorch(input_ids, t_values, mask_token_id=mask_token_id, special_token_lut=special_token_lut)

    mdlm_loss = mdlm_loss_pytorch
else:
    apply_masking_pytorch = None
    mdlm_loss_pytorch = None
    apply_masking = None
    mdlm_loss = None


# =========================================================================
# TIMESTEP SAMPLING
# =========================================================================

def sample_uniform_timesteps(batch_size: int, eps: float = 1e-5):
    """Uniform timestep sampler for standard MDLM baseline training."""
    u = np.random.uniform(0.0, 1.0, (batch_size, 1))
    return np.clip(u, eps, 1.0)

def sample_beta_timesteps(batch_size: int, alpha: float = 1.5, beta: float = 1.5, eps: float = 1e-5):
    """Beta(α, β) timestep sampler matching MDLM paper specification."""
    t = np.random.beta(alpha, beta, size=(batch_size, 1)).astype(np.float32)
    return np.clip(t, eps, 1.0)
