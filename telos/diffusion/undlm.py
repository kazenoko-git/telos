"""
Uniform Noise Diffusion Language Modeling (UNDLM) loss and noise logic.
Contains implementations for both MLX and PyTorch.
"""

# =========================================================================
# MLX IMPLEMENTATION
# =========================================================================

try:
    import mlx.core as mx
    import mlx.nn as mx_nn
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


if MLX_AVAILABLE:
    def apply_uniform_noise_mlx(input_ids, t_values, vocab_size, special_token_lut=None):
        B, T = input_ids.shape
        rand_matrix = mx.random.uniform(0.0, 1.0, (B, T))
        raw_mask = rand_matrix < t_values

        if special_token_lut is not None:
            is_special = special_token_lut[input_ids]
        else:
            is_special = input_ids < 4

        corrupt_mask = raw_mask & (~is_special)
        random_tokens = mx.random.randint(0, vocab_size, (B, T))
        noisy_ids = mx.where(corrupt_mask, random_tokens, input_ids)
        return noisy_ids, corrupt_mask, t_values

    def undlm_loss_mlx(model, noisy_ids, clean_targets, t_values, vocab_size, special_token_lut=None):
        logits = model(noisy_ids)
        B, T, V = logits.shape

        logits_flat = logits.reshape(-1, V)
        targets_flat = clean_targets.reshape(-1)

        ce_per_token = mx_nn.losses.cross_entropy(logits_flat, targets_flat, reduction="none").reshape(B, T)

        if special_token_lut is not None:
            content_mask = ~special_token_lut[clean_targets]
            ce_per_token = ce_per_token * content_mask.astype(mx.float32)
            content_count = mx.clip(mx.sum(content_mask.astype(mx.float32), axis=1), 1.0, float(T))
            per_example_ce = mx.sum(ce_per_token, axis=1) / content_count
        else:
            per_example_ce = mx.mean(ce_per_token, axis=1)

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
    def apply_uniform_noise_pytorch(input_ids: torch.Tensor, t_values: torch.Tensor, vocab_size: int, special_token_lut: torch.Tensor | None = None):
        B, T = input_ids.shape
        rand_matrix = torch.rand(B, T, device=input_ids.device)
        raw_mask = rand_matrix < t_values

        if special_token_lut is not None:
            is_special = special_token_lut[input_ids]
        else:
            is_special = input_ids < 4

        corrupt_mask = raw_mask & (~is_special)
        random_tokens = torch.randint(0, vocab_size, (B, T), device=input_ids.device)
        noisy_ids = torch.where(corrupt_mask, random_tokens, input_ids)
        return noisy_ids, corrupt_mask, t_values

    def undlm_loss_pytorch(
        logits: torch.Tensor,
        clean_targets: torch.Tensor,
        t_values: torch.Tensor,
        special_token_lut: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict[str, float]]:
        batch_size, seq_len, vocab_size = logits.shape

        flat_logits = logits.view(-1, vocab_size)
        flat_targets = clean_targets.view(-1)

        ce_loss_per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none").view(batch_size, seq_len)

        if special_token_lut is not None:
            content_mask = ~special_token_lut[clean_targets]
            ce_loss_per_token = ce_loss_per_token * content_mask.float()
            content_count = content_mask.sum(dim=1).float().clamp(min=1.0)
            per_example_ce = ce_loss_per_token.sum(dim=1) / content_count
        else:
            per_example_ce = ce_loss_per_token.mean(dim=1)

        t_weights = 1.0 / t_values.squeeze(-1).clamp(min=1e-3)
        reweighted_per_example_loss = per_example_ce * t_weights
        total_loss = reweighted_per_example_loss.mean()

        metrics = {
            "loss": total_loss,
            "unweighted_ce": per_example_ce.mean()
        }

        return total_loss, metrics
