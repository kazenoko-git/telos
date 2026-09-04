"""
Autoregressive (AR) causal next-token prediction loss.
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
    def ar_loss_fn_mlx(model, batch_seqs, vocab_size, special_token_lut=None):
        logits = model(batch_seqs)  # [B, T, V]
        B, T, V = logits.shape

        shift_logits = logits[:, :-1, :].reshape(-1, V)
        shift_targets = batch_seqs[:, 1:].reshape(-1)

        ce_per_token = mx_nn.losses.cross_entropy(shift_logits, shift_targets, reduction="none").reshape(B, T - 1)
        
        if special_token_lut is not None:
            shift_target_2d = batch_seqs[:, 1:]
            content_mask = ~special_token_lut[shift_target_2d]
            ce_per_token = ce_per_token * content_mask
            content_count = mx.clip(mx.sum(content_mask, axis=1), 1.0, float(T - 1))
            per_example_ce = mx.sum(ce_per_token, axis=1) / content_count
            loss = mx.mean(per_example_ce)
        else:
            loss = mx.mean(ce_per_token)

        return loss, loss


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
    def ar_loss_fn_pytorch(
        logits: torch.Tensor,
        batch_seqs: torch.Tensor,
        special_token_lut: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict[str, float]]:
        batch_size, seq_len, vocab_size = logits.shape
        
        shift_logits = logits[:, :-1, :].contiguous().view(-1, vocab_size)
        shift_targets = batch_seqs[:, 1:].contiguous().view(-1)

        ce_loss_per_token = F.cross_entropy(shift_logits, shift_targets, reduction="none").view(batch_size, seq_len - 1)

        if special_token_lut is not None:
            shift_target_2d = batch_seqs[:, 1:]
            content_mask = ~special_token_lut[shift_target_2d]
            ce_loss_per_token = ce_loss_per_token * content_mask.float()
            content_count = content_mask.sum(dim=1).float().clamp(min=1.0)
            per_example_ce = ce_loss_per_token.sum(dim=1) / content_count
        else:
            per_example_ce = ce_loss_per_token.mean(dim=1)

        loss = per_example_ce.mean()

        metrics = {
            "loss": loss,
            "unweighted_ce": loss
        }

        return loss, metrics
