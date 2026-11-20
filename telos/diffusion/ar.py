"""
Autoregressive (AR) causal next-token prediction loss.
Contains implementations for both MLX and PyTorch.
"""

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
    def ar_loss_fn_mlx(model, batch_seqs, vocab_size, special_token_lut=None):
        B, T = batch_seqs.shape
        shift_targets_2d = batch_seqs[:, 1:]
        if getattr(model, "fused_vocab_loss_enabled", False):
            ce_per_token = model.output_cross_entropy(
                batch_seqs, shift_targets_2d,
                chunk_size=getattr(model, "fused_vocab_chunk_size", 256),
                next_token_only=True,
            )
        else:
            # Skip the final vocabulary projection: its logits have no
            # next-token target and slicing after projection materializes a
            # large non-contiguous reshape on MLX.
            if hasattr(model, "next_token_logits"):
                logits = model.next_token_logits(batch_seqs)  # [B, T - 1, V]
            else:
                logits = model(batch_seqs)[:, :-1, :]  # [B, T - 1, V]
            V = logits.shape[-1]
            shift_logits = logits.reshape(-1, V)
            shift_targets = shift_targets_2d.reshape(-1)
            ce_per_token = cross_entropy(shift_logits, shift_targets, reduction="none").reshape(B, T - 1)
        
        if special_token_lut is not None:
            content_mask = ~special_token_lut[shift_targets_2d]
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
        
        if seq_len == batch_seqs.shape[1] - 1:
            shift_logits = logits.reshape(-1, vocab_size)
        elif seq_len == batch_seqs.shape[1]:
            shift_logits = logits[:, :-1, :].contiguous().view(-1, vocab_size)
        else:
            raise ValueError("AR logits must contain T or T-1 positions")
        shift_targets = batch_seqs[:, 1:].contiguous().view(-1)
        loss = F.cross_entropy(shift_logits, shift_targets, reduction="mean")

        metrics = {
            "loss": loss,
            "unweighted_ce": loss
        }

        return loss, metrics
