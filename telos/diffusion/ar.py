"""
Autoregressive (AR) causal next-token prediction loss.
Contains implementations for both MLX and PyTorch.
"""

# =========================================================================
# MLX IMPLEMENTATION
# =========================================================================

try:
    import mlx.core as mx
    from telos.training.core import metal_usable
    MLX_AVAILABLE = metal_usable()
    if MLX_AVAILABLE:
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
                batch_seqs[:, :-1], shift_targets_2d,
                chunk_size=getattr(model, "fused_vocab_chunk_size", 256),
            )
        else:
            logits = model(batch_seqs)  # [B, T, V]
            V = logits.shape[-1]
            shift_logits = logits[:, :-1, :].reshape(-1, V)
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
        
        shift_logits = logits[:, :-1, :].contiguous().view(-1, vocab_size)
        shift_targets = batch_seqs[:, 1:].contiguous().view(-1)
        total_tokens = shift_targets.numel()

        # Chunk CE on CUDA to save memory; use fused unchunked CE on XLA
        chunk_size = 4096
        is_xla = logits.device.type == "xla" or str(logits.device).startswith("xla")
        if not is_xla and total_tokens > chunk_size:
            ce_chunks = []
            for i in range(0, total_tokens, chunk_size):
                chunk_l = shift_logits[i : i + chunk_size]
                chunk_t = shift_targets[i : i + chunk_size]
                ce_chunks.append(F.cross_entropy(chunk_l, chunk_t, reduction="none"))
            ce_loss_per_token = torch.cat(ce_chunks, dim=0).view(batch_size, seq_len - 1)
        else:
            ce_loss_per_token = F.cross_entropy(shift_logits, shift_targets, reduction="none").view(batch_size, seq_len - 1)

        # Mean cross-entropy across all tokens
        loss = ce_loss_per_token.mean()

        metrics = {
            "loss": loss,
            "unweighted_ce": loss
        }

        return loss, metrics
