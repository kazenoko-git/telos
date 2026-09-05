"""
COROSred losses (Phase A and Phase B).
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
    def crsr_phase_a_loss_fn_mlx(model, batch_seqs, vocab_size, special_token_lut=None, k_amb: int = 5):
        """
        Phase A: Binary Cross Entropy for Reliability Head using Ambiguity Exclusion.
        """
        B, T = batch_seqs.shape

        # Forward pass under causal mask
        logits, raw_r_scores = model(batch_seqs, mask_override="causal", return_reliability=True)

        shift_logits = logits[:, :-1, :]
        shift_r_scores = raw_r_scores[:, :-1]
        shift_targets = batch_seqs[:, 1:]

        argmax_indices = mx.argmax(shift_logits, axis=-1)
        is_exact_match = (argmax_indices == shift_targets)

        top_k_indices = mx.argpartition(shift_logits, -k_amb, axis=-1)[..., -k_amb:]
        expanded_targets = mx.expand_dims(shift_targets, -1)
        is_target_in_top_k = mx.any(top_k_indices == expanded_targets, axis=-1)

        labels = mx.where(is_exact_match, mx.ones_like(shift_r_scores), mx.zeros_like(shift_r_scores))

        bce_raw = mx_nn.losses.binary_cross_entropy(shift_r_scores, labels, with_logits=True)

        is_ambiguous = mx.logical_and(is_target_in_top_k, mx.logical_not(is_exact_match))
        valid_mask = mx.logical_not(is_ambiguous)

        if special_token_lut is not None:
            content_mask = ~special_token_lut[shift_targets]
            valid_mask = mx.logical_and(valid_mask, content_mask)

        valid_mask_f32 = valid_mask
        masked_bce = bce_raw * valid_mask_f32

        valid_count = mx.clip(mx.sum(valid_mask_f32, axis=1), 1.0, float(T - 1))
        per_example_loss = mx.sum(masked_bce, axis=1) / valid_count

        loss = mx.mean(per_example_loss)
        return loss, loss

    def crsr_phase_b_loss_fn_mlx(model, batch_seqs, vocab_size, mask_token_id: int, mask_prob: float = 0.15):
        """
        Phase B: Bidirectional Masked Denoising Language Model (MDLM) loss.
        """
        B, T = batch_seqs.shape

        rand_probs = mx.random.uniform(shape=(B, T))
        mask_positions = rand_probs < mask_prob

        corrupted_seqs = mx.where(mask_positions, mx.full((B, T), mask_token_id, dtype=batch_seqs.dtype), batch_seqs)

        # Forward pass in Bidirectional Attention Mode
        logits = model(corrupted_seqs, mask_override=None, return_reliability=False)
        logits_f32 = logits.reshape(-1, vocab_size)
        targets_flat = batch_seqs.reshape(-1)

        ce_all = mx_nn.losses.cross_entropy(logits_f32, targets_flat, reduction="none").reshape(B, T)
        masked_ce = ce_all * mask_positions

        denom = mx.clip(mx.sum(mask_positions, axis=1), 1.0, float(T))
        loss = mx.mean(mx.sum(masked_ce, axis=1) / denom)

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
    def crsr_phase_a_loss_fn_pytorch(
        model,
        batch_seqs: torch.Tensor,
        vocab_size: int,
        special_token_lut: torch.Tensor | None = None,
        k_amb: int = 5
    ) -> tuple[torch.Tensor, dict[str, float]]:
        B, T = batch_seqs.shape

        # Forward pass under causal mask
        logits, raw_r_scores = model(batch_seqs, return_reliability=True, mask_override=True)

        shift_logits = logits[:, :-1, :]
        shift_r_scores = raw_r_scores[:, :-1]
        shift_targets = batch_seqs[:, 1:]

        # Explicitly detach logits used for label generation to prevent XLA from tracing
        # a massive, non-differentiable graph branch during backward.
        with torch.no_grad():
            detached_logits = shift_logits.detach()
            
            argmax_indices = detached_logits.argmax(dim=-1)
            is_exact_match = (argmax_indices == shift_targets)

            # PyTorch XLA Optimization: torch.topk forces a CPU fallback or brutal brute-force 
            # sort on TPU VM which causes Eigen ThreadPool SIGSEGV in PJRT. 
            # Instead, we check if the target is in the top-k by counting how many elements 
            # have a strictly greater logit score.
            expanded_targets = shift_targets.unsqueeze(-1)
            target_logits = detached_logits.gather(dim=-1, index=expanded_targets)
            
            # Count elements strictly greater than the target's logit
            num_greater = (detached_logits > target_logits).sum(dim=-1)
            is_target_in_top_k = (num_greater < k_amb)

            labels = is_exact_match.float()

        bce_raw = F.binary_cross_entropy_with_logits(shift_r_scores, labels, reduction="none")

        is_ambiguous = is_target_in_top_k & (~is_exact_match)
        valid_mask = ~is_ambiguous

        if special_token_lut is not None:
            content_mask = ~special_token_lut[shift_targets]
            valid_mask = valid_mask & content_mask

        masked_bce = bce_raw * valid_mask.float()
        valid_count = valid_mask.sum(dim=1).float().clamp(min=1.0)
        per_example_loss = masked_bce.sum(dim=1) / valid_count
        loss = per_example_loss.mean()

        metrics = {"loss": loss, "unweighted_ce": loss}
        return loss, metrics


    def crsr_phase_b_loss_fn_pytorch(
        model,
        batch_seqs: torch.Tensor,
        vocab_size: int,
        mask_token_id: int,
        mask_prob: float = 0.15
    ) -> tuple[torch.Tensor, dict[str, float]]:
        B, T = batch_seqs.shape

        rand_probs = torch.rand((B, T), device=batch_seqs.device)
        mask_positions = rand_probs < mask_prob

        corrupted_seqs = torch.where(mask_positions, torch.full((B, T), mask_token_id, dtype=batch_seqs.dtype, device=batch_seqs.device), batch_seqs)

        # Forward pass in Bidirectional Attention Mode
        logits = model(corrupted_seqs, return_reliability=False, mask_override=False)
        logits_f32 = logits.view(-1, vocab_size)
        targets_flat = batch_seqs.view(-1)

        ce_all = F.cross_entropy(logits_f32, targets_flat, reduction="none").view(B, T)
        masked_ce = ce_all * mask_positions.float()

        denom = mask_positions.sum(dim=1).float().clamp(min=1.0)
        loss = (masked_ce.sum(dim=1) / denom).mean()

        metrics = {"loss": loss, "unweighted_ce": loss}
        return loss, metrics
