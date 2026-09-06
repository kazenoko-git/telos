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

        shift_logits_f32 = shift_logits.reshape(-1, vocab_size)
        shift_targets_flat = shift_targets.reshape(-1)
        ar_loss = mx.mean(mx_nn.losses.cross_entropy(shift_logits_f32, shift_targets_flat, reduction="none"))

        valid_count = mx.clip(mx.sum(valid_mask_f32, axis=1), 1.0, float(T - 1))
        per_example_loss = mx.sum(masked_bce, axis=1) / valid_count

        r_loss = mx.mean(per_example_loss)
        total_loss = ar_loss + r_loss
        return total_loss, ar_loss

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

    def crsr_expected_gain_loss_fn_mlx(
        model,
        batch_seqs,
        vocab_size: int,
        mask_token_id: int,
        mask_prob: float = 0.15
    ):
        """
        Expected Gain Router Loss (MLX):
        Trains the router to predict the scalar cross-entropy reduction (expected gain)
        achieved when a token is infilled by the bidirectional denoiser.
        """
        B, T = batch_seqs.shape

        # Step 1: Causal forward pass with router head output
        causal_logits, raw_gain_scores = model(batch_seqs, mask_override="causal", return_reliability=True)
        shift_causal = causal_logits[:, :-1, :].reshape(-1, vocab_size)
        shift_gain_scores = raw_gain_scores[:, :-1]
        shift_targets = batch_seqs[:, 1:].reshape(-1)

        # Compute causal cross entropy loss before refinement
        ce_before_flat = mx_nn.losses.cross_entropy(shift_causal, shift_targets, reduction="none")
        ce_before = ce_before_flat.reshape(B, T - 1)

        # Step 2: Bidirectional pass on masked sequence
        rand_probs = mx.random.uniform(shape=(B, T))
        mask_positions = rand_probs < mask_prob
        mask_shift = mask_positions[:, 1:]

        corrupted = mx.where(mask_positions, mx.full((B, T), mask_token_id, dtype=batch_seqs.dtype), batch_seqs)
        denoise_logits = model(corrupted, mask_override=None, return_reliability=False)
        shift_denoise = denoise_logits[:, 1:, :].reshape(-1, vocab_size)

        ce_after_flat = mx_nn.losses.cross_entropy(shift_denoise, shift_targets, reduction="none")
        ce_after = ce_after_flat.reshape(B, T - 1)

        # Step 3: Compute actual cross-entropy reduction (empirical gain) on masked positions
        # Gain is clamped at 0.0 (negative gain means refinement made it worse -> 0 expected reward)
        empirical_gain = mx.clip(ce_before - ce_after, a_min=0.0, a_max=10.0)

        # Step 4: Huber / Smooth L1 regression for router predictions
        diff = shift_gain_scores - empirical_gain
        abs_diff = mx.abs(diff)
        huber = mx.where(abs_diff < 1.0, 0.5 * (diff ** 2), abs_diff - 0.5)

        masked_huber = huber * mask_shift
        denom = mx.clip(mx.sum(mask_shift), 1.0, float(B * (T - 1)))
        loss = mx.sum(masked_huber) / denom

        return loss, loss

    def crsr_phase_b_self_conditioned_loss_fn_mlx(
        model,
        batch_seqs,
        vocab_size: int,
        mask_token_id: int,
        mask_prob: float = 0.15,
        self_cond_prob: float = 0.5
    ):
        """
        Self-Conditioned Phase B Loss (MLX):
        With probability `self_cond_prob`, trains the bidirectional denoiser on
        model-generated draft sequences with errors rather than 100% clean sequences,
        eliminating the train/test distribution mismatch.
        """
        B, T = batch_seqs.shape

        # Sample random coin flip for self-conditioning branch
        use_self_cond = float(mx.random.uniform(shape=(1,))[0]) < self_cond_prob

        rand_probs = mx.random.uniform(shape=(B, T))
        mask_positions = rand_probs < mask_prob

        if use_self_cond:
            # Generate causal draft tokens
            causal_logits = model(batch_seqs, mask_override="causal", return_reliability=False)
            draft_preds = mx.argmax(causal_logits[:, :-1, :], axis=-1)
            draft_seqs = mx.concatenate([batch_seqs[:, :1], draft_preds], axis=1)

            # Corrupt the draft sequence at masked positions
            corrupted = mx.where(mask_positions, mx.full((B, T), mask_token_id, dtype=batch_seqs.dtype), draft_seqs)
        else:
            corrupted = mx.where(mask_positions, mx.full((B, T), mask_token_id, dtype=batch_seqs.dtype), batch_seqs)

        # Forward pass in Bidirectional Attention Mode
        logits = model(corrupted, mask_override=None, return_reliability=False)
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
            num_greater = (detached_logits > target_logits).float().sum(dim=-1)
            is_target_in_top_k = (num_greater < k_amb)

            labels = is_exact_match.float()

        bce_raw = F.binary_cross_entropy_with_logits(shift_r_scores, labels, reduction="none")

        is_ambiguous = is_target_in_top_k & (~is_exact_match)
        valid_mask = ~is_ambiguous

        if special_token_lut is not None:
            # Special tokens in Télos tokenizer are IDs 0, 1, 2, 3 (PAD, BOS, EOS, MASK).
            # Direct comparison avoids unpartitioned 1D tensor lookup indexing across SPMD shards.
            content_mask = (shift_targets >= 4)
            valid_mask = valid_mask & content_mask

        # 1. Autoregressive Causal Language Modeling Loss (trains the backbone model)
        ar_ce = F.cross_entropy(shift_logits.reshape(-1, vocab_size), shift_targets.reshape(-1))

        # 2. Reliability Head Binary Cross Entropy Loss (trains the reliability head)
        masked_bce = bce_raw * valid_mask.float()
        valid_count = valid_mask.sum(dim=1).float().clamp(min=1.0)
        per_example_r_loss = masked_bce.sum(dim=1) / valid_count
        r_loss = per_example_r_loss.mean()

        total_loss = ar_ce + r_loss
        metrics = {"loss": total_loss, "unweighted_ce": ar_ce, "r_loss": r_loss}
        return total_loss, metrics


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

    def crsr_expected_gain_loss_fn_pytorch(
        model,
        batch_seqs: torch.Tensor,
        vocab_size: int,
        mask_token_id: int,
        mask_prob: float = 0.15
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Expected Gain Router Loss (PyTorch):
        Trains the router head to predict scalar cross-entropy reduction (expected gain)
        achieved when a token is infilled by the bidirectional denoiser.
        """
        B, T = batch_seqs.shape

        # Step 1: Causal forward pass with router head output
        causal_logits, raw_gain_scores = model(batch_seqs, return_reliability=True, mask_override=True)
        shift_causal = causal_logits[:, :-1, :].reshape(-1, vocab_size)
        shift_gain_scores = raw_gain_scores[:, :-1]
        shift_targets = batch_seqs[:, 1:].reshape(-1)

        with torch.no_grad():
            ce_before = F.cross_entropy(shift_causal, shift_targets, reduction="none").view(B, T - 1)

            # Step 2: Bidirectional pass on masked sequence
            rand_probs = torch.rand((B, T), device=batch_seqs.device)
            mask_positions = (rand_probs < mask_prob)
            mask_shift = mask_positions[:, 1:]

            corrupted = torch.where(
                mask_positions,
                torch.full((B, T), mask_token_id, dtype=batch_seqs.dtype, device=batch_seqs.device),
                batch_seqs
            )
            denoise_logits = model(corrupted, return_reliability=False, mask_override=False)
            shift_denoise = denoise_logits[:, 1:, :].reshape(-1, vocab_size)

            ce_after = F.cross_entropy(shift_denoise, shift_targets, reduction="none").view(B, T - 1)

            # Empirical gain: clamped to min 0.0 (no negative reward for unfixable tokens)
            empirical_gain = torch.clamp(ce_before - ce_after, min=0.0, max=10.0)

        # Step 3: Huber / Smooth L1 loss on masked positions
        huber = F.smooth_l1_loss(shift_gain_scores, empirical_gain, reduction="none")

        masked_huber = huber * mask_shift.float()
        denom = mask_shift.sum().float().clamp(min=1.0)
        loss = masked_huber.sum() / denom

        metrics = {
            "loss": loss.item() if hasattr(loss, "item") else float(loss),
            "mean_gain": float(empirical_gain[mask_shift].mean().item()) if mask_shift.any() else 0.0,
            "pred_gain": float(shift_gain_scores[mask_shift].mean().item()) if mask_shift.any() else 0.0,
        }
        return loss, metrics

    def crsr_phase_b_self_conditioned_loss_fn_pytorch(
        model,
        batch_seqs: torch.Tensor,
        vocab_size: int,
        mask_token_id: int,
        mask_prob: float = 0.15,
        self_cond_prob: float = 0.5
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Self-Conditioned Phase B Loss (PyTorch & TPU-safe):
        Trains the bidirectional denoiser on model-generated draft sequences with errors,
        eliminating the train/test distribution mismatch. Fully vectorized without host
        synchronization (.item()) for zero-stall Cloud TPU execution.
        """
        B, T = batch_seqs.shape

        rand_probs = torch.rand((B, T), device=batch_seqs.device)
        mask_positions = (rand_probs < mask_prob)

        if self_cond_prob > 0.0:
            # Vectorized on-device coin flip per sequence (no .item() host barrier)
            rand_sc = torch.rand((B, 1), device=batch_seqs.device) < self_cond_prob
            with torch.no_grad():
                causal_logits = model(batch_seqs, return_reliability=False, mask_override=True)
                draft_preds = torch.argmax(causal_logits[:, :-1, :], dim=-1)
                draft_seqs = torch.cat([batch_seqs[:, :1], draft_preds], dim=1)

            base_seqs = torch.where(rand_sc, draft_seqs, batch_seqs)
        else:
            base_seqs = batch_seqs

        corrupted_seqs = torch.where(
            mask_positions,
            torch.full((B, T), mask_token_id, dtype=batch_seqs.dtype, device=batch_seqs.device),
            base_seqs
        )

        # Forward pass in Bidirectional Attention Mode
        logits = model(corrupted_seqs, return_reliability=False, mask_override=False)
        logits_f32 = logits.view(-1, vocab_size)
        targets_flat = batch_seqs.view(-1)

        ce_all = F.cross_entropy(logits_f32, targets_flat, reduction="none").view(B, T)
        masked_ce = ce_all * mask_positions.float()

        denom = mask_positions.sum(dim=1).float().clamp(min=1.0)
        loss = (masked_ce.sum(dim=1) / denom).mean()

        metrics = {
            "loss": loss,
            "unweighted_ce": loss,
            "self_cond": float(self_cond_prob),
            "self_cond_prob": float(self_cond_prob)
        }
        return loss, metrics

