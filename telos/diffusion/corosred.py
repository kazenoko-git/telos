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
        logits = model(corrupted_seqs, mask_override=False, return_reliability=False)
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
        denoise_logits = model(corrupted, mask_override=False, return_reliability=False)
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
        Confidence-Routed Self-Conditioned Phase B Loss (MLX):
        Leverages the Learned Reliability Head to selectively mask low-confidence
        positions rather than uniform random tokens, training the bidirectional
        denoiser directly on model draft error patterns.
        """
        B, T = batch_seqs.shape

        use_self_cond = float(mx.random.uniform(shape=(1,))[0]) < self_cond_prob
        has_reliability = getattr(model, "use_reliability_head", False) and getattr(model, "reliability_head", None) is not None

        if use_self_cond or has_reliability:
            if has_reliability:
                # Pass 1: Causal forward pass to compute draft predictions and reliability scores
                causal_logits, raw_r_scores = model(batch_seqs, mask_override="causal", return_reliability=True)
                r_scores = mx.stop_gradient(raw_r_scores[:, :-1])
                # Prepend high confidence score for token 0 (BOS) so it is never masked
                r_scores_with_bos = mx.concatenate([mx.full((B, 1), 1e9), r_scores], axis=1)

                # Strictly enforce 15% total mask budget (e.g. 76 tokens for seq_len=512)
                total_k = int(T * mask_prob)
                k_targeted = int(total_k * 0.70)  # 70% targeted at model's lowest-confidence positions
                k_explore = total_k - k_targeted  # 30% uniform random exploration

                # 1. Target lowest k_targeted confidence tokens (stop_gradient prevents scatter VJP error)
                sorted_indices = mx.stop_gradient(mx.argsort(r_scores_with_bos, axis=-1))
                targeted_indices = sorted_indices[:, :k_targeted]
                mask_positions = mx.zeros((B, T), dtype=mx.bool_)
                row_indices = mx.broadcast_to(mx.arange(B)[:, None], (B, k_targeted))
                mask_positions[row_indices, targeted_indices] = True
                mask_positions[:, 0] = False

                # 2. Add k_explore uniform exploration positions
                rand_scores = mx.random.uniform(shape=(B, T))
                rand_scores_mod = mx.where(mask_positions, mx.full((B, T), -1e9), rand_scores)
                rand_scores_mod[:, 0] = -1e9
                rand_indices = mx.stop_gradient(mx.argsort(rand_scores_mod, axis=-1)[:, -k_explore:])
                row_indices_exp = mx.broadcast_to(mx.arange(B)[:, None], (B, k_explore))
                mask_positions[row_indices_exp, rand_indices] = True
            else:
                causal_logits = model(batch_seqs, mask_override="causal", return_reliability=False)
                mask_positions = mx.random.uniform(shape=(B, T)) < mask_prob
                mask_positions[:, 0] = False

            if use_self_cond:
                draft_preds = mx.stop_gradient(mx.argmax(causal_logits[:, :-1, :], axis=-1))
                draft_seqs = mx.concatenate([batch_seqs[:, :1], draft_preds], axis=1)
                base_seqs = draft_seqs
            else:
                base_seqs = batch_seqs
        else:
            base_seqs = batch_seqs
            mask_positions = mx.random.uniform(shape=(B, T)) < mask_prob
            mask_positions[:, 0] = False

        corrupted = mx.where(mask_positions, mx.full((B, T), mask_token_id, dtype=batch_seqs.dtype), base_seqs)

        # Forward pass in Bidirectional Attention Mode
        logits = model(corrupted, mask_override=False, return_reliability=False)
        logits_f32 = logits.reshape(-1, vocab_size)
        targets_flat = batch_seqs.reshape(-1)

        ce_all = mx_nn.losses.cross_entropy(logits_f32, targets_flat, reduction="none").reshape(B, T)
        masked_ce = ce_all * mask_positions

        denom = mx.clip(mx.sum(mask_positions, axis=1), 1.0, float(T))
        loss = mx.mean(mx.sum(masked_ce, axis=1) / denom)

        return loss, loss

    # Canonical alias for Phase C (Self-Conditioned Model Drafts + Confidence Routing)
    crsr_phase_c_loss_fn_mlx = crsr_phase_b_self_conditioned_loss_fn_mlx

    def corosred_unified_loss_fn_mlx(
        model,
        batch_seqs: mx.array,
        vocab_size: int,
        alpha: float = 0.85,
        beta: float = 0.15,
        gamma: float = 0.0,
        mask_token_id: int = 1,
        mask_prob: float = 0.15,
        k_amb: int = 5,
        causal_ratio: float = 0.75,
        special_token_lut: mx.array | None = None,
    ):
        """
        Unified Multi-Objective Loss Function (Apple Silicon / MLX):
        Executes causal next-token prediction on B_c sequences and bidirectional infilling on B_m sequences,
        normalizing across pooled tokens with schedule weights alpha(t), beta(t), and gamma(t).
        """
        B, T = batch_seqs.shape

        # 1. Partition heterogeneous microbatch: B_c causal + B_m infill
        # Compute exact integer split of causal vs infill sequences
        B_c = int(round(B * causal_ratio))
        # Ensure at least 1 causal sequence and at least 1 infill sequence when batch size > 1
        B_c = max(1, min(B_c, B - 1)) if B > 1 else 1
        B_m = B - B_c

        # Slice causal microbatch subset
        batch_causal = batch_seqs[:B_c]
        # Slice infill microbatch subset (or fall back to first sequence if B=1)
        batch_infill = batch_seqs[B_c:] if B_m > 0 else batch_seqs[:1]

        # 2. Causal Forward Pass (Exact causal slice)
        # Check if model has reliability head module enabled
        has_rel = getattr(model, "use_reliability_head", False) and getattr(model, "reliability_head", None) is not None

        if has_rel:
            # Forward pass returning causal logits and unnormalized reliability scores
            logits_c, raw_r_scores = model(batch_causal, mask_override="causal", return_reliability=True)
        else:
            # Forward pass returning causal logits only
            logits_c = model(batch_causal, mask_override="causal", return_reliability=False)
            raw_r_scores = None

        # Align causal logits with targets by shifting position by 1
        shift_logits_c = logits_c[:, :-1, :]
        shift_targets_c = batch_causal[:, 1:]

        # Compute per-token cross entropy loss on causal slice
        ce_c_all = mx_nn.losses.cross_entropy(
            shift_logits_c.reshape(-1, vocab_size),
            shift_targets_c.reshape(-1),
            reduction="none"
        )
        sum_ce_c = mx.sum(ce_c_all)
        n_causal_tokens = float(B_c * (T - 1))

        # 3. Auxiliary Reliability Head Loss (with Ambiguity Exclusion)
        if has_rel and raw_r_scores is not None:
            # Align reliability prediction scores with shifted target tokens
            shift_r_scores = raw_r_scores[:, :-1]
            # Identify model greedy predictions
            argmax_indices = mx.argmax(shift_logits_c, axis=-1)
            is_exact_match = (argmax_indices == shift_targets_c)

            # Determine top-k candidate tokens to detect ambiguous cases
            top_k_indices = mx.argpartition(shift_logits_c, -k_amb, axis=-1)[..., -k_amb:]
            expanded_targets = mx.expand_dims(shift_targets_c, -1)
            is_target_in_top_k = mx.any(top_k_indices == expanded_targets, axis=-1)

            # Assign ground truth labels: 1.0 if greedy prediction matches target, 0.0 otherwise
            labels = mx.where(is_exact_match, mx.ones_like(shift_r_scores), mx.zeros_like(shift_r_scores))
            # Compute raw binary cross entropy loss with logits
            bce_raw = mx_nn.losses.binary_cross_entropy(shift_r_scores, labels, with_logits=True)

            # Ambiguity exclusion: filter out positions where target is in top-k but not rank 1
            is_ambiguous = mx.logical_and(is_target_in_top_k, mx.logical_not(is_exact_match))
            valid_mask = mx.logical_not(is_ambiguous)
            if special_token_lut is not None:
                valid_mask = mx.logical_and(valid_mask, ~special_token_lut[shift_targets_c])

            # Mask out ambiguous and special tokens from BCE loss
            valid_mask_f32 = valid_mask.astype(mx.float32)
            masked_bce = bce_raw * valid_mask_f32
            valid_count = mx.clip(mx.sum(valid_mask_f32), 1.0, float(B_c * (T - 1)))
            l_head = mx.sum(masked_bce) / valid_count
        else:
            l_head = mx.array(0.0)

        # 4. Infill Forward Pass (Exact infill slice)
        if B_m > 0:
            # Generate uniform random mask positions for infill sequences
            rand_probs = mx.random.uniform(shape=(B_m, T))
            mask_positions = rand_probs < mask_prob
            # Never mask sequence start BOS token
            mask_positions[:, 0] = False

            # Replace masked token positions with mask token ID
            corrupted_m = mx.where(mask_positions, mx.full((B_m, T), mask_token_id, dtype=batch_infill.dtype), batch_infill)
            # Forward pass in bidirectional attention mode
            logits_m = model(corrupted_m, mask_override=False, return_reliability=False)

            # Compute cross entropy on all tokens and select only masked positions
            ce_m_all = mx_nn.losses.cross_entropy(
                logits_m.reshape(-1, vocab_size),
                batch_infill.reshape(-1),
                reduction="none"
            ).reshape(B_m, T)
            masked_ce_m = ce_m_all * mask_positions
            sum_ce_m = mx.sum(masked_ce_m)
            n_infill_tokens = mx.clip(mx.sum(mask_positions.astype(mx.float32)), 1.0, float(B_m * T))
        else:
            sum_ce_m = mx.array(0.0)
            n_infill_tokens = mx.array(1.0)

        # 5. Unified Pooled Normalization
        # Combines causal and infill token sums divided by weighted evaluated token counts
        pooled_numerator = alpha * sum_ce_c + beta * sum_ce_m
        pooled_denominator = alpha * n_causal_tokens + beta * n_infill_tokens
        pooled_ce = pooled_numerator / pooled_denominator
        total_loss = pooled_ce + gamma * l_head

        return total_loss, pooled_ce


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

        # Phase B: 15% Uniform Random Masking on clean ground-truth tokens (never masking BOS at index 0)
        rand_probs = torch.rand((B, T), device=batch_seqs.device)
        mask_positions = (rand_probs < mask_prob)
        mask_positions = torch.cat([torch.zeros((B, 1), device=batch_seqs.device, dtype=torch.bool), mask_positions[:, 1:]], dim=1)

        # Ensure at least one token is masked per sequence as fallback
        no_mask = ~mask_positions.any(dim=1, keepdim=True)
        fallback = (torch.rand((B, T), device=batch_seqs.device) < mask_prob)
        fallback = torch.cat([torch.zeros((B, 1), device=batch_seqs.device, dtype=torch.bool), fallback[:, 1:]], dim=1)
        mask_positions = torch.where(no_mask, fallback, mask_positions)

        if hasattr(torch, "clear_autocast_cache"):
            torch.clear_autocast_cache()

        corrupted_seqs = torch.where(
            mask_positions,
            torch.full((B, T), mask_token_id, dtype=batch_seqs.dtype, device=batch_seqs.device),
            batch_seqs
        )

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
        Confidence-Routed Self-Conditioned Phase B Loss (PyTorch & TPU-safe):
        Leverages the Learned Reliability Head (LRH) to selectively mask low-confidence
        draft positions (r_scores < 0) rather than blindly masking uniform random tokens.
        Eliminates the train/test distribution mismatch by training the bidirectional
        denoiser directly on the model's actual draft error patterns.
        Fully vectorized without host synchronization (.item()) for zero-stall Cloud TPU execution.
        """
        B, T = batch_seqs.shape

        raw_model = getattr(model, "module", model)
        has_reliability = (
            getattr(raw_model, "reliability_head", None) is not None
            or getattr(getattr(raw_model, "config", None), "use_reliability_head", False)
        )

        with torch.no_grad():
            if self_cond_prob > 0.0 or has_reliability:
                causal_out = raw_model(batch_seqs, return_reliability=has_reliability, mask_override=True)
                if has_reliability and isinstance(causal_out, tuple):
                    causal_logits, raw_r_scores = causal_out
                else:
                    causal_logits = causal_out
                    raw_r_scores = None

                draft_preds = torch.argmax(causal_logits[:, :-1, :], dim=-1)
                del causal_logits
                draft_seqs = torch.cat([batch_seqs[:, :1], draft_preds], dim=1)

                if self_cond_prob > 0.0:
                    rand_sc = torch.rand((B, 1), device=batch_seqs.device) < self_cond_prob
                    base_seqs = torch.where(rand_sc, draft_seqs, batch_seqs)
                else:
                    base_seqs = batch_seqs
            else:
                base_seqs = batch_seqs
                raw_r_scores = None

        if hasattr(torch, "clear_autocast_cache"):
            torch.clear_autocast_cache()

        # Determine mask positions:
        # If the Learned Reliability Head is available, route masks to low-confidence positions!
        if raw_r_scores is not None:
            # Shift r_scores to align with next-token prediction
            r_scores = raw_r_scores[:, :-1]
            r_scores_with_bos = torch.cat([torch.full((B, 1), 1e9, device=batch_seqs.device, dtype=r_scores.dtype), r_scores], dim=1)

            # Strictly enforce 15% total mask budget (e.g. 76 tokens for seq_len=512)
            total_k = int(T * mask_prob)
            k_targeted = int(total_k * 0.70)  # 70% targeted at lowest-confidence positions
            k_explore = total_k - k_targeted  # 30% uniform random exploration

            # 1. Target lowest k_targeted confidence tokens
            sorted_indices = torch.argsort(r_scores_with_bos, dim=-1)
            targeted_indices = sorted_indices[:, :k_targeted]
            mask_positions = torch.zeros((B, T), device=batch_seqs.device, dtype=torch.bool)
            mask_positions.scatter_(1, targeted_indices, True)
            mask_positions[:, 0] = False

            # 2. Add k_explore uniform exploration positions
            rand_scores = torch.rand((B, T), device=batch_seqs.device)
            rand_scores = torch.where(mask_positions, torch.full((B, T), -1e9, device=batch_seqs.device), rand_scores)
            rand_scores[:, 0] = -1e9
            rand_indices = torch.argsort(rand_scores, dim=-1)[:, -k_explore:]
            mask_positions.scatter_(1, rand_indices, True)
        else:
            rand_probs = torch.rand((B, T), device=batch_seqs.device)
            mask_positions = (rand_probs < mask_prob)
            mask_positions = torch.cat([torch.zeros((B, 1), device=batch_seqs.device, dtype=torch.bool), mask_positions[:, 1:]], dim=1)

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

    # Canonical alias for Phase C (Self-Conditioned Model Drafts + Confidence Routing)
    crsr_phase_c_loss_fn_pytorch = crsr_phase_b_self_conditioned_loss_fn_pytorch

