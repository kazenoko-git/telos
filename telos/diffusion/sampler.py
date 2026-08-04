"""Iterative Denoising Samplers for Masked Diffusion Inference.

Includes:
- MDLMSampler: Standard confidence-based unmasking sampler (linear / cosine schedules).
- NonMonotonicMDLMSampler: Non-monotonic discrete diffusion sampler with dynamic re-masking.
- WindowedMDLMSampler: Progressive windowed infill sampler.
"""

import math
import torch
import torch.nn.functional as F


class MDLMSampler:
    """Iterative confidence-based unmasking sampler (Cosine & Linear schedules)."""

    def __init__(
        self,
        model: torch.nn.Module,
        mask_token_id: int,
        num_steps: int = 64,
        temperature: float = 1.0,
        repetition_penalty: float = 1.2,
        schedule: str = "cosine"
    ):
        """
        Args:
            model: trained TelosTransformer model.
            mask_token_id: token ID used for [MASK].
            num_steps: speed vs quality knob (e.g. 16–128).
            temperature: sampling temperature.
            repetition_penalty: penalty factor (e.g. 1.2) for already unmasked tokens.
            schedule: "cosine" or "linear".
        """
        self.model = model
        self.mask_token_id = mask_token_id
        self.num_steps = num_steps
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.schedule = schedule

    def _get_num_to_unmask(self, step: int, total_masked: int) -> int:
        """Determines how many masked tokens should be unmasked at current step."""
        if self.schedule == "cosine":
            progress = (step + 1) / self.num_steps
            ratio = 1.0 - math.cos(progress * math.pi / 2.0)
            target_unmasked = math.ceil(ratio * total_masked)
        else:
            target_unmasked = math.ceil(((step + 1) / self.num_steps) * total_masked)
            
        return min(target_unmasked, total_masked)

    @torch.no_grad()
    def sample(
        self,
        seq_len: int,
        prompt_ids: torch.Tensor | None = None,
        device: str | torch.device = "cpu"
    ) -> torch.Tensor:
        self.model.eval()

        # Initialize sequence: start with prompt followed by [MASK] tokens
        seq = torch.full((1, seq_len), self.mask_token_id, dtype=torch.long, device=device)
        
        prompt_len = 0
        if prompt_ids is not None:
            prompt_len = prompt_ids.shape[1]
            seq[:, :prompt_len] = prompt_ids

        total_masked_positions = seq_len - prompt_len
        if total_masked_positions <= 0:
            return seq

        already_unmasked_count = 0

        for step in range(self.num_steps):
            current_mask = (seq == self.mask_token_id)
            current_mask[:, :prompt_len] = False  # protect prompt

            num_currently_masked = current_mask.sum().item()
            if num_currently_masked == 0:
                break

            target_unmasked_count = self._get_num_to_unmask(step, total_masked_positions)
            num_to_unmask_this_step = target_unmasked_count - already_unmasked_count

            if num_to_unmask_this_step <= 0:
                continue

            logits = self.model(seq)
            logits = logits.clone()
            logits[:, :, self.mask_token_id] = -float("inf")

            if self.repetition_penalty != 1.0:
                gen_seq = seq[:, prompt_len:] if prompt_len is not None else seq
                unmasked_tokens = gen_seq[(gen_seq != self.mask_token_id) & (gen_seq > 3)]
                for tok_id in set(unmasked_tokens.tolist()):
                    logits[:, :, tok_id] = torch.where(
                        logits[:, :, tok_id] > 0,
                        logits[:, :, tok_id] / self.repetition_penalty,
                        logits[:, :, tok_id] * self.repetition_penalty
                    )

            scaled_logits = logits / max(self.temperature, 1e-5)
            probs = F.softmax(scaled_logits, dim=-1)

            top2_probs, top2_indices = torch.topk(probs, k=2, dim=-1)
            margin_conf = top2_probs[:, :, 0] - top2_probs[:, :, 1]
            argmax_tokens = top2_indices[:, :, 0]

            if self.temperature > 0.05:
                gumbel_noise = -torch.log(-torch.log(torch.rand_like(margin_conf) + 1e-8) + 1e-8)
                confidence_scores = margin_conf + 0.1 * self.temperature * gumbel_noise
            else:
                confidence_scores = margin_conf.clone()

            confidence_scores[~current_mask] = -float("inf")

            k = min(num_to_unmask_this_step, num_currently_masked)
            _, topk_indices = torch.topk(confidence_scores[0], k=k)

            if self.temperature > 0.05:
                selected_probs = probs[0, topk_indices]
                sampled_tokens = torch.multinomial(selected_probs, num_samples=1).squeeze(-1)
            else:
                sampled_tokens = argmax_tokens[0, topk_indices]

            seq[0, topk_indices] = sampled_tokens
            already_unmasked_count += k

        return seq


class NonMonotonicMDLMSampler:
    """Non-monotonic discrete diffusion sampler featuring dynamic re-masking."""

    def __init__(
        self,
        model: torch.nn.Module,
        mask_token_id: int,
        num_steps: int = 64,
        temperature: float = 0.0,
        repetition_penalty: float = 1.0,
        schedule: str = "cosine",
        remask_threshold: float = 0.15
    ):
        self.model = model
        self.mask_token_id = mask_token_id
        self.num_steps = num_steps
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.schedule = schedule
        self.remask_threshold = remask_threshold

    def _get_target_unmasked_count(self, step: int, total_masked: int) -> int:
        if self.schedule == "cosine":
            progress = (step + 1) / self.num_steps
            ratio = 1.0 - math.cos(progress * math.pi / 2.0)
            return math.ceil(ratio * total_masked)
        else:
            return math.ceil(((step + 1) / self.num_steps) * total_masked)

    @torch.no_grad()
    def sample(
        self,
        seq_len: int,
        prompt_ids: torch.Tensor | None = None,
        device: str | torch.device = "cpu"
    ) -> torch.Tensor:
        self.model.eval()

        seq = torch.full((1, seq_len), self.mask_token_id, dtype=torch.long, device=device)

        prompt_len = 0
        if prompt_ids is not None:
            prompt_len = prompt_ids.shape[1]
            seq[:, :prompt_len] = prompt_ids

        total_gen_len = seq_len - prompt_len
        if total_gen_len <= 0:
            return seq

        for step in range(self.num_steps):
            is_final_step = (step == self.num_steps - 1)

            logits = self.model(seq).clone()
            logits[:, :, self.mask_token_id] = -float("inf")

            scaled_logits = logits / max(self.temperature, 1e-5)
            probs = F.softmax(scaled_logits, dim=-1)

            top2_probs, top2_indices = torch.topk(probs, k=2, dim=-1)
            margins = top2_probs[0, :, 0] - top2_probs[0, :, 1]
            top1_tokens = top2_indices[0, :, 0]

            if not is_final_step and step > 0:
                unmasked_gen_positions = (seq[0] != self.mask_token_id)
                unmasked_gen_positions[:prompt_len] = False

                token_changed = (seq[0] != top1_tokens) & unmasked_gen_positions
                low_confidence = (margins < self.remask_threshold) & unmasked_gen_positions
                remask_mask = token_changed | low_confidence

                if remask_mask.any():
                    seq[0, remask_mask] = self.mask_token_id

            current_mask = (seq[0] == self.mask_token_id)
            current_mask[:prompt_len] = False

            num_currently_masked = current_mask.sum().item()
            if num_currently_masked == 0 and not is_final_step:
                continue

            target_unmasked = self._get_target_unmasked_count(step, total_gen_len)
            current_unmasked = total_gen_len - num_currently_masked
            num_to_unmask = max(1, target_unmasked - current_unmasked)

            if is_final_step:
                seq[0, current_mask] = top1_tokens[current_mask]
                break

            if self.temperature > 0.05:
                gumbel_noise = -torch.log(-torch.log(torch.rand_like(margins) + 1e-8) + 1e-8)
                scores = margins + 0.1 * self.temperature * gumbel_noise
            else:
                scores = margins.clone()

            scores[~current_mask] = -float("inf")

            k = min(num_to_unmask, num_currently_masked)
            if k > 0:
                _, topk_indices = torch.topk(scores, k=k)

                if self.temperature > 0.05:
                    selected_probs = probs[0, topk_indices]
                    sampled = torch.multinomial(selected_probs, num_samples=1).squeeze(-1)
                else:
                    sampled = top1_tokens[topk_indices]

                seq[0, topk_indices] = sampled

        return seq


class WindowedMDLMSampler:
    """Progressive Windowed Infill Sampler."""

    def __init__(
        self,
        model: torch.nn.Module,
        mask_token_id: int,
        window_size: int = 32,
        num_steps_per_window: int = 16,
        temperature: float = 0.0,
        remask_threshold: float = 0.15
    ):
        self.model = model
        self.mask_token_id = mask_token_id
        self.window_size = window_size
        self.num_steps_per_window = num_steps_per_window
        self.temperature = temperature
        self.remask_threshold = remask_threshold

    @torch.no_grad()
    def sample(
        self,
        target_tokens: int,
        prompt_ids: torch.Tensor | None = None,
        device: str | torch.device = "cpu"
    ) -> torch.Tensor:
        self.model.eval()

        prompt_len = prompt_ids.shape[1] if prompt_ids is not None else 0
        total_seq_len = prompt_len + target_tokens

        seq = torch.full((1, total_seq_len), self.mask_token_id, dtype=torch.long, device=device)
        if prompt_ids is not None:
            seq[:, :prompt_len] = prompt_ids

        current_gen_pos = prompt_len

        while current_gen_pos < total_seq_len:
            window_end = min(current_gen_pos + self.window_size, total_seq_len)
            active_len = window_end

            sub_seq = seq[:, :active_len].clone()
            sub_prompt_len = current_gen_pos
            sub_gen_len = active_len - sub_prompt_len

            if sub_gen_len <= 0:
                break

            for step in range(self.num_steps_per_window):
                is_final_step = (step == self.num_steps_per_window - 1)

                logits = self.model(sub_seq).clone()
                logits[:, :, self.mask_token_id] = -float("inf")

                scaled_logits = logits / max(self.temperature, 1e-5)
                probs = F.softmax(scaled_logits, dim=-1)

                top2_probs, top2_indices = torch.topk(probs, k=2, dim=-1)
                margins = top2_probs[0, :, 0] - top2_probs[0, :, 1]
                top1_tokens = top2_indices[0, :, 0]

                if not is_final_step and step > 0:
                    unmasked_in_window = (sub_seq[0] != self.mask_token_id)
                    unmasked_in_window[:sub_prompt_len] = False

                    token_changed = (sub_seq[0] != top1_tokens) & unmasked_in_window
                    low_margin = (margins < self.remask_threshold) & unmasked_in_window
                    remask_mask = token_changed | low_margin

                    if remask_mask.any():
                        sub_seq[0, remask_mask] = self.mask_token_id

                current_mask = (sub_seq[0] == self.mask_token_id)
                current_mask[:sub_prompt_len] = False

                num_masked = current_mask.sum().item()
                if num_masked == 0:
                    break

                if is_final_step:
                    sub_seq[0, current_mask] = top1_tokens[current_mask]
                    break

                progress = (step + 1) / self.num_steps_per_window
                target_unmasked = math.ceil(progress * sub_gen_len)
                current_unmasked = sub_gen_len - num_masked
                num_to_unmask = max(1, target_unmasked - current_unmasked)

                scores = margins.clone()
                scores[~current_mask] = -float("inf")

                k = min(num_to_unmask, num_masked)
                if k > 0:
                    _, topk_indices = torch.topk(scores, k=k)
                    sub_seq[0, topk_indices] = top1_tokens[topk_indices]

            seq[0, :active_len] = sub_seq[0]
            current_gen_pos = window_end

        return seq
