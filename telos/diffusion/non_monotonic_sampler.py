"""Non-Monotonic Discrete Diffusion Sampler with Re-Masking & Re-evaluation.

Unlike monotonic samplers that permanently lock predictions at step 1, this sampler
dynamically re-evaluates previously unmasked tokens at every step. If a token's confidence
under updated surrounding context falls below threshold or conflicts with new model predictions,
it is re-masked back to [MASK] to allow iterative self-correction and revisions.
"""

import math
import torch
import torch.nn.functional as F


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
        """
        Args:
            model: trained TelosTransformer model.
            mask_token_id: token ID used for [MASK].
            num_steps: number of iterative refinement steps.
            temperature: sampling temperature (0.0 = argmax).
            repetition_penalty: repetition penalty factor (default 1.0 = off).
            schedule: "cosine" or "linear".
            remask_threshold: margin threshold below which unmasked tokens are re-masked.
        """
        self.model = model
        self.mask_token_id = mask_token_id
        self.num_steps = num_steps
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.schedule = schedule
        self.remask_threshold = remask_threshold

    def _get_target_unmasked_count(self, step: int, total_masked: int) -> int:
        """Determines target number of unmasked tokens at current step."""
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

        # Initialize sequence: prompt + [MASK] tokens
        seq = torch.full((1, seq_len), self.mask_token_id, dtype=torch.long, device=device)

        prompt_len = 0
        if prompt_ids is not None:
            prompt_len = prompt_ids.shape[1]
            seq[:, :prompt_len] = prompt_ids

        total_gen_len = seq_len - prompt_len
        if total_gen_len <= 0:
            return seq

        # Iterative Non-Monotonic Denoising Loop
        for step in range(self.num_steps):
            is_final_step = (step == self.num_steps - 1)

            # 1. Forward Pass to compute probabilities across sequence
            logits = self.model(seq).clone()
            logits[:, :, self.mask_token_id] = -float("inf")

            # Temperature scaling & Softmax
            scaled_logits = logits / max(self.temperature, 1e-5)
            probs = F.softmax(scaled_logits, dim=-1)

            # Compute Probability Margin (top1 - top2) and Argmax tokens
            top2_probs, top2_indices = torch.topk(probs, k=2, dim=-1)
            margins = top2_probs[0, :, 0] - top2_probs[0, :, 1]
            top1_tokens = top2_indices[0, :, 0]

            # 2. Dynamic Re-Masking Step (skip on final step)
            if not is_final_step and step > 0:
                # Identify currently unmasked positions in generated region
                unmasked_gen_positions = (seq[0] != self.mask_token_id)
                unmasked_gen_positions[:prompt_len] = False  # protect prompt

                # Re-mask if model's current top1 prediction changed OR margin is low
                token_changed = (seq[0] != top1_tokens) & unmasked_gen_positions
                low_confidence = (margins < self.remask_threshold) & unmasked_gen_positions
                remask_mask = token_changed | low_confidence

                if remask_mask.any():
                    seq[0, remask_mask] = self.mask_token_id

            # 3. Selection & Unmasking Step
            current_mask = (seq[0] == self.mask_token_id)
            current_mask[:prompt_len] = False  # protect prompt

            num_currently_masked = current_mask.sum().item()
            if num_currently_masked == 0 and not is_final_step:
                continue

            target_unmasked = self._get_target_unmasked_count(step, total_gen_len)
            current_unmasked = total_gen_len - num_currently_masked
            num_to_unmask = max(1, target_unmasked - current_unmasked)

            if is_final_step:
                # On final step, unmask ALL remaining masked positions with argmax
                seq[0, current_mask] = top1_tokens[current_mask]
                break

            # Add Gumbel noise to margins if temperature > 0
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
