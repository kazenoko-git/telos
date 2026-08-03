"""Windowed Progressive Localized Masked Diffusion Sampler.

Eliminates the 500-token blank slate problem by evaluating and unmasking tokens in small,
localized progressive windows (e.g. window size W=32) immediately adjacent to the prompt context.

This matches the model's training distribution (local context surrounding masks) and prevents
the model from defaulting to unconditioned global token frequencies (e.g., license headers, 'self', '0').
"""

import math
import torch
import torch.nn.functional as F


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
        """
        Args:
            model: trained TelosTransformer model.
            mask_token_id: token ID used for [MASK].
            window_size: localized window size W (default 32).
            num_steps_per_window: unmasking steps per window.
            temperature: sampling temperature (0.0 = argmax).
            remask_threshold: margin threshold for non-monotonic re-masking inside window.
        """
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

        # Full sequence initialized with prompt + [MASK] tokens
        seq = torch.full((1, total_seq_len), self.mask_token_id, dtype=torch.long, device=device)
        if prompt_ids is not None:
            seq[:, :prompt_len] = prompt_ids

        current_gen_pos = prompt_len

        # Progressive Sliding Window Loop
        while current_gen_pos < total_seq_len:
            window_end = min(current_gen_pos + self.window_size, total_seq_len)
            active_len = window_end

            # Slice sequence up to active window end
            sub_seq = seq[:, :active_len].clone()
            sub_prompt_len = current_gen_pos
            sub_gen_len = active_len - sub_prompt_len

            if sub_gen_len <= 0:
                break

            # Run iterative unmasking inside the active window
            for step in range(self.num_steps_per_window):
                is_final_step = (step == self.num_steps_per_window - 1)

                logits = self.model(sub_seq).clone()
                logits[:, :, self.mask_token_id] = -float("inf")

                scaled_logits = logits / max(self.temperature, 1e-5)
                probs = F.softmax(scaled_logits, dim=-1)

                top2_probs, top2_indices = torch.topk(probs, k=2, dim=-1)
                margins = top2_probs[0, :, 0] - top2_probs[0, :, 1]
                top1_tokens = top2_indices[0, :, 0]

                # Non-monotonic re-masking within current active window
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

                # Unmask highest margin tokens in active window
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

            # Write unmasked window tokens back to main sequence
            seq[0, :active_len] = sub_seq[0]
            current_gen_pos = window_end

        return seq
