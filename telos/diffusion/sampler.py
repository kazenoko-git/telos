"""iterative denoising sampler for masked diffusion inference.

start with a sequence of all [MASK] tokens (or prompt tokens + [MASK] tokens).
iteratively run the model forward, evaluate prediction confidence at masked positions,
and progressively unmask tokens using a confidence-based schedule across N steps.
"""

import math
import torch
import torch.nn.functional as F


class MDLMSampler:
    """iterative confidence-based unmasking sampler."""

    def __init__(
        self,
        model: torch.nn.Module,
        mask_token_id: int,
        num_steps: int = 64,
        temperature: float = 1.0,
        repetition_penalty: float = 1.2,
        schedule: str = "linear"
    ):
        """
        Args:
            model: trained TelosTransformer model.
            mask_token_id: token ID used for [MASK].
            num_steps: speed vs quality knob 16–128.
            temperature: sampling temperature.
            repetition_penalty: penalty factor (e.g. 1.2) for already unmasked tokens.
            schedule: "linear" or "cosine".
        """
        self.model = model
        self.mask_token_id = mask_token_id
        self.num_steps = num_steps
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.schedule = schedule

    def _get_num_to_unmask(self, step: int, total_masked: int) -> int:
        """determines how many masked tokens should be unmasked at current step."""
        if self.schedule == "cosine":
            # Cosine schedule: unmask slowly at start, faster in middle, slow at end
            progress = (step + 1) / self.num_steps
            ratio = 1.0 - math.cos(progress * math.pi / 2.0)
            target_unmasked = math.ceil(ratio * total_masked)
        else:
            # Linear schedule: fixed fraction per step
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

        # initialize sequence: start with prompt followed by [MASK] tokens
        seq = torch.full((1, seq_len), self.mask_token_id, dtype=torch.long, device=device)
        
        prompt_len = 0
        if prompt_ids is not None:
            prompt_len = prompt_ids.shape[1]
            seq[:, :prompt_len] = prompt_ids

        # total positions that need unmasking (exclude fixed prompt tokens)
        total_masked_positions = seq_len - prompt_len
        if total_masked_positions <= 0:
            return seq

        already_unmasked_count = 0

        # iterative Denoising Loop
        for step in range(self.num_steps):
            # find currently masked positions (excluding prompt region)
            current_mask = (seq == self.mask_token_id)
            current_mask[:, :prompt_len] = False  # protect prompt

            num_currently_masked = current_mask.sum().item()
            if num_currently_masked == 0:
                break  # all tokens unmasked early

            # calculate total target unmasked count by this step
            target_unmasked_count = self._get_num_to_unmask(step, total_masked_positions)
            num_to_unmask_this_step = target_unmasked_count - already_unmasked_count

            if num_to_unmask_this_step <= 0:
                continue

            # model forward pass: get unnormalized logits [1, seq_len, vocab_size]
            logits = self.model(seq)

            # Zero out mask_token_id in logits so it is never predicted
            logits = logits.clone()
            logits[:, :, self.mask_token_id] = -float("inf")

            # Apply repetition penalty to already unmasked tokens to prevent repetition loops
            if self.repetition_penalty != 1.0:
                unmasked_tokens = seq[seq != self.mask_token_id]
                for tok_id in set(unmasked_tokens.tolist()):
                    if tok_id > 3:  # Skip special tokens
                        logits[:, :, tok_id] = torch.where(
                            logits[:, :, tok_id] > 0,
                            logits[:, :, tok_id] / self.repetition_penalty,
                            logits[:, :, tok_id] * self.repetition_penalty
                        )

            # Apply temperature scaling and compute probabilities
            scaled_logits = logits / max(self.temperature, 1e-5)
            probs = F.softmax(scaled_logits, dim=-1)

            B, L, V = probs.shape

            # Confidence score is max probability at each position
            max_conf, argmax_tokens = torch.max(probs, dim=-1)

            # Add Gumbel noise to confidence scores for position selection diversity
            if self.temperature > 0.05:
                gumbel_noise = -torch.log(-torch.log(torch.rand_like(max_conf) + 1e-8) + 1e-8)
                confidence_scores = max_conf + 0.1 * self.temperature * gumbel_noise
            else:
                confidence_scores = max_conf.clone()

            confidence_scores[~current_mask] = -float("inf")

            # Select top-k positions with highest model certainty to unmask
            k = min(num_to_unmask_this_step, num_currently_masked)
            _, topk_indices = torch.topk(confidence_scores[0], k=k)

            # For selected top-k positions, sample predicted tokens
            if self.temperature > 0.05:
                selected_probs = probs[0, topk_indices]  # [k, V]
                sampled_tokens = torch.multinomial(selected_probs, num_samples=1).squeeze(-1)
            else:
                sampled_tokens = argmax_tokens[0, topk_indices]

            # Update sequence at selected positions
            seq[0, topk_indices] = sampled_tokens
            already_unmasked_count += k

        return seq
