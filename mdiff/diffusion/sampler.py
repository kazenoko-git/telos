"""Iterative Denoising Samplers for Masked Diffusion Inference.

Includes:
- MDLMSampler: Standard confidence-based unmasking sampler (linear / cosine schedules).
"""

import math
import torch
import torch.nn.functional as F


class MDLMSampler:
    """Iterative confidence-based unmasking sampler for Masked Diffusion Models.
    
    This sampler executes the core iterative reverse diffusion process. It begins 
    with a completely masked target sequence and progressively unmasks tokens over 
    a predefined number of steps. At each step, it predicts the full vocabulary 
    distribution for all remaining masked positions, measures its confidence 
    (the margin between the top-1 and top-2 token probabilities), and permanently 
    locks in the highest confidence predictions. The quantity of tokens unmasked 
    per step is governed by either a linear or cosine unmasking schedule.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        mask_token_id: int,
        num_steps: int = 64,
        temperature: float = 1.0,
        repetition_penalty: float = 1.2,
        schedule: str = "cosine"
    ):
        """Initializes the iterative sampler with model and hyperparameters.
        
        Args:
            model: The fully initialized TelosTransformer neural network.
            mask_token_id: The vocabulary ID corresponding to the [MASK] token.
            num_steps: The total count of discrete denoising iteration steps.
            temperature: Softmax scaling constant applied prior to sampling.
            repetition_penalty: Logit divisor applied to previously generated tokens.
            schedule: String identifier for unmasking curve ("cosine" or "linear").
        """
        self.model = model
        self.mask_token_id = mask_token_id
        self.num_steps = num_steps
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.schedule = schedule

    def _get_num_to_unmask(self, step: int, total_masked: int) -> int:
        """Determines the exact count of tokens to unmask this iteration."""
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
        """Executes the complete multi-step iterative denoising inference loop.
        
        Args:
            seq_len: The absolute total length of the required output sequence.
            prompt_ids: Optional prefix tokens acting as structural conditioning context.
            device: String or torch device specifying tensor placement execution.
            
        Returns:
            A fully denoised tensor sequence combining prompt and generated tokens.
        """
        self.model.eval()

        # Initialize sequence with masks.
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
            # Protect prompt tokens.
            current_mask[:, :prompt_len] = False

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
