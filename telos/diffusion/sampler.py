"""Iterative Denoising Samplers for Masked Diffusion Inference.

Includes:
- MDLMSampler: PyTorch confidence-based unmasking sampler (linear / cosine schedules).
- MLXMDLMSampler: MLX Apple Silicon native confidence-based unmasking sampler.
"""

from __future__ import annotations
import math

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    
    class _DummyTorch:
        @staticmethod
        def no_grad():
            def decorator(func):
                return func
            return decorator
    
    torch = _DummyTorch()

try:
    import mlx.core as mx
    import mlx.nn as mx_nn
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


class MDLMSampler:
    """Iterative confidence-based unmasking sampler for PyTorch Masked Diffusion Models."""

    def __init__(
        self,
        model: 'torch.nn.Module',
        mask_token_id: int,
        num_steps: int = 64,
        temperature: float = 1.0,
        repetition_penalty: float = 1.2,
        schedule: str = "cosine"
    ):
        self.model = model
        self.mask_token_id = mask_token_id
        self.num_steps = num_steps
        self.temperature = max(temperature, 1e-5)
        self.repetition_penalty = repetition_penalty
        self.schedule = schedule

    def _get_num_to_unmask(self, step: int, total_masked: int) -> int:
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

            scaled_logits = logits / self.temperature
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


class MLXMDLMSampler:
    """Apple Silicon MLX native iterative confidence-based unmasking sampler."""

    def __init__(
        self,
        model,
        mask_token_id: int,
        num_steps: int = 64,
        temperature: float = 0.8,
        schedule: str = "cosine"
    ):
        if not MLX_AVAILABLE:
            raise ImportError("MLX is required for MLXMDLMSampler.")
        self.model = model
        self.mask_token_id = mask_token_id
        self.num_steps = num_steps
        self.temperature = max(temperature, 1e-5)
        self.schedule = schedule

    def _get_num_to_unmask(self, step: int, total_masked: int) -> int:
        if self.schedule == "cosine":
            progress = (step + 1) / self.num_steps
            ratio = 1.0 - math.cos(progress * math.pi / 2.0)
            target_unmasked = math.ceil(ratio * total_masked)
        else:
            target_unmasked = math.ceil(((step + 1) / self.num_steps) * total_masked)
        return min(target_unmasked, total_masked)

    def sample(self, seq_len: int, prompt_ids=None):
        import numpy as np
        
        # Start with all mask tokens
        seq = np.full((1, seq_len), self.mask_token_id, dtype=np.int32)
        prompt_len = 0
        if prompt_ids is not None:
            if hasattr(prompt_ids, "tolist"):
                p_list = prompt_ids.tolist()
            else:
                p_list = list(prompt_ids)
            p_arr = np.array(p_list, dtype=np.int32)
            if p_arr.ndim == 1:
                p_arr = p_arr[np.newaxis, :]
            prompt_len = p_arr.shape[1]
            seq[:, :prompt_len] = p_arr

        total_masked = seq_len - prompt_len
        if total_masked <= 0:
            return mx.array(seq)

        already_unmasked = 0

        for step in range(self.num_steps):
            mask_locs = (seq == self.mask_token_id)[0]
            mask_locs[:prompt_len] = False
            num_masked = np.sum(mask_locs)
            if num_masked == 0:
                break

            target_unmasked = self._get_num_to_unmask(step, total_masked)
            k = min(target_unmasked - already_unmasked, int(num_masked))
            if k <= 0:
                continue

            logits_mx = self.model(mx.array(seq))
            probs_mx = mx_nn.softmax(logits_mx.astype(mx.float32) / self.temperature, axis=-1)
            mx.eval(probs_mx)
            probs = np.array(probs_mx[0])  # [seq_len, vocab_size]

            # Set mask token prob to 0
            probs[:, self.mask_token_id] = 0.0

            # Confidence margin: top1 - top2
            sorted_probs = np.sort(probs, axis=-1)
            margin = sorted_probs[:, -1] - sorted_probs[:, -2]
            
            # Mask out non-masked positions
            margin[~mask_locs] = -1e9

            # Pick top k positions with highest confidence
            top_k_pos = np.argsort(margin)[-k:]
            
            # Pick best token for those positions
            best_tokens = np.argmax(probs[top_k_pos], axis=-1)
            seq[0, top_k_pos] = best_tokens
            already_unmasked += k

        return mx.array(seq)
