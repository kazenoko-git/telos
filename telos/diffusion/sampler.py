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
    from telos.training.core import MLX_AVAILABLE
    if MLX_AVAILABLE:
        import mlx.core as mx
        import mlx.nn as mx_nn
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
        schedule: str = "cosine",
        confidence_threshold: float | None = None,
        early_commit_margin: float | None = None,
    ):
        self.model = model
        self.mask_token_id = mask_token_id
        self.num_steps = num_steps
        self.temperature = max(temperature, 1e-5)
        self.repetition_penalty = repetition_penalty
        self.schedule = schedule
        self.confidence_threshold = confidence_threshold
        self.early_commit_margin = early_commit_margin

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
        device: str | torch.device | None = None
    ) -> torch.Tensor:
        self.model.eval()
        if device is None:
            if hasattr(self.model, "parameters"):
                try:
                    device = next(self.model.parameters()).device
                except (StopIteration, Exception):
                    device = "cpu"
            else:
                device = "cpu"
        seq = torch.full((1, seq_len), self.mask_token_id, dtype=torch.long, device=device)
        
        prompt_len = 0
        if prompt_ids is not None:
            prompt_len = prompt_ids.shape[1]
            seq[:, :prompt_len] = prompt_ids.to(device)

        total_masked_positions = seq_len - prompt_len
        if total_masked_positions <= 0:
            return seq

        already_unmasked_count = 0

        for step in range(self.num_steps):
            current_mask = (seq == self.mask_token_id)
            current_mask[:, :prompt_len] = False

            masked_positions = current_mask[0].nonzero(as_tuple=False).squeeze(-1)
            num_currently_masked = int(masked_positions.numel())
            if num_currently_masked == 0:
                break

            target_unmasked_count = self._get_num_to_unmask(step, total_masked_positions)
            num_to_unmask_this_step = target_unmasked_count - already_unmasked_count
            if num_to_unmask_this_step <= 0:
                continue

            if hasattr(self.model, "hidden_states") and hasattr(self.model, "output_projection"):
                hidden = self.model.hidden_states(seq)
                masked_logits = self.model.output_projection(hidden[0, masked_positions]).clone()
            else:
                logits = self.model(seq)
                # Slice only masked positions before repetition penalty, softmax, and topk.
                masked_logits = logits[0, masked_positions].clone()
            masked_logits[:, self.mask_token_id] = -float("inf")

            if self.repetition_penalty != 1.0:
                gen_seq = seq[0, prompt_len:] if prompt_len is not None else seq[0]
                unmasked_tokens = gen_seq[(gen_seq != self.mask_token_id) & (gen_seq > 3)]
                if unmasked_tokens.numel() > 0:
                    unique_toks = torch.unique(unmasked_tokens)
                    selected = masked_logits[:, unique_toks]
                    penalized = torch.where(
                        selected > 0,
                        selected / self.repetition_penalty,
                        selected * self.repetition_penalty,
                    )
                    masked_logits[:, unique_toks] = penalized

            probs = F.softmax(masked_logits / self.temperature, dim=-1)

            top2_probs, top2_indices = torch.topk(probs, k=2, dim=-1)
            margin_conf = top2_probs[:, 0] - top2_probs[:, 1]
            argmax_tokens = top2_indices[:, 0]

            # Optional Prophet-style early commit when every remaining masked token has high top-2 margin.
            if (self.early_commit_margin is not None
                    and float(margin_conf.min().item()) >= self.early_commit_margin):
                if self.temperature > 0.05:
                    sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
                else:
                    sampled_tokens = argmax_tokens
                seq[0, masked_positions] = sampled_tokens
                already_unmasked_count += num_currently_masked
                break

            # Optional Fast-dLLM-style confidence-threshold parallel unmasking.
            if self.confidence_threshold is not None:
                high_conf_local = (top2_probs[:, 0] >= self.confidence_threshold).nonzero(as_tuple=False).squeeze(-1)
                if high_conf_local.numel() > 0:
                    chosen_local = high_conf_local
                else:
                    k = min(num_to_unmask_this_step, num_currently_masked)
                    _, chosen_local = torch.topk(margin_conf, k=k)
            else:
                if self.temperature > 0.05:
                    gumbel_noise = -torch.log(-torch.log(torch.rand_like(margin_conf) + 1e-8) + 1e-8)
                    confidence_scores = margin_conf + 0.1 * self.temperature * gumbel_noise
                else:
                    confidence_scores = margin_conf
                k = min(num_to_unmask_this_step, num_currently_masked)
                _, chosen_local = torch.topk(confidence_scores, k=k)

            chosen_positions = masked_positions[chosen_local]
            if self.temperature > 0.05:
                sampled_tokens = torch.multinomial(probs[chosen_local], num_samples=1).squeeze(-1)
            else:
                sampled_tokens = argmax_tokens[chosen_local]

            seq[0, chosen_positions] = sampled_tokens
            already_unmasked_count += int(chosen_positions.numel())

        return seq


class MLXMDLMSampler:
    """Apple Silicon MLX native iterative confidence-based unmasking sampler."""

    def __init__(
        self,
        model,
        mask_token_id: int,
        num_steps: int = 64,
        temperature: float = 0.8,
        schedule: str = "cosine",
        confidence_threshold: float | None = None,
        early_commit_margin: float | None = None,
    ):
        if not MLX_AVAILABLE:
            raise ImportError("MLX is required for MLXMDLMSampler.")
        self.model = model
        self.mask_token_id = mask_token_id
        self.num_steps = num_steps
        self.temperature = max(temperature, 1e-5)
        self.schedule = schedule
        self.confidence_threshold = confidence_threshold
        self.early_commit_margin = early_commit_margin

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
            masked_idx = np.flatnonzero(mask_locs)
            num_masked = int(masked_idx.size)
            if num_masked == 0:
                break

            target_unmasked = self._get_num_to_unmask(step, total_masked)
            k = min(target_unmasked - already_unmasked, num_masked)
            if k <= 0:
                continue

            masked_idx_mx = mx.array(masked_idx, dtype=mx.int32)
            if hasattr(self.model, "hidden_states") and hasattr(self.model, "logits_from_hidden"):
                hidden = self.model.hidden_states(mx.array(seq))[0]
                masked_logits = self.model.logits_from_hidden(hidden[masked_idx_mx]).astype(mx.float32)
            else:
                logits_mx = self.model(mx.array(seq))[0]
                masked_logits = logits_mx[masked_idx_mx].astype(mx.float32)
            masked_logits[:, self.mask_token_id] = -1e9
            probs_mx = mx_nn.softmax(masked_logits / self.temperature, axis=-1)

            # Compute top-2 margin and argmax on device instead of full-vocab CPU sort.
            top2_probs = mx.topk(probs_mx, 2, axis=-1)
            margin_mx = top2_probs[:, 0] - top2_probs[:, 1]
            best_tokens_mx = mx.argmax(probs_mx, axis=-1)
            mx.eval(margin_mx, top2_probs, best_tokens_mx)

            margin = np.array(margin_mx)
            best_tokens = np.array(best_tokens_mx)
            top1_probs = np.array(top2_probs[:, 0])

            if self.early_commit_margin is not None and float(margin.min()) >= self.early_commit_margin:
                seq[0, masked_idx] = best_tokens
                already_unmasked += num_masked
                break

            if self.confidence_threshold is not None:
                high_conf = np.flatnonzero(top1_probs >= self.confidence_threshold)
                chosen_local = high_conf if high_conf.size > 0 else np.argpartition(margin, -k)[-k:]
            else:
                chosen_local = np.argpartition(margin, -k)[-k:]

            seq[0, masked_idx[chosen_local]] = best_tokens[chosen_local]
            already_unmasked += int(chosen_local.size)

        return mx.array(seq)
