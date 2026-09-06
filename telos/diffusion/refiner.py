"""Multi-Stage Iterative Refiner for COROSred (Causal Routing with Self-Refinement).

Integrates:
1. Region Routing: Morphological span dilation and gap-bridging to prevent isolated-token holes.
2. Expected-Gain Routing: Evaluates whether editing a position yields positive expected reward E[ΔL].
3. Multi-Pass Contextual Refinement: Iterative refinement passes with early-stopping gates.
"""

from __future__ import annotations
import math
from typing import Optional, Union

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from telos.diffusion.region import form_refinement_regions


class COROSredRefiner:
    """Iterative multi-pass refiner leveraging causal routing and bidirectional denoising."""

    def __init__(
        self,
        model: 'torch.nn.Module',
        mask_token_id: int = 1,
        mode: str = "expected_gain",
        threshold: Optional[float] = None,
        top_k_ratio: Optional[float] = 0.08,
        radius: int = 1,
        max_gap: int = 1,
        max_passes: int = 2,
        temperature: float = 0.0,
    ):
        """
        Args:
            model: TelosTransformer with `use_reliability_head=True`.
            mask_token_id: ID of the [MASK] token in vocabulary.
            mode: Scoring mode for routing ('expected_gain', 'reliability', or 'error_prob').
            threshold: Absolute cutoff for routing. If set, only positions exceeding threshold are refined.
            top_k_ratio: Fraction of tokens to flag if threshold is None (e.g. 0.08 seed tokens).
            radius: Dilation radius for expanding flagged positions into surrounding context.
            max_gap: Maximum gap between nearby flags to bridge into contiguous spans.
            max_passes: Maximum number of iterative refinement passes.
            temperature: Sampling temperature for bidirectional infilling (0.0 for greedy argmax).
        """
        self.model = model
        self.mask_token_id = mask_token_id
        self.mode = mode
        self.threshold = threshold
        self.top_k_ratio = top_k_ratio
        self.radius = radius
        self.max_gap = max_gap
        self.max_passes = max_passes
        self.temperature = max(temperature, 0.0)

    @torch.no_grad()
    def refine(
        self,
        seq: torch.Tensor,
        prompt_len: int = 0,
        return_metrics: bool = False
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict]]:
        """Performs multi-stage region-based refinement on an initial draft sequence.

        Args:
            seq: Tensor of shape `(B, T)` containing the initial draft tokens.
            prompt_len: Number of prefix prompt tokens to freeze and protect from refinement.
            return_metrics: If True, returns auxiliary pass statistics.

        Returns:
            Refined token sequence `(B, T)`, and optionally dict of pass metrics.
        """
        self.model.eval()
        curr_seq = seq.clone()
        b, t = curr_seq.shape

        metrics = {
            "passes_executed": 0,
            "total_tokens_refined": 0,
            "mask_pct_per_pass": [],
        }

        for p_idx in range(self.max_passes):
            # Step 1: Forward pass under causal mask to obtain routing signals
            causal_out = self.model(curr_seq, return_reliability=True, mask_override=True)
            if isinstance(causal_out, tuple):
                causal_logits, scores = causal_out
            else:
                raise ValueError("Model must return reliability/gain scores (use_reliability_head=True).")

            shift_scores = scores[:, :-1]

            # Step 2: Early termination check for expected gain
            # If all predicted gains are below threshold, compute is complete
            if self.mode == "expected_gain" and self.threshold is not None:
                # Disregard prompt positions in threshold check
                valid_scores = shift_scores[:, prompt_len:] if prompt_len > 0 else shift_scores
                if valid_scores.numel() > 0 and valid_scores.max().item() < self.threshold:
                    break

            # Step 3: Form contiguous refinement regions via morphological dilation & bridging
            shift_mask = form_refinement_regions(
                shift_scores,
                mode=self.mode,
                threshold=self.threshold,
                top_k_ratio=self.top_k_ratio if self.threshold is None else None,
                radius=self.radius,
                max_gap=self.max_gap,
                prompt_len=max(0, prompt_len - 1)
            )

            # Prepend False for token 0 (always prompt or BOS)
            full_mask = torch.cat([torch.zeros_like(curr_seq[:, :1], dtype=torch.bool), shift_mask], dim=1)
            if prompt_len > 0:
                full_mask[:, :prompt_len] = False

            num_masked = full_mask.sum().item()
            if num_masked == 0:
                break

            mask_pct = (num_masked / float(b * t)) * 100.0
            metrics["mask_pct_per_pass"].append(mask_pct)
            metrics["total_tokens_refined"] += num_masked
            metrics["passes_executed"] += 1

            # Step 4: Bidirectional infilling of masked regions
            corrupted_seq = torch.where(full_mask, self.mask_token_id, curr_seq)
            denoise_out = self.model(corrupted_seq, return_reliability=False, mask_override=False)
            denoise_logits = denoise_out[0] if isinstance(denoise_out, tuple) else denoise_out

            # Step 5: Token selection for masked positions
            if self.temperature > 0.05:
                probs = F.softmax(denoise_logits / self.temperature, dim=-1)
                flat_probs = probs.reshape(-1, probs.shape[-1])
                pred_tokens = torch.multinomial(flat_probs, num_samples=1).reshape(b, t)
            else:
                pred_tokens = torch.argmax(denoise_logits, dim=-1)

            # Step 6: Infill the refined tokens into the draft sequence
            curr_seq = torch.where(full_mask, pred_tokens, curr_seq)

        if return_metrics:
            return curr_seq, metrics
        return curr_seq
