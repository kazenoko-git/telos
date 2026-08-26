"""
Inference Sampler for COROSred: Confidence-Routed Selective Re-Diffusion.

Pipeline:
1. Phase 1 — Draft (Causal Mask, Autoregressive next-token emission + reliability scoring)
2. Phase 2 — Flag (Thresholding on r_i cost-quality knob)
3. Phase 3 — Refine (Bidirectional Mask, Full Recomputation via MDLM re-diffusion steps)
"""

import math
import mlx.core as mx


class COROSredSampler:
    """
    Inference orchestrator for drafting, routing, and refining sequences.
    """

    def __init__(
        self,
        model,
        mask_token_id: int,
        reliability_threshold: float = 0.0,
        refine_steps: int = 2,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ):
        self.model = model
        self.mask_token_id = mask_token_id
        self.reliability_threshold = reliability_threshold
        self.refine_steps = refine_steps
        self.temperature = max(temperature, 1e-5)
        self.top_p = top_p

    def _sample_token(self, logits: mx.array) -> int:
        """Top-p nucleus sampling on raw unnormalized logits."""
        # Scale logits by sampling temperature
        scaled_logits = logits / self.temperature
        probs = mx.softmax(scaled_logits, axis=-1)

        # Sort probabilities in descending order for cumulative mass calculation
        sorted_indices = mx.argsort(-probs, axis=-1)
        sorted_probs = probs[sorted_indices]
        cumulative_probs = mx.cumsum(sorted_probs, axis=-1)

        # Zero out probabilities beyond top_p threshold
        cutoff_mask = cumulative_probs > self.top_p
        # Shift mask right so at least the first token is always preserved
        cutoff_mask = mx.concatenate([mx.zeros((1,), dtype=mx.bool_), cutoff_mask[:-1]], axis=0)
        sorted_probs = mx.where(cutoff_mask, mx.zeros_like(sorted_probs), sorted_probs)
        sorted_probs = sorted_probs / mx.sum(sorted_probs)

        # Categorical draw from filtered distribution
        sampled_idx_in_sorted = mx.random.categorical(mx.log(mx.clip(sorted_probs, 1e-10, 1.0)))
        return sorted_indices[sampled_idx_in_sorted].item()

    def draft(self, prompt_tokens: list[int], max_new_tokens: int) -> tuple[list[int], list[float]]:
        """
        Phase 1: Generates tokens autoregressively and stores pre-decision reliability scores.
        """
        generated = list(prompt_tokens)
        # Gold prompt tokens are always marked reliable
        reliability_scores = [float("inf")] * len(prompt_tokens)

        for _ in range(max_new_tokens):
            inp = mx.array([generated], dtype=mx.int32)
            # Forward pass under causal mask extracting pre-decision scores
            logits, r_scores = self.model(inp, is_causal=True, return_reliability=True)

            # Extract last position logits and reliability prediction for the upcoming token
            last_logit = logits[0, -1]
            last_r_score = r_scores[0, -1].item()

            next_token = self._sample_token(last_logit)
            generated.append(next_token)
            # Store r_i at the instant of emitting token x_i
            reliability_scores.append(last_r_score)

        return generated, reliability_scores

    def flag(self, scores: list[float]) -> list[int]:
        """
        Phase 2: Identifies token positions falling below the cost-quality threshold.
        """
        # Flag all positions where reliability falls below threshold
        return [idx for idx, score in enumerate(scores) if score < self.reliability_threshold]

    def refine(self, sequence: list[int], flagged_indices: list[int]) -> list[int]:
        """
        Phase 3: Selectively masks flagged tokens and performs bidirectional MDLM refinement passes.
        """
        if not flagged_indices:
            return sequence

        refined = list(sequence)
        # Apply mask token to all flagged unreliable positions
        for idx in flagged_indices:
            refined[idx] = self.mask_token_id

        # Perform iterative refinement passes with full bidirectional recomputation (no KV cache reuse)
        for _ in range(self.refine_steps):
            inp = mx.array([refined], dtype=mx.int32)
            # Full bidirectional forward pass across entire sequence
            logits = self.model(inp, is_causal=False, return_reliability=False)

            # Update only masked positions with new highest-confidence unmasking predictions
            for idx in flagged_indices:
                token_logits = logits[0, idx]
                # Greedily replace masked token with model's bidirectional consensus
                refined[idx] = mx.argmax(token_logits, axis=-1).item()

        return refined

    def generate(self, prompt_tokens: list[int], max_new_tokens: int) -> dict:
        """
        Full COROSred Generation Pipeline: Draft -> Flag -> Refine.
        """
        drafted_seq, r_scores = self.draft(prompt_tokens, max_new_tokens)
        flagged_positions = self.flag(r_scores)
        final_seq = self.refine(drafted_seq, flagged_positions)

        return {
            "draft_tokens": drafted_seq,
            "final_tokens": final_seq,
            "reliability_scores": r_scores,
            "flagged_positions": flagged_positions,
            "refine_count": len(flagged_positions),
        }
