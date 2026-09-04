"""
Iterative Denoising Sampler for UNDLM with True Bidirectional Self-Correction.

Key Distinction from MDLM:
- MDLM unmasks tokens monotonically (once unmasked, permanently locked).
- UNDLM maintains full sequence representations and performs iterative 
  Self-Correction: at every denoising step, all positions are re-evaluated.
"""

import math
import numpy as np

try:
    import mlx.core as mx
    import mlx.nn as nn
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


class UNDLMSampler:
    # Self-correcting iterative denoising sampler for Uniform Noise Discrete DLMs.

    def __init__(
        self,
        model,
        vocab_size: int,
        num_steps: int = 64,
        temperature: float = 0.8,
        schedule: str = "cosine",
        mode: str = "self_correction"  # 'self_correction' (confidence-guided revision) or 'posterior'
    ):
        self.model = model
        self.vocab_size = vocab_size
        self.num_steps = num_steps
        self.temperature = max(temperature, 1e-4)
        self.schedule = schedule
        self.mode = mode

    def _get_noise_level(self, step: int) -> float:
        """Returns the noise level t for a given step (decreasing from 1 -> 0)."""
        progress = step / self.num_steps  # 0 -> 1 as step increases
        if self.schedule == "cosine":
            # Cosine schedule: smooth start and end
            return 1.0 - (1.0 - math.cos(math.pi * progress)) / 2.0
        else:
            # Linear schedule: uniform decrease
            return 1.0 - progress

    def sample(self, seq_len: int, prompt_ids=None) -> "mx.array":
        if not MLX_AVAILABLE:
            raise ImportError("MLX is required for MLX UNDLMSampler.")

        # Step 0: Initialize with uniform categorical noise (t=1.0)
        x_t = mx.random.randint(0, self.vocab_size, shape=(1, seq_len))

        prompt_len = 0
        if prompt_ids is not None:
            prompt_len = prompt_ids.shape[1]
            x_t[:, :prompt_len] = prompt_ids

        for step in range(1, self.num_steps + 1):
            t_current = self._get_noise_level(step - 1)  # Noise level before step
            t_next = self._get_noise_level(step)          # Noise level after step

            # Step A: Evaluate full bidirectional logits under current noisy/partially-resolved state
            logits = self.model(x_t)  # [1, seq_len, vocab_size]

            # Step B: Compute predictive probabilities and token confidences
            scaled_logits = logits / self.temperature
            probs = mx.softmax(scaled_logits, axis=-1)  # [1, seq_len, vocab_size]
            
            # Confidence metric: top-1 probability margin or maximum probability
            confidences = mx.max(probs, axis=-1)  # [1, seq_len], range [0, 1]

            # Step C: Sample clean token candidates via Gumbel-Max
            gumbel_noise = -mx.log(-mx.log(mx.random.uniform(shape=probs.shape) + 1e-8) + 1e-8)
            x_clean_pred = mx.argmax(mx.log(probs + 1e-8) + gumbel_noise, axis=-1)  # [1, seq_len]

            if step == self.num_steps:
                # Final step: Lock in clean predictions
                x_t = x_clean_pred
            else:
                if self.mode == "self_correction":
                    # --- Self-Correction Sampling ---
                
                    # Confidence threshold evolves inversely with noise:
                    # At t=1.0, threshold is low (explore), at t=0, threshold is high (lock).
                    conf_threshold = (1.0 - t_next) * 0.85
                    
                    # Update mask: High confidence positions lock/revise to clean prediction
                    lock_mask = confidences >= conf_threshold
                    
                    # For positions not yet confident enough, instead of blind random noise,
                    # sample from the model's posterior distribution with temperature annealing:
                    annealed_probs = mx.softmax(logits / max(self.temperature * (t_next + 0.2), 0.1), axis=-1)
                    gumbel_annealed = -mx.log(-mx.log(mx.random.uniform(shape=annealed_probs.shape) + 1e-8) + 1e-8)
                    x_refine = mx.argmax(mx.log(annealed_probs + 1e-8) + gumbel_annealed, axis=-1)
                    
                    # Merge locked clean tokens with refined exploring tokens (Self-Correction Step)
                    x_t = mx.where(lock_mask, x_clean_pred, x_refine)

                else:
                    # --- Discrete Diffusion Posterior Transition q(x_{t-1} | x_t, x_0) ---
                    # Exact Markov transition probability to step t_next
                    # Alpha parameters: alpha_t = 1 - t
                    alpha_t = 1.0 - t_current
                    alpha_s = 1.0 - t_next
                    
                    # Probability of transitioning to clean prediction vs retaining uniform state
                    p_clean = (alpha_s - alpha_t) / max(1.0 - alpha_t, 1e-6)
                    p_clean = min(max(p_clean, 0.0), 1.0)
                    
                    transition_to_clean = mx.random.uniform(shape=(1, seq_len)) < p_clean
                    x_t = mx.where(transition_to_clean, x_clean_pred, x_t)

            # Preserve conditioned prompt prefix
            if prompt_ids is not None:
                x_t[:, :prompt_len] = prompt_ids

            mx.eval(x_t)

        return x_t