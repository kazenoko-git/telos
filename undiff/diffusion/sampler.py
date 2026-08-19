"""
Iterative Denoising Sampler for UNDLM with Self Correction

Unlike MDLM's confidence based unmasking (once, unmasked, permanent),
the UNDLM sampler re-predicts and re-corrupts at each step, allowing the model to
correct previous mistakes.
"""

import math, mlx.core as mx
import mlx.nn as nn

class UNDLMSampler:
    """Self-correcting iterative denoising sampler for Uniform Noise DLMs."""

    def __init__(self, model, vocab_size: int, num_steps: int = 64, temperature: float = 0.8, schedule: str = "linear"):
        self.model = model
        self.vocab_size = vocab_size
        self.num_steps = num_steps
        self.temperature = temperature
        self.schedule = schedule
        
    def _get_noise_level(self, step: int) -> float:
        """Returns the noise level t for a given step (decreasing from 1 -> 0)."""
        progress = step / self.num_steps  # 0 -> 1 as step increases
        if self.schedule == "cosine":
            # Cosine schedule: starts slow, accelerates, then slows at the end
            return 1.0 - (1.0 - math.cos(math.pi * progress)) / 2.0
        else: 
            # Linear schedule: uniform decrease
            return 1.0 - progress

    def sample(self, seq_len: int, prompt_ids=None) -> mx.array:
        # generate a sequence via iterative denoising from pure noise

        # initialise with pure random noise (t=1)
        x_t = mx.random.randint(0, self.vocab_size, shape=(1, seq_len))

        # if prompt provided, lock prefix positions
        prompt_len = 0
        if prompt_ids is not None:
            prompt_len = prompt_ids.shape[1]
            x_t[:, :prompt_len] = prompt_ids
        
        for step in range(1, self.num_steps+1):
            t_current = self._get_noise_level(step - 1) # noise at start of step
            t_next = self._get_noise_level(step) # noise after the step

            # model predicts clean tokens from noisy input
            logits = self.model(x_t) # [1, seq_len, vocab_size]
            # sample from predicted distribution with temperature
            probs = mx.softmax(logits / self.temperature, axis = -1)
            # gumbel-max trick for efficient categorical sampling
            gumbel_noise = -mx.log(-mx.log(mx.random.uniform(shape=probs.shape) + 1e-8) + 1e-8)
            x_clean_pred = mx.argmax(mx.log(probs + 1e-8) + gumbel_noise, axis = -1) # [1, seq_len]

            if step == self.num_steps:
                # final step
                x_t = x_clean_pred
            else:
                # recorrupt at reduced noise level t_next
                random_tokens = mx.random.randint(0, self.vocab_size, shape=(1, seq_len))            
                corrupt_mask = mx.random.uniform(shape=(1,seq_len)) < t_next 
                x_t = mx.where(corrupt_mask, random_tokens, x_clean_pred)
            
            # always preserve prompt prefix
            if prompt_ids is not None:
                x_t[:, :prompt_len] = prompt_ids
            
            mx.eval(x_t)
            
        return x_t