"""forward (noising) process for UNDLM (Uniform Noise Diffusion Language Modeling)

instead of absorbing tokens into [MASK], each token is independently replaced with a
uniformly random vocab token with probability t.
this enables self-correction (in theory) during sampling, as corruption is reversible.
"""

import math, numpy as np
try: 
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

def apply_uniform_noise_mlx(
    input_ids,
    t_values,
    vocab_size: int,
    special_token_lut=None,
):
    B, T = input_ids.shape
    rand_matrix = mx.random.uniform(0.0, 1.0, (B, T))
    noise_tokens = mx.random.randint(0, vocab_size, (B, T))
    raw_corrupt_mask = rand_matrix < t_values

    if special_token_lut is not None:
        is_special = special_token_lut[input_ids]
    else:
        is_special = input_ids < 4

    corrupt_mask = raw_corrupt_mask & (~is_special)
    noisy_ids = mx.where(corrupt_mask, noise_tokens, input_ids)
    return noisy_ids, corrupt_mask, t_values

# TIMESTEP SAMPLERS

def _sample_beta_timesteps(batch_size: int, alpha: float = 1.5, beta: float = 1.5, eps: float = 1e-5):
    t = np.random.beta(alpha, beta, size=(batch_size, 1)).astype(np.float32)
    return mx.clip(mx.array(t), eps, 1.0)

def _sample_cosine_timesteps(batch_size: int, eps: float = 1e-5):
    u = mx.random.uniform(0.0, 1.0, (batch_size, 1))
    t = 0.5 - 0.5 * mx.cos(math.pi * u)
    return mx.clip(t, eps, 1.0)

def _sample_uniform_timesteps(batch_size: int, eps: float = 1e-5):
    u = mx.random.uniform(0.0, 1.0, (batch_size, 1))
    return mx.clip(u, eps, 1.0)