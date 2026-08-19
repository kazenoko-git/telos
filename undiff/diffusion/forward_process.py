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
    vocab_size: int,
    special_token_lut=None,
    strategy: str = "beta"
):
    """Applies uniform noise corruption at sampled timestep t.

    For each token independently:
      - With probability (1 - t): token stays as the original clean token
      - With probability t: token is replaced by a uniform random sample from [0, vocab_size)
    """
    B, T = input_ids.shape
    if strategy == "beta":
        t_values = _sample_beta_timesteps(B)
    elif strategy == "cosine":
        t_values = _sample_cosine_timesteps(B)
    else:
        t_values = _sample_uniform_timesteps(B)
    
    # Sample uniform random replacement tokens from full vocabulary
    random_tokens = mx.random.randint(0, vocab_size, shape=(B, T))

    # Bernoulli corruption mask: True where we replace with random token
    rand_matrix = mx.random.uniform(0.0, 1.0, (B, T))
    corrupt_mask = rand_matrix < t_values  # broadcast [B, 1] over [B, T]

    # Protect special tokens (PAD=0, MASK=1, BOS=2, EOS=3) from corruption
    if special_token_lut is not None:
        is_special = special_token_lut[input_ids]
    else:
        is_special = (input_ids == 0) | (input_ids == 1) | (input_ids == 2) | (input_ids == 3)
    corrupt_mask = corrupt_mask & (~is_special)

    # Apply corruption: where corrupt_mask is True, use random token; else keep original
    noisy_ids = mx.where(corrupt_mask, random_tokens, input_ids)

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