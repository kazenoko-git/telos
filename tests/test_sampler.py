"""Unit test suite for consolidated iterative unmasking samplers."""

import torch
from telos.model.transformer import TelosTransformer, TelosConfig
from telos.diffusion.sampler import MDLMSampler, NonMonotonicMDLMSampler, WindowedMDLMSampler


def test_standard_mdlm_sampler():
    """Verifies MDLMSampler unmasks all tokens."""
    config = TelosConfig(vocab_size=50, d_model=32, n_layers=2, n_heads=2, max_seq_len=16)
    model = TelosTransformer(config)

    sampler = MDLMSampler(model, mask_token_id=1, num_steps=5, schedule="cosine")
    sampled = sampler.sample(seq_len=16)

    assert (sampled != 1).all(), "MDLMSampler left unmasked [MASK] tokens!"
    assert sampled.shape == (1, 16)


def test_non_monotonic_sampler():
    """Verifies NonMonotonicMDLMSampler unmasks all tokens."""
    config = TelosConfig(vocab_size=50, d_model=32, n_layers=2, n_heads=2, max_seq_len=16)
    model = TelosTransformer(config)

    sampler = NonMonotonicMDLMSampler(model, mask_token_id=1, num_steps=5, schedule="cosine")
    sampled = sampler.sample(seq_len=16)

    assert (sampled != 1).all(), "NonMonotonicMDLMSampler left unmasked [MASK] tokens!"
    assert sampled.shape == (1, 16)


def test_windowed_sampler():
    """Verifies WindowedMDLMSampler infills target tokens properly."""
    config = TelosConfig(vocab_size=50, d_model=32, n_layers=2, n_heads=2, max_seq_len=32)
    model = TelosTransformer(config)

    sampler = WindowedMDLMSampler(model, mask_token_id=1, window_size=8, num_steps_per_window=4)
    sampled = sampler.sample(target_tokens=16)

    assert (sampled != 1).all(), "WindowedMDLMSampler left unmasked [MASK] tokens!"
    assert sampled.shape == (1, 16)
