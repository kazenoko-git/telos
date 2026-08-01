"""Unit test for iterative unmasking sampler."""

import torch
from telos.model.transformer import TelosTransformer, TelosConfig
from telos.diffusion.sampler import MDLMSampler


def test_sampler_unmasks_all_tokens():
    """Verifies sampler removes all [MASK] tokens upon completion."""
    config = TelosConfig(vocab_size=50, d_model=32, n_layers=2, n_heads=2, max_seq_len=16)
    model = TelosTransformer(config)
    
    sampler = MDLMSampler(model, mask_token_id=1, num_steps=5, schedule="linear")
    
    # Run sampling
    sampled = sampler.sample(seq_len=16)

    # Ensure output sequence contains NO remaining [MASK] tokens (ID 1)
    assert (sampled != 1).all(), "Sampler left unmasked [MASK] tokens!"
    assert sampled.shape == (1, 16)
