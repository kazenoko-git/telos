"""Unit test suite for iterative unmasking samplers."""

import torch
from mdiff.model.transformer import TelosTransformer, TelosConfig
from mdiff.diffusion.sampler import MDLMSampler


def test_standard_mdlm_sampler_cosine():
    """Verifies MDLMSampler unmasks all tokens with cosine schedule."""
    config = TelosConfig(vocab_size=50, d_model=32, n_layers=2, n_heads=2, max_seq_len=16)
    model = TelosTransformer(config)

    sampler = MDLMSampler(model, mask_token_id=1, num_steps=5, schedule="cosine")
    sampled = sampler.sample(seq_len=16)

    assert (sampled != 1).all(), "MDLMSampler left unmasked [MASK] tokens!"
    assert sampled.shape == (1, 16)


def test_standard_mdlm_sampler_linear():
    """Verifies MDLMSampler unmasks all tokens with linear schedule."""
    config = TelosConfig(vocab_size=50, d_model=32, n_layers=2, n_heads=2, max_seq_len=16)
    model = TelosTransformer(config)

    sampler = MDLMSampler(model, mask_token_id=1, num_steps=5, schedule="linear")
    sampled = sampler.sample(seq_len=16)

    assert (sampled != 1).all(), "MDLMSampler left unmasked [MASK] tokens!"
    assert sampled.shape == (1, 16)


def test_mdlm_sampler_with_prompt():
    """Verifies MDLMSampler respects prefix prompt tokens during generation."""
    config = TelosConfig(vocab_size=50, d_model=32, n_layers=2, n_heads=2, max_seq_len=16)
    model = TelosTransformer(config)

    prompt = torch.tensor([[10, 20, 30]], dtype=torch.long)
    sampler = MDLMSampler(model, mask_token_id=1, num_steps=4, schedule="linear")
    sampled = sampler.sample(seq_len=16, prompt_ids=prompt)

    assert (sampled[:, :3] == prompt).all(), "Prompt prefix was overwritten during sampling!"
    assert (sampled != 1).all(), "Unmasked tokens remained in sequence!"
    assert sampled.shape == (1, 16)
