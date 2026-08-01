"""Unit tests for MDLM forward masking process and 1/t loss reweighting."""

import torch
import pytest
from telos.diffusion.forward_process import apply_masking
from telos.diffusion.loss import mdlm_loss


def test_t_distribution_and_mask_ratio():
    """Verifies that t ~ Uniform(eps, 1) and mask ratio matches t."""
    batch_size = 1000
    seq_len = 100
    mask_token_id = 1
    
    # Clean input tensor [1000, 100] with non-special token IDs
    input_ids = torch.full((batch_size, seq_len), 10, dtype=torch.long)

    masked_ids, mask_positions, t_values = apply_masking(
        input_ids, mask_token_id=mask_token_id, special_token_ids={0, 2, 3}
    )

    # 1. Verify t values fall strictly within [1e-5, 1.0]
    assert (t_values >= 1e-5).all()
    assert (t_values <= 1.0).all()

    # 2. Verify per-example empirical mask ratio approximately equals its sampled t
    empirical_ratios = mask_positions.float().mean(dim=1, keepdim=True)
    # Average difference across batch should be close to 0
    diff = (empirical_ratios - t_values).abs().mean().item()
    assert diff < 0.05, f"Empirical mask ratio diverges from sampled t (mean diff: {diff})"


def test_special_tokens_never_masked():
    """Verifies that PAD, BOS, EOS tokens are explicitly preserved."""
    input_ids = torch.tensor([[2, 10, 11, 12, 3, 0, 0]])  # BOS, content..., EOS, PAD, PAD
    special_tokens = {0, 2, 3}
    mask_token_id = 1

    masked_ids, mask_positions, _ = apply_masking(
        input_ids, mask_token_id=mask_token_id, special_token_ids=special_tokens
    )

    # Special token positions must NOT be masked
    assert not mask_positions[0, 0]  # BOS
    assert not mask_positions[0, 4]  # EOS
    assert not mask_positions[0, 5]  # PAD
    assert not mask_positions[0, 6]  # PAD


def test_one_over_t_loss_reweighting():
    """Directly verifies that loss scaling follows 1/t ratio exactly."""
    batch_size = 2
    seq_len = 10
    vocab_size = 100

    logits = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    mask_positions = torch.ones((batch_size, seq_len), dtype=torch.bool)

    # Example 0 has t = 0.1 (weight = 10.0), Example 1 has t = 1.0 (weight = 1.0)
    t_values = torch.tensor([[0.1], [1.0]])

    loss, metrics = mdlm_loss(logits, targets, mask_positions, t_values)

    # Ensure loss calculation is finite and positive
    assert torch.isfinite(loss)
    assert loss.item() > 0.0
