"""Unit tests for COROSredRefiner."""

import pytest
import torch
from telos.models import TelosTransformer, TelosConfig
from telos.diffusion.refiner import COROSredRefiner


def test_corosred_refiner_preserves_prompt():
    """Verify COROSredRefiner protects prefix prompt tokens."""
    cfg = TelosConfig(vocab_size=60, d_model=32, n_layers=2, n_heads=2, max_seq_len=16, use_reliability_head=True)
    model = TelosTransformer(cfg)

    refiner = COROSredRefiner(
        model,
        mask_token_id=1,
        mode="expected_gain",
        top_k_ratio=0.25,
        radius=1,
        max_gap=1,
        max_passes=2
    )

    prompt = torch.tensor([[10, 20, 30, 40], [10, 25, 35, 45]])
    tail = torch.randint(4, 60, (2, 12))
    input_seq = torch.cat([prompt, tail], dim=1)

    refined, metrics = refiner.refine(input_seq, prompt_len=4, return_metrics=True)

    # Verify prompt tokens are completely unchanged
    assert (refined[:, :4] == prompt).all(), "Prefix prompt tokens were modified!"
    assert refined.shape == input_seq.shape
    assert metrics["passes_executed"] >= 1


def test_corosred_refiner_early_stopping_on_threshold():
    """Verify refiner terminates with 0 passes if gain threshold is not met."""
    cfg = TelosConfig(vocab_size=60, d_model=32, n_layers=2, n_heads=2, max_seq_len=16, use_reliability_head=True)
    model = TelosTransformer(cfg)

    # Set threshold very high so no tokens qualify
    refiner = COROSredRefiner(
        model,
        mask_token_id=1,
        mode="expected_gain",
        threshold=1000.0,
        max_passes=3
    )

    input_seq = torch.randint(4, 60, (2, 16))
    refined, metrics = refiner.refine(input_seq, prompt_len=0, return_metrics=True)

    # Should exit immediately without making edits
    assert (refined == input_seq).all()
    assert metrics["passes_executed"] == 0
