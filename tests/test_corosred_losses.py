"""Unit tests for COROSred Expected Gain Router Loss and Self-Conditioned Phase B Loss."""

import pytest
import torch
from telos.models import TelosTransformer, TelosConfig
from telos.diffusion.corosred import (
    crsr_expected_gain_loss_fn_pytorch,
    crsr_phase_b_self_conditioned_loss_fn_pytorch,
)


def test_expected_gain_loss_pytorch():
    """Verify expected gain loss computes gradients for reliability head."""
    cfg = TelosConfig(vocab_size=60, d_model=32, n_layers=2, n_heads=2, max_seq_len=16, use_reliability_head=True)
    model = TelosTransformer(cfg)

    batch = torch.randint(4, 60, (2, 16), dtype=torch.long)
    loss, metrics = crsr_expected_gain_loss_fn_pytorch(
        model=model,
        batch_seqs=batch,
        vocab_size=60,
        mask_token_id=1,
        mask_prob=0.25
    )

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert "loss" in metrics
    assert "mean_gain" in metrics
    assert "pred_gain" in metrics

    loss.backward()
    # Ensure gradient reaches the reliability head
    rel_grad = next(model.reliability_head.parameters()).grad
    assert rel_grad is not None
    assert not torch.isnan(rel_grad).any()


def test_self_conditioned_phase_b_loss_pytorch():
    """Verify self-conditioned Phase B loss executes cleanly under both branches."""
    cfg = TelosConfig(vocab_size=60, d_model=32, n_layers=2, n_heads=2, max_seq_len=16)
    model = TelosTransformer(cfg)

    batch = torch.randint(4, 60, (2, 16), dtype=torch.long)

    # 1. Force self-conditioning branch (self_cond_prob=1.0)
    loss_sc, metrics_sc = crsr_phase_b_self_conditioned_loss_fn_pytorch(
        model=model,
        batch_seqs=batch,
        vocab_size=60,
        mask_token_id=1,
        mask_prob=0.20,
        self_cond_prob=1.0
    )
    assert isinstance(loss_sc, torch.Tensor)
    assert metrics_sc["self_cond"] == 1.0

    loss_sc.backward()
    assert next(model.parameters()).grad is not None

    model.zero_grad()

    # 2. Force standard clean branch (self_cond_prob=0.0)
    loss_clean, metrics_clean = crsr_phase_b_self_conditioned_loss_fn_pytorch(
        model=model,
        batch_seqs=batch,
        vocab_size=60,
        mask_token_id=1,
        mask_prob=0.20,
        self_cond_prob=0.0
    )
    assert isinstance(loss_clean, torch.Tensor)
    assert metrics_clean["self_cond"] == 0.0
