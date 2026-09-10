"""
Unit tests for unified COROSred heterogeneous step and routing cache.
"""

import pytest
import torch
from telos.models.config import TelosConfig
from telos.models.transformer import TelosTransformer
from telos.diffusion.corosred_unified import (
    RoutingMaskCache,
    corosred_unified_step_pytorch,
)
from telos.training.schedule import COROSredSchedule, DynamicMetricTracker


@pytest.fixture
def toy_model():
    cfg = TelosConfig(
        vocab_size=128,
        d_model=64,
        n_heads=2,
        n_layers=2,
        max_seq_len=32,
        use_reliability_head=True,
    )
    return TelosTransformer(cfg)


def test_routing_mask_cache(toy_model):
    cache = RoutingMaskCache(refresh_every_steps=10, k_targeted_ratio=0.70)
    assert cache.should_refresh(0) is True

    seqs = torch.randint(4, 128, (8, 32))
    cache.refresh(toy_model, seqs, current_step=0, mask_prob=0.15)
    assert cache.should_refresh(5) is False
    assert cache.should_refresh(10) is True

    # Test popping mask with blend = 0.0 (uniform)
    corr_u, mask_u = cache.pop_or_generate_mask(seqs[:4], mask_token_id=1, mask_prob=0.15, mask_blend=0.0)
    assert corr_u.shape == (4, 32)
    assert mask_u.shape == (4, 32)
    assert mask_u[:, 0].sum() == 0  # BOS preserved

    # Test popping mask with blend = 1.0 (confidence routed from cache)
    corr_c, mask_c = cache.pop_or_generate_mask(seqs[:4], mask_token_id=1, mask_prob=0.15, mask_blend=1.0)
    assert corr_c.shape == (4, 32)
    assert mask_c.shape == (4, 32)
    assert mask_c[:, 0].sum() == 0  # BOS preserved


def test_corosred_unified_step_forward_backward(toy_model):
    batch = torch.randint(4, 128, (4, 32))
    schedule = COROSredSchedule(max_steps=100)
    # Set step where gamma is active
    weights = schedule.get_weights(step=50, lrh_acc_ema=0.75)
    assert weights["gamma"] > 0.0
    tracker = DynamicMetricTracker()

    loss, metrics = corosred_unified_step_pytorch(
        model=toy_model,
        batch_seqs=batch,
        vocab_size=128,
        schedule_weights=weights,
        metric_tracker=tracker,
    )

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert not torch.isnan(loss)
    assert "causal_ce" in metrics
    assert "infill_ce" in metrics
    assert "lrh_acc" in metrics
    assert "lrh_auc" in metrics

    # Verify backward pass computes gradients on both backbone and reliability head
    loss.backward()

    # Backbone embeddings has grad
    assert toy_model.tok_embeddings.weight.grad is not None
    assert not torch.all(toy_model.tok_embeddings.weight.grad == 0)

    # Reliability head has grad
    assert toy_model.reliability_head[0].weight.grad is not None
    assert not torch.all(toy_model.reliability_head[0].weight.grad == 0)


def test_dual_metric_monitor(toy_model):
    from telos.training.monitor import DualMetricMonitor
    import numpy as np

    dummy_val = np.random.randint(4, 128, (50, 32), dtype=np.int32)
    monitor = DualMetricMonitor(
        val_dataset_matrix=dummy_val,
        vocab_size=128,
        seq_len=32,
        device=torch.device("cpu"),
        num_probe_sequences=20,
    )

    res = monitor.evaluate(toy_model, current_step=100, batch_size=10)
    assert res["step"] == 100
    assert 0.0 <= res["causal_top1"] <= 1.0
    assert 0.0 <= res["infill_top1"] <= 1.0
    assert res["causal_ce"] > 0.0
    assert res["infill_ce"] > 0.0
    assert monitor.check_divergence() is None
