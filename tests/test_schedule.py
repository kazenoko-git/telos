"""
Unit tests for COROSred multi-objective schedule and metric tracker.
"""

import pytest
import torch
from telos.training.schedule import (
    COROSredSchedule,
    DynamicMetricTracker,
    compute_vectorized_roc_auc,
)


def test_roc_auc_perfect_ranking():
    # Positives have higher scores than negatives
    scores = torch.tensor([0.1, 0.2, 0.8, 0.9])
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    auc = compute_vectorized_roc_auc(scores, labels)
    assert pytest.approx(auc, rel=1e-3) == 1.0


def test_roc_auc_inverted_ranking():
    # Positives have lower scores than negatives
    scores = torch.tensor([0.9, 0.8, 0.2, 0.1])
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    auc = compute_vectorized_roc_auc(scores, labels)
    assert pytest.approx(auc, rel=1e-3) == 0.0


def test_roc_auc_uniform_labels():
    # All positive or all negative -> returns 0.50
    scores = torch.tensor([0.1, 0.2, 0.3])
    labels = torch.tensor([1.0, 1.0, 1.0])
    assert compute_vectorized_roc_auc(scores, labels) == 0.50


def test_schedule_hold_phase():
    sched = COROSredSchedule(
        max_steps=1000,
        alpha_max=0.85,
        alpha_min=0.20,
        beta_min=0.15,
        beta_max=0.70,
        hold_fraction=0.20,
    )
    # At step 0 to 200, alpha should be exactly alpha_max
    for step in [0, 50, 100, 200]:
        w = sched.get_weights(step)
        assert pytest.approx(w["alpha"], rel=1e-4) == 0.85
        assert pytest.approx(w["beta"], rel=1e-4) == 0.15


def test_schedule_decay_phase_and_floor():
    sched = COROSredSchedule(
        max_steps=1000,
        alpha_max=0.85,
        alpha_min=0.20,
        beta_min=0.15,
        beta_max=0.70,
        hold_fraction=0.20,
        decay_power=2.5,
    )
    # Midway through decay
    w_mid = sched.get_weights(600)
    assert 0.20 < w_mid["alpha"] < 0.85
    assert 0.15 < w_mid["beta"] < 0.70

    # Final step
    w_final = sched.get_weights(1000)
    assert pytest.approx(w_final["alpha"], rel=1e-3) == 0.20  # Never drops below floor!
    assert pytest.approx(w_final["beta"], rel=1e-3) == 0.70


def test_schedule_acc_gate_for_gamma():
    sched = COROSredSchedule(
        max_steps=1000,
        acc_gate_threshold=0.65,
        gamma_max=0.10,
    )
    # When lrh_acc is below threshold, gamma must be 0
    w_low = sched.get_weights(step=500, lrh_acc_ema=0.55)
    assert w_low["gamma"] == 0.0

    # When lrh_acc exceeds threshold, gamma is activated
    w_high = sched.get_weights(step=500, lrh_acc_ema=0.70)
    assert w_high["gamma"] == 0.10


def test_schedule_auc_gate_for_mask_blend():
    sched = COROSredSchedule(
        max_steps=1000,
        auc_gate_min=0.50,
        auc_gate_target=0.75,
    )
    # AUC 0.50 (random guessing) -> mask_blend 0.0 (uniform random)
    w_min = sched.get_weights(step=500, lrh_auc_ema=0.50)
    assert pytest.approx(w_min["mask_blend"], abs=1e-4) == 0.0

    # AUC 0.75 (high ranking fidelity) -> mask_blend 1.0 (confidence routed)
    w_max = sched.get_weights(step=500, lrh_auc_ema=0.75)
    assert pytest.approx(w_max["mask_blend"], abs=1e-4) == 1.0

    # Intermediate AUC
    w_mid = sched.get_weights(step=500, lrh_auc_ema=0.625)
    assert pytest.approx(w_mid["mask_blend"], abs=1e-4) == 0.50


def test_dynamic_metric_tracker_trust_region():
    tracker = DynamicMetricTracker(ema_decay=0.90, trust_region=0.20, rebalance_temp=0.5)

    # Prime with initial losses
    tracker.update_losses(loss_c=6.0, loss_m=6.0)

    # Simulate steps where causal loss collapses by 10x while infill remains high
    for _ in range(30):
        tracker.update_losses(loss_c=0.6, loss_m=6.0)

    alpha_nom = 0.50
    beta_nom = 0.50
    a_eff, b_eff, telem = tracker.get_balanced_weights(alpha_nom, beta_nom)

    # Without trust region, rate_m / rate_c would be ~10x
    # But trust region strictly bounds the adjustment factor to [0.80, 1.20]
    assert telem["clamped"] is True
    assert pytest.approx(telem["factor"], rel=1e-3) == 1.20
    assert pytest.approx(b_eff, rel=1e-3) == 0.60
    assert pytest.approx(a_eff, rel=1e-3) == 0.50
