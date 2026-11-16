"""DPO must learn when policy and frozen reference initially agree."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")


def test_dpo_has_correct_gradient_when_policy_matches_reference():
    from telos.posttrain.trainer import _mlx_preference_loss
    zero = mx.array([0.])
    loss, gradient = mx.value_and_grad(
        lambda chosen: _mlx_preference_loss(chosen, zero, zero, zero, .1))(zero)
    assert loss.item() == pytest.approx(np.log(2), rel=1e-6)
    assert gradient.item() == pytest.approx(-.05, rel=1e-6)
    for value in (-10000., 10000.):
        loss, gradient = mx.value_and_grad(
            lambda chosen: _mlx_preference_loss(chosen, zero, zero, zero, .1))(mx.array([value]))
        assert mx.isfinite(loss).item()
        assert mx.all(mx.isfinite(gradient)).item()


