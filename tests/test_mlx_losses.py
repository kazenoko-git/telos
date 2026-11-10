"""Numerical contracts for the Metal loss and its first-order backward pass."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
if not mx.metal.is_available():
    pytest.skip("Metal is required", allow_module_level=True)

from telos.training.mlx_losses import cross_entropy


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize("classes", [1, 17, 257, 8192])
def test_loss_and_gradient_match_fp32_reference(dtype, classes):
    mx.random.seed(7)
    logits = mx.random.normal((2, 3, classes)).astype(dtype)
    targets = mx.random.randint(0, classes, (2, 3))
    weights = mx.array([[0.0, 0.3, 1.0], [2.0, 0.0, 0.7]])

    def reference(x):
        return mx.sum(nn.losses.cross_entropy(x.astype(mx.float32), targets) * weights)

    def fused(x):
        return mx.sum(cross_entropy(x, targets) * weights)

    ref, ref_grad = mx.value_and_grad(reference)(logits)
    actual, actual_grad = mx.compile(mx.value_and_grad(fused))(logits)
    np.testing.assert_allclose(np.array(actual), np.array(ref), rtol=2e-6, atol=2e-6)
    # Conversion to numpy does not support BF16 directly.
    np.testing.assert_allclose(
        np.array(actual_grad.astype(mx.float32)), np.array(ref_grad.astype(mx.float32)),
        rtol=0.009 if dtype == mx.bfloat16 else 0.002, atol=2e-6,
    )


def test_noncontiguous_inputs_and_extreme_logits():
    logits = mx.array([[10000.0, -10000.0, 0.0], [-10000.0, 10000.0, 0.0]])
    logits = mx.transpose(logits)
    targets = mx.array([1, 0, 1])
    actual, gradient = mx.value_and_grad(lambda x: cross_entropy(x, targets, reduction="sum"))(logits)
    ref, ref_grad = mx.value_and_grad(lambda x: nn.losses.cross_entropy(x, targets, reduction="sum"))(logits)
    np.testing.assert_allclose(np.array(actual), np.array(ref), rtol=1e-6)
    np.testing.assert_allclose(np.array(gradient), np.array(ref_grad), atol=1e-6)


@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_reductions(reduction):
    logits = mx.array([[1.0, 2.0], [-1.0, 3.0]])
    targets = mx.array([0, 1])
    np.testing.assert_allclose(
        np.array(cross_entropy(logits, targets, reduction=reduction)),
        np.array(nn.losses.cross_entropy(logits, targets, reduction=reduction)), atol=1e-6,
    )


def test_invalid_shapes_and_reduction_rejected():
    logits = mx.zeros((2, 3))
    with pytest.raises(ValueError, match="shape"):
        cross_entropy(logits, mx.zeros((3,), dtype=mx.int32))
    with pytest.raises(ValueError, match="integer"):
        cross_entropy(logits, mx.zeros((2,)))
    with pytest.raises(ValueError, match="reduction"):
        cross_entropy(logits, mx.zeros((2,), dtype=mx.int32), reduction="invalid")
