"""Parity checks for compiling backward work with gradient accumulation."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
optim = pytest.importorskip("mlx.optimizers")
if not mx.metal.is_available():
    pytest.skip("Metal is required", allow_module_level=True)

from mlx.utils import tree_flatten, tree_map
from telos.training.mlx_step import MLXTrainingStep, build_mlx_accumulation_step


def _arrays(tree):
    values = [value for _, value in tree_flatten(tree)]
    mx.eval(*values)
    return [(key, np.asarray(value.astype(mx.float32) if value.dtype == mx.bfloat16 else value))
            for key, value in tree_flatten(tree)]


def _make_step(model, *, compiled_accumulation):
    def loss_fn(module, x):
        noise = mx.random.uniform(shape=(x.shape[0], 4))
        output = module(x) * noise
        loss = mx.mean(output.astype(mx.float32) ** 2)
        return loss, loss

    grad_fn = nn.value_and_grad(model, loss_fn)

    def microbatch(x):
        (loss, metric), grads = grad_fn(model, x)
        return loss, metric, grads

    targets = [model.state, mx.random.state]
    compiled_micro = mx.compile(microbatch, inputs=targets, outputs=targets)
    accumulation = build_mlx_accumulation_step(microbatch, targets) if compiled_accumulation else None
    optimizer = optim.AdamW(1e-3, betas=[0.9, 0.95], bias_correction=True)
    model.set_dtype(mx.bfloat16)
    model.use_master_weights = True
    return MLXTrainingStep(model, optimizer, compiled_micro, accumulation_step=accumulation), optimizer


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("accumulation", [2, 8])
def test_compiled_backward_accumulation_matches_gradient_sum(dtype, accumulation):
    mx.random.seed(121)
    model = nn.Linear(8, 4)
    model.set_dtype(dtype)
    model.use_master_weights = True

    def loss_fn(module, x):
        output = module(x).astype(mx.float32)
        value = mx.mean((output - 0.25) ** 2)
        return value, value

    grad_fn = nn.value_and_grad(model, loss_fn)

    def microbatch(x):
        (loss, metric), grads = grad_fn(model, x)
        return loss, metric, grads

    targets = [model.state, mx.random.state]
    compiled_micro = mx.compile(microbatch, inputs=targets, outputs=targets)
    combined = build_mlx_accumulation_step(microbatch, targets)
    batches = [mx.full((3, 8), (i + 1) / accumulation, dtype=dtype) for i in range(accumulation)]

    loss_sum = mx.array(0.0, dtype=mx.float32)
    metric_sum = mx.array(0.0, dtype=mx.float32)
    grads = None
    for i, batch in enumerate(batches):
        if i == 0:
            loss, metric, grads = compiled_micro(batch)
            loss_sum = loss_sum + loss
            metric_sum = metric_sum + metric
        else:
            grads, loss_sum, metric_sum = combined(grads, loss_sum, metric_sum, batch)
    mx.eval(loss_sum, metric_sum, grads)

    # Reference path uses the same microbatch gradients and left-to-right additions.
    ref_loss = mx.array(0.0, dtype=mx.float32)
    ref_metric = mx.array(0.0, dtype=mx.float32)
    ref_grads = None
    for batch in batches:
        loss, metric, current = compiled_micro(batch)
        ref_loss = ref_loss + loss
        ref_metric = ref_metric + metric
        ref_grads = current if ref_grads is None else tree_map(lambda a, b: a + b, ref_grads, current)
        mx.eval(ref_loss, ref_metric, ref_grads)
    np.testing.assert_allclose(np.array(loss_sum), np.array(ref_loss), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.array(metric_sum), np.array(ref_metric), rtol=1e-5, atol=1e-6)
    expected, actual = _arrays(ref_grads), _arrays(grads)
    assert [key for key, _ in actual] == [key for key, _ in expected]
    tolerance = 2e-3 if dtype == mx.bfloat16 else 2e-6
    for (_, want), (_, got) in zip(expected, actual):
        np.testing.assert_allclose(got, want, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("accumulation", [2, 8])
def test_compiled_backward_accumulation_preserves_training_state_and_rng(accumulation):
    def run(compiled_accumulation):
        mx.random.seed(73)
        model = nn.Linear(8, 4)
        step, optimizer = _make_step(model, compiled_accumulation=compiled_accumulation)
        losses = []
        for lr in [1e-3, 0.0, 2e-3]:
            optimizer.learning_rate = lr
            batches = [mx.full((3, 8), (i + 1) / accumulation,
                               dtype=mx.float32) for i in range(accumulation)]
            losses.append(step(batches, grad_accum=accumulation)[0])
        state = [model.parameters(), optimizer.state, optimizer.master_params,
                 mx.random.uniform(shape=(16,))]
        mx.eval(*[value for _, value in tree_flatten(state)])
        return losses, _arrays(state)

    expected_losses, expected = run(False)
    actual_losses, actual = run(True)
    np.testing.assert_allclose(np.asarray(actual_losses), np.asarray(expected_losses), rtol=1e-5, atol=1e-6)
    assert [key for key, _ in actual] == [key for key, _ in expected]
    for (_, want), (_, got) in zip(expected, actual):
        np.testing.assert_allclose(got, want, rtol=3e-3, atol=3e-4)
