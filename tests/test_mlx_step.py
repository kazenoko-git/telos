"""Full-step parity, dynamic schedules, and compiled random-state regression tests."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
optim = pytest.importorskip("mlx.optimizers")
if not mx.metal.is_available():
    pytest.skip("Metal is required", allow_module_level=True)

from mlx.utils import tree_flatten
from telos.training.mlx_step import MLXTrainingStep


@pytest.mark.parametrize("master", [False, True])
@pytest.mark.parametrize("accumulation", [1, 2])
def test_step_parity_with_reference_and_learning_rate_changes(master, accumulation):
    mx.random.seed(42)
    reference = nn.Linear(8, 4)
    if master:
        reference.set_dtype(mx.bfloat16)
    candidate = nn.Linear(8, 4)
    candidate.update(reference.parameters())
    reference.use_master_weights = candidate.use_master_weights = master
    x = mx.random.normal((3, 8)).astype(mx.bfloat16 if master else mx.float32)

    def make(model, fused):
        optimizer = optim.AdamW(1e-3, betas=[0.9, 0.95], bias_correction=True)
        def loss(m, x):
            value = mx.mean(m(x).astype(mx.float32) ** 2)
            return value, value
        grad_fn = nn.value_and_grad(model, loss)
        def micro(x):
            (loss, metric), grads = grad_fn(model, x)
            return loss, metric, grads
        return MLXTrainingStep(model, optimizer, micro, fused=fused), optimizer

    ref, ref_opt = make(reference, False)
    actual, opt = make(candidate, True)
    for step, lr in enumerate([1e-3, 2e-3, 0.0, 1e-4, 1e-3]):
        ref_opt.learning_rate = opt.learning_rate = lr
        expected = ref([x] * accumulation, grad_accum=accumulation, eval_every_microbatch=True)
        observed = actual([x] * accumulation, grad_accum=accumulation, asynchronous=(step % 2 == 0))
        actual.synchronize()
        np.testing.assert_allclose(np.array(observed[0]), np.array(expected[0]), rtol=2e-5, atol=2e-5)
        if master:
            a, b = opt.master_params, ref_opt.master_params
        else:
            a, b = candidate.parameters(), reference.parameters()
        for (_, left), (_, right) in zip(tree_flatten(a), tree_flatten(b)):
            np.testing.assert_allclose(np.array(left), np.array(right), rtol=2e-5, atol=2e-6)


def test_compiled_random_state_advances_and_replays():
    def run():
        mx.random.seed(17)
        model = nn.Linear(4, 4)
        opt = optim.AdamW(0.0)
        def loss(m, x):
            value = mx.mean((m(x) * mx.random.uniform(shape=x.shape)) ** 2)
            return value, value
        grad_fn = nn.value_and_grad(model, loss)
        def micro(x):
            (a, b), g = grad_fn(model, x)
            return a, b, g
        step = MLXTrainingStep(model, opt, micro)
        x = mx.ones((8, 4))
        return [step([x])[0].item() for _ in range(5)]
    first, second = run(), run()
    assert len(set(first)) == 5
    np.testing.assert_allclose(first, second, rtol=0, atol=0)


@pytest.mark.parametrize('master', [False, True])
@pytest.mark.parametrize('accumulations', [(2, 4, 2, 4), (2, 1, 2, 1)])
def test_async_accumulated_updates_preserve_full_state(master, accumulations):
    def run(asynchronous):
        mx.random.seed(921)
        model = nn.Linear(8, 4)
        model.use_master_weights = master
        if master:
            model.set_dtype(mx.bfloat16)
        opt = optim.AdamW(1e-3, betas=[.9, .95], bias_correction=True)
        def loss(m, x):
            value = mx.mean((m(x).astype(mx.float32) * mx.random.uniform(shape=(3, 4))) ** 2)
            return value, value
        grad = nn.value_and_grad(model, loss)
        def micro(x):
            (a, b), g = grad(model, x)
            return a, b, g
        from telos.training.mlx_step import build_training_step
        step = build_training_step(model, opt, micro, [model.state, mx.random.state], {}, grad_clip=.1)
        losses = []
        for index, (accum, lr) in enumerate(zip(accumulations, [.001, .002, 0., .0001])):
            opt.learning_rate = lr
            losses.append(step((mx.full((3, 8), i + .5) for i in range(accum)),
                               grad_accum=accum, asynchronous=asynchronous))
            assert (step._pending is not None) == (asynchronous and index > 0)
        step.synchronize()
        assert step._pending is None
        state = [losses, model.parameters(), opt.state, getattr(opt, 'master_params', {}), list(mx.random.state)]
        mx.eval(state)
        return {k: np.asarray(v.astype(mx.float32) if v.dtype == mx.bfloat16 else v).copy()
                for k, v in tree_flatten(state)}
    expected, actual = run(False), run(True)
    assert actual.keys() == expected.keys()
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
