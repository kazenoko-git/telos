"""Cached accumulated updates must honor changing clipping and accumulation."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
optim = pytest.importorskip("mlx.optimizers")
from mlx.utils import tree_flatten, tree_map
from telos.training.core import clip_grad_norm_mlx, execute_mlx_training_step

if not mx.metal.is_available():
    pytest.skip("Metal required", allow_module_level=True)


@pytest.mark.parametrize("master", [False, True])
def test_cached_optimizer_matches_reference_with_changing_clip_and_scale(master):
    def run(reference):
        mx.random.seed(314)
        model = nn.Linear(16, 8)
        model.use_master_weights = master
        if master:
            model.set_dtype(mx.bfloat16)
        optimizer = optim.AdamW(1e-3, betas=[.9, .95], bias_correction=True)
        def objective(module, x):
            value = mx.mean((module(x).astype(mx.float32) - .2) ** 2)
            return value, value
        grad = nn.value_and_grad(model, objective)
        def micro(x):
            (loss, metric), grads = grad(model, x)
            return loss, metric, grads
        history = []
        schedule = [(2, .1, .001), (8, .1, 0.), (2, 0., .002), (4, 1., .0001), (8, .01, .001)]
        for step, (accumulation, clip, lr) in enumerate(schedule):
            optimizer.learning_rate = lr
            batches = [mx.full((3, 16), (i + 1) / accumulation,
                               dtype=mx.bfloat16 if master else mx.float32)
                       for i in range(accumulation)]
            if not reference or step == 0:
                execute_mlx_training_step(model, optimizer, micro, iter(batches),
                                          accumulation, clip, step == 0,
                                          async_accumulation=False)
            else:
                summed = None
                for batch in batches:
                    _, _, current = micro(batch)
                    summed = current if summed is None else tree_map(lambda a, b: a + b, summed, current)
                    mx.eval(summed)
                if clip > 0:
                    summed, _ = clip_grad_norm_mlx(summed, max_norm=clip, scale=float(accumulation))
                else:
                    summed = tree_map(lambda g: g / float(accumulation), summed)
                if master:
                    optimizer.master_params = optimizer.apply_gradients(summed, optimizer.master_params)
                    model.update(tree_map(lambda p: p.astype(mx.bfloat16), optimizer.master_params))
                else:
                    optimizer.update(model, summed)
            state = [model.parameters(), optimizer.state, getattr(optimizer, "master_params", {})]
            mx.eval(state)
            history.append({key: np.asarray(value.astype(mx.float32) if value.dtype == mx.bfloat16 else value).copy()
                            for key, value in tree_flatten(state)})
        return history
    for actual, expected in zip(run(False), run(True)):
        assert actual.keys() == expected.keys()
        for key in expected:
            np.testing.assert_allclose(actual[key], expected[key], rtol=3e-5, atol=3e-6)
