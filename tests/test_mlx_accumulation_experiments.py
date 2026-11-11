"""Exact parity checks for isolated accumulation experiments."""
import numpy as np
import pytest

mx = pytest.importorskip('mlx.core')
nn = pytest.importorskip('mlx.nn')
optim = pytest.importorskip('mlx.optimizers')
if not mx.metal.is_available():
    pytest.skip('Metal is required', allow_module_level=True)
from mlx.utils import tree_flatten
from telos.training.mlx_step import MLXTrainingStep
from scripts.mlx_accumulation_experiments import ExperimentalAccumulationStep

@pytest.mark.parametrize('master', [False, True])
@pytest.mark.parametrize('accumulation', [2, 8])
@pytest.mark.parametrize('mode', ['pipeline_accumulation', 'compiled_accumulation', 'defer_last_microbatch'])
def test_accumulation_variants_preserve_weights_optimizer_and_rng(master, accumulation, mode):
    def run(pipeline):
        mx.random.seed(73)
        model = nn.Linear(8, 4)
        if master:
            model.set_dtype(mx.bfloat16)
        model.use_master_weights = master
        opt = optim.AdamW(1e-3, betas=[.9, .95], bias_correction=True)

        def loss(m, x):
            y = m(x) * mx.random.uniform(shape=(3, 4))
            value = mx.mean(y.astype(mx.float32) ** 2)
            return value, value

        grad_fn = nn.value_and_grad(model, loss)

        def micro(x):
            (a, b), g = grad_fn(model, x)
            return a, b, g

        state = [model.state, mx.random.state]
        fn = mx.compile(micro, inputs=state, outputs=state)
        step = (ExperimentalAccumulationStep(model, opt, fn, **{mode: True}) if pipeline
                else MLXTrainingStep(model, opt, fn))
        dtype = mx.bfloat16 if master else mx.float32
        losses = []
        for lr in [1e-3, 0., 2e-3, 1e-4]:
            opt.learning_rate = lr
            batches = [mx.full((3, 8), (i + 1) / accumulation, dtype=dtype)
                       for i in range(accumulation)]
            losses.append(step(batches, grad_accum=accumulation)[0].item())
        values = [model.parameters(), opt.state, mx.random.uniform(shape=(16,))]
        if master:
            values.append(opt.master_params)
        mx.eval(values)
        return losses, [(k, np.array(v.astype(mx.float32) if getattr(v, 'dtype', None) == mx.bfloat16 else v))
                        for k, v in tree_flatten(values)]

    ref_loss, reference = run(False)
    actual_loss, actual = run(True)
    np.testing.assert_array_equal(actual_loss, ref_loss)
    assert [k for k, _ in actual] == [k for k, _ in reference]
    for (_, expected), (_, observed) in zip(reference, actual):
        np.testing.assert_array_equal(observed, expected)
