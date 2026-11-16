"""Numerical and state checks for the opt-in packed QKV/RoPE Metal VJP."""
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
if not mx.metal.is_available():
    pytest.skip("Metal is required", allow_module_level=True)

from mlx.utils import tree_flatten
from telos.models.mlx_components import MLXBlock
from telos.models.mlx_qkv_rope import make_rope_table, experimental_qkv_rope
from telos.models.mlx_transformer import MLXTelosTransformer
from telos.diffusion.ar import ar_loss_fn_mlx
from telos.training.mlx_step import MLXTrainingStep, build_mlx_accumulation_step


def relative_l2(actual, expected):
    a, b = actual.astype(mx.float32), expected.astype(mx.float32)
    return float((mx.sqrt(mx.sum((a-b)**2)) / mx.maximum(mx.sqrt(mx.sum(b*b)), 1e-12)).item())


def reference(packed):
    batch, sequence, width = packed.shape
    heads = width // 192
    q, k, v = [p.reshape(batch, sequence, heads, 64).transpose(0, 2, 1, 3)
               for p in mx.split(packed, 3, axis=-1)]
    return (mx.fast.rope(q, 64, traditional=True, base=10000.0, scale=1.0, offset=0),
            mx.fast.rope(k, 64, traditional=True, base=10000.0, scale=1.0, offset=0), v)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_projection_output_input_and_parameter_gradients(dtype):
    mx.random.seed(81)
    x = mx.random.normal((2, 17, 128)).astype(dtype)
    weight = (mx.random.normal((384, 128)) * 0.02).astype(dtype)
    table = make_rope_table(17)
    cotangents = [mx.random.normal((2, 2, 17, 64)).astype(dtype) for _ in range(3)]
    expected = reference(x @ weight.T)
    actual = experimental_qkv_rope(x @ weight.T, table)
    mx.eval(actual, expected)
    tolerance = 2e-5 if dtype == mx.float32 else 0.005
    assert all(relative_l2(a, b) < tolerance for a, b in zip(actual, expected))
    assert all(bool(mx.array_equal(a, b).item()) for a, b in zip(actual, expected))
    assert bool(mx.array_equal(actual[2], expected[2]).item())

    def objective(inputs, parameters, fused):
        packed = inputs @ parameters.T
        outputs = experimental_qkv_rope(packed, table) if fused else reference(packed)
        return sum(mx.sum(a.astype(mx.float32) * g.astype(mx.float32))
                   for a, g in zip(outputs, cotangents))
    expected_grad = mx.grad(lambda x, w: objective(x, w, False), argnums=(0, 1))(x, weight)
    actual_grad = mx.grad(lambda x, w: objective(x, w, True), argnums=(0, 1))(x, weight)
    mx.eval(expected_grad, actual_grad)
    assert all(relative_l2(a, b) < tolerance for a, b in zip(actual_grad, expected_grad))
    assert all(bool(mx.array_equal(a, b).item()) for a, b in zip(actual_grad, expected_grad))


def test_table_is_not_model_state_and_unsupported_shapes_fall_back():
    for width, heads, kv_heads in [(128, 2, 2), (128, 4, 4), (128, 2, 1)]:
        mx.random.seed(91)
        block = MLXBlock(width, heads, kv_heads, is_causal=True)
        assert block.use_experimental_qkv_rope is False
        before = {k: p.shape for k, p in tree_flatten(block.state)}
        x = mx.random.normal((2, 17, width))
        expected = block(x)
        mx.eval(expected)
        object.__setattr__(block, "_qkv_rope_table", make_rope_table(16))
        block.use_experimental_qkv_rope = True
        # Oversized sequence, non-64 heads and GQA use the reference path.
        actual = block(x)
        mx.eval(actual)
        assert bool(mx.array_equal(actual, expected).item())
        assert before == {k: p.shape for k, p in tree_flatten(block.state)}


@pytest.mark.parametrize("checkpoint", [False, True])
def test_masked_ar_loss_and_compiled_accumulated_optimizer_updates(checkpoint):
    import mlx.optimizers as optim
    mx.random.seed(101)
    kwargs = dict(vocab_size=97, d_model=128, n_layers=1, n_heads=2,
                  is_causal=True, precision="bfloat16", use_grad_checkpoint=checkpoint)
    baseline = MLXTelosTransformer(**kwargs)
    candidate = MLXTelosTransformer(**kwargs)
    candidate.load_weights(list(tree_flatten(baseline.parameters())))
    table = make_rope_table(17)
    for layer in candidate.layers:
        object.__setattr__(layer, "_qkv_rope_table", table)
        layer.use_experimental_qkv_rope = True
    # One entirely masked sequence and one content sequence test the actual
    # per-example content normalization, including zero content targets.
    tokens = mx.array([[0, 1, 2, 3]*4+[0], list(range(4, 21))], dtype=mx.int32)
    special = mx.array([True]*4 + [False]*93)
    expected_loss, expected_grad = nn.value_and_grad(baseline, ar_loss_fn_mlx)(baseline, tokens, 97, special)
    actual_loss, actual_grad = nn.value_and_grad(candidate, ar_loss_fn_mlx)(candidate, tokens, 97, special)
    mx.eval(expected_loss, expected_grad, actual_loss, actual_grad)
    assert abs(float((actual_loss[0]-expected_loss[0]).item())) < 0.001
    expected_flat, actual_flat = dict(tree_flatten(expected_grad)), dict(tree_flatten(actual_grad))
    assert expected_flat.keys() == actual_flat.keys()
    for name, gradient in expected_flat.items():
        assert relative_l2(actual_flat[name], gradient) < 0.01, name
        assert bool(mx.array_equal(actual_flat[name], gradient).item()), name

    def prepare(model):
        model.use_master_weights = True
        optimizer = optim.AdamW(learning_rate=1e-4, betas=[0.9, 0.95], weight_decay=0.01)
        vg = nn.value_and_grad(model, ar_loss_fn_mlx)
        def microbatch(ids):
            (loss, metric), gradients = vg(model, ids, 97, special)
            return loss, metric, gradients
        state = [model.state, mx.random.state]
        compiled = mx.compile(microbatch, inputs=state, outputs=state)
        step = MLXTrainingStep(model, optimizer, compiled,
                               accumulation_step=build_mlx_accumulation_step(microbatch, state))
        return step, optimizer

    ref_step, ref_opt = prepare(baseline)
    new_step, new_opt = prepare(candidate)
    for update in range(4):
        ref_opt.learning_rate = new_opt.learning_rate = 1e-4 / (update+1)
        expected, _ = ref_step([tokens, tokens], grad_accum=2)
        actual, _ = new_step([tokens, tokens], grad_accum=2)
        assert abs(float((actual-expected).item())) < 0.005
        for (name, a), (other, b) in zip(tree_flatten(candidate.parameters()), tree_flatten(baseline.parameters())):
            assert name == other
            assert relative_l2(a, b) < 0.005, name
            assert bool(mx.array_equal(a, b).item()), name
        for (name, a), (other, b) in zip(tree_flatten(new_opt.master_params), tree_flatten(ref_opt.master_params)):
            assert name == other
            assert relative_l2(a, b) < 0.005, name
            assert bool(mx.array_equal(a, b).item()), name
        new_state, ref_state = dict(tree_flatten(new_opt.state)), dict(tree_flatten(ref_opt.state))
        assert new_state.keys() == ref_state.keys()
        for name, value in ref_state.items():
            if isinstance(value, mx.array):
                assert relative_l2(new_state[name], value) < 0.02, name
                assert bool(mx.array_equal(new_state[name], value).item()), name


def test_rejects_malformed_table():
    with pytest.raises(ValueError, match="RoPE table"):
        experimental_qkv_rope(mx.ones((1, 17, 192)), mx.ones((16, 32, 2)))
