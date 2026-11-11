"""Numerical checks for the opt-in packed SwiGLU Metal VJP."""

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
if not mx.metal.is_available():
    pytest.skip("Metal is required", allow_module_level=True)

from telos.models.mlx_components import MLXSwiGLU, _experimental_packed_swiglu
from mlx.utils import tree_flatten
from telos.diffusion.ar import ar_loss_fn_mlx
from telos.models.mlx_transformer import MLXTelosTransformer


def _relative_l2(actual, expected):
    actual = actual.astype(mx.float32)
    expected = expected.astype(mx.float32)
    numerator = mx.sqrt(mx.sum(mx.square(actual - expected)))
    denominator = mx.sqrt(mx.sum(mx.square(expected)))
    return float((numerator / mx.maximum(denominator, 1e-12)).item())


def test_packed_swiglu_is_disabled_by_default():
    module = MLXSwiGLU(128)
    assert module.use_experimental_packed_kernel is False


def test_experimental_packed_swiglu_bf16_output_and_vjp():
    mx.random.seed(17)
    rows, hidden = 34, 384
    packed = mx.random.normal((rows, 2 * hidden)).astype(mx.bfloat16)
    cotangent = mx.random.normal((rows, hidden)).astype(mx.bfloat16)

    def reference(x):
        gate, up = mx.split(x, 2, axis=-1)
        return nn.silu(gate) * up

    def reference_loss(x):
        return mx.sum(reference(x).astype(mx.float32) * cotangent.astype(mx.float32))

    def experimental_loss(x):
        output = _experimental_packed_swiglu(x)
        return mx.sum(output.astype(mx.float32) * cotangent.astype(mx.float32))

    expected = reference(packed)
    expected_loss, expected_grad = mx.value_and_grad(reference_loss)(packed)
    actual = _experimental_packed_swiglu(packed)
    actual_loss, actual_grad = mx.value_and_grad(experimental_loss)(packed)
    mx.eval(expected, expected_loss, expected_grad, actual, actual_loss, actual_grad)

    assert _relative_l2(actual, expected) < 0.01
    assert _relative_l2(actual_grad, expected_grad) < 0.01
    assert abs(float(actual_loss.item() - expected_loss.item())) < 1.0


def test_module_opt_in_keeps_parameter_shapes_and_close_gradients():
    mx.random.seed(29)
    width = 128
    reference = MLXSwiGLU(width)
    experimental = MLXSwiGLU(width)
    experimental.update(reference.state)
    experimental.use_experimental_packed_kernel = True

    x = mx.random.normal((2, 17, width)).astype(mx.bfloat16)
    cotangent = mx.random.normal((2, 17, width)).astype(mx.bfloat16)

    def loss(module, inputs):
        output = module(inputs)
        return mx.sum(output.astype(mx.float32) * cotangent.astype(mx.float32))

    reference_loss, reference_grads = nn.value_and_grad(reference, loss)(reference, x)
    experimental_loss, experimental_grads = nn.value_and_grad(experimental, loss)(experimental, x)
    mx.eval(
        reference_loss,
        experimental_loss,
        *[value for _, value in tree_flatten(reference_grads)],
        *[value for _, value in tree_flatten(experimental_grads)],
    )

    assert abs(float(reference_loss.item() - experimental_loss.item())) < 1.0
    reference_flat = dict(tree_flatten(reference_grads))
    experimental_flat = dict(tree_flatten(experimental_grads))
    assert reference_flat.keys() == experimental_flat.keys()
    for name in reference_flat:
        assert _relative_l2(experimental_flat[name], reference_flat[name]) < 0.01, name


def test_experimental_path_preserves_small_bf16_ar_loss_and_gradients():
    mx.random.seed(41)
    kwargs = dict(
        vocab_size=64,
        d_model=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=4,
        is_causal=True,
        precision="bfloat16",
    )
    reference = MLXTelosTransformer(**kwargs)
    experimental = MLXTelosTransformer(**kwargs)
    experimental.update(reference.state)
    for layer in experimental.layers:
        layer.mlp.use_experimental_packed_kernel = True

    tokens = mx.random.randint(0, kwargs["vocab_size"], (2, 16))
    reference_fn = nn.value_and_grad(reference, ar_loss_fn_mlx)
    experimental_fn = nn.value_and_grad(experimental, ar_loss_fn_mlx)
    (reference_loss, _), reference_grads = reference_fn(reference, tokens, kwargs["vocab_size"])
    (experimental_loss, _), experimental_grads = experimental_fn(experimental, tokens, kwargs["vocab_size"])
    reference_flat = dict(tree_flatten(reference_grads))
    experimental_flat = dict(tree_flatten(experimental_grads))
    mx.eval(
        reference_loss,
        experimental_loss,
        *reference_flat.values(),
        *experimental_flat.values(),
    )

    assert abs(float(reference_loss.item() - experimental_loss.item())) < 1e-3
    assert reference_flat.keys() == experimental_flat.keys()
    for name in reference_flat:
        assert _relative_l2(experimental_flat[name], reference_flat[name]) < 0.01, name
