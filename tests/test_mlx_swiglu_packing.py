"""Check packed SwiGLU behavior and migration from split checkpoints."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
if not mx.metal.is_available():
    pytest.skip("Metal is required", allow_module_level=True)

from mlx.utils import tree_flatten
from telos.models.mlx_components import MLXSwiGLU
from telos.models.mlx_transformer import MLXTelosTransformer


def test_packed_swiglu_preserves_split_forward_and_gradients():
    mx.random.seed(1)
    width = 128
    hidden = 384
    gate = nn.Linear(width, hidden, bias=False)
    up = nn.Linear(width, hidden, bias=False)
    out = nn.Linear(hidden, width, bias=False)
    packed = MLXSwiGLU(width)
    packed.load_weights([
        ("gate_up.weight", mx.concatenate((gate.weight, up.weight), axis=0)),
        ("w3.weight", out.weight),
    ])
    x = mx.random.normal((2, 17, width))

    def reference(inputs):
        return out(nn.silu(gate(inputs)) * up(inputs))

    expected, expected_grad = mx.value_and_grad(lambda inputs: mx.sum(reference(inputs)))(x)
    actual, actual_grad = mx.value_and_grad(lambda inputs: mx.sum(packed(inputs)))(x)
    mx.eval(expected, expected_grad, actual, actual_grad)
    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(np.array(actual_grad), np.array(expected_grad), rtol=2e-6, atol=2e-6)


def test_transformer_loads_legacy_split_swiglu_weights():
    mx.random.seed(4)
    kwargs = dict(vocab_size=64, d_model=64, n_layers=2, n_heads=1, precision="float32")
    current = MLXTelosTransformer(**kwargs)
    legacy = []
    for name, value in tree_flatten(current.parameters()):
        if ".mlp.gate_up.weight" in name:
            gate, up = mx.split(value, 2, axis=0)
            legacy.extend([
                (name.replace(".mlp.gate_up.weight", ".mlp.w1.weight"), gate),
                (name.replace(".mlp.gate_up.weight", ".mlp.w2.weight"), up),
            ])
        else:
            legacy.append((name, value))
    restored = MLXTelosTransformer(**kwargs)
    restored.load_weights(legacy, strict=True)
    tokens = mx.array([[2, 8, 11, 3]])
    np.testing.assert_allclose(
        np.array(restored(tokens)), np.array(current(tokens)), rtol=0, atol=0,
    )
