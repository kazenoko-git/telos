"""Checkpointing must preserve gradients for every transformer parameter."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
if not mx.metal.is_available():
    pytest.skip("Metal is required", allow_module_level=True)

from mlx.utils import tree_flatten
from telos.models.mlx_transformer import MLXTelosTransformer
from telos.training.mlx_losses import cross_entropy


@pytest.mark.parametrize("precision", ["float32", "bfloat16"])
def test_checkpoint_gradients_match_uncheckpointed(precision):
    mx.random.seed(42)
    kwargs = dict(vocab_size=128, d_model=64, n_layers=2, n_heads=2, precision=precision, is_causal=True)
    plain = MLXTelosTransformer(**kwargs)
    checkpointed = MLXTelosTransformer(**kwargs, use_grad_checkpoint=True)
    checkpointed.update(plain.parameters())
    tokens = mx.random.randint(4, 128, (2, 16))

    def loss(model):
        return cross_entropy(model(tokens)[:, :-1], tokens[:, 1:], reduction="mean")

    ref_loss, ref_grad = nn.value_and_grad(plain, loss)(plain)
    def candidate():
        return nn.value_and_grad(checkpointed, loss)(checkpointed)
    actual_loss, actual_grad = mx.compile(candidate, inputs=[checkpointed.state])()
    np.testing.assert_allclose(np.array(actual_loss), np.array(ref_loss), atol=2e-6)
    for (name, expected), (actual_name, actual) in zip(tree_flatten(ref_grad), tree_flatten(actual_grad)):
        assert name == actual_name
        assert mx.abs(actual).sum().item() > 0, f"Checkpointing disconnected {name}"
        np.testing.assert_allclose(
            np.array(actual.astype(mx.float32)), np.array(expected.astype(mx.float32)),
            rtol=0.02 if precision == "bfloat16" else 1e-4,
            atol=2e-5 if precision == "bfloat16" else 2e-6,
        )
