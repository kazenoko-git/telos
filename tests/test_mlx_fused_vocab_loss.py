"""Parity checks for the experimental tiled autoregressive output loss."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
mx_nn = pytest.importorskip("mlx.nn")
if not mx.metal.is_available():
    pytest.skip("Metal is required", allow_module_level=True)

from mlx.utils import tree_flatten
from telos.diffusion.ar import ar_loss_fn_mlx
from telos.models.mlx_transformer import MLXTelosTransformer


def test_tiled_vocab_loss_matches_full_projection_loss_and_gradients():
    mx.random.seed(19)
    kwargs = dict(vocab_size=257, d_model=64, n_layers=2, n_heads=1, precision="float32")
    full = MLXTelosTransformer(**kwargs, is_causal=True)
    tiled = MLXTelosTransformer(**kwargs, is_causal=True)
    tiled.load_weights(list(tree_flatten(full.parameters())))
    tiled.fused_vocab_loss_enabled = True
    batch = mx.random.randint(4, kwargs["vocab_size"], (2, 17))
    special = mx.array([True, True, True, True] + [False] * (kwargs["vocab_size"] - 4))
    full_loss = mx_nn.value_and_grad(
        full, lambda model, tokens: ar_loss_fn_mlx(model, tokens, kwargs["vocab_size"], special)[0]
    )
    tiled_loss = mx_nn.value_and_grad(
        tiled, lambda model, tokens: ar_loss_fn_mlx(model, tokens, kwargs["vocab_size"], special)[0]
    )
    expected, expected_grad = full_loss(full, batch)
    actual, actual_grad = tiled_loss(tiled, batch)
    mx.eval(expected, expected_grad, actual, actual_grad)
    # Tiled log-sum-exp changes FP32 reduction order; observed loss drift is
    # under 7e-5 on this small vocab/model comparison.
    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=2e-5, atol=1e-4)
    for (name, want), (actual_name, got) in zip(tree_flatten(expected_grad), tree_flatten(actual_grad)):
        assert name == actual_name
        np.testing.assert_allclose(np.array(got), np.array(want), rtol=0.02, atol=3e-4)


def test_tiled_vocab_loss_rejects_untied_embeddings():
    model = MLXTelosTransformer(vocab_size=64, d_model=64, n_layers=1, n_heads=1,
                                tied_embeddings=False, precision="float32")
    with pytest.raises(ValueError, match="tied embeddings"):
        model.output_cross_entropy(mx.ones((1, 3), dtype=mx.int32), mx.ones((1, 3), dtype=mx.int32))
