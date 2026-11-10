"""Parity checks for skipping the unused final AR vocabulary projection."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
mx_nn = pytest.importorskip("mlx.nn")
if not mx.metal.is_available():
    pytest.skip("Metal is required", allow_module_level=True)

from mlx.utils import tree_flatten

from telos.diffusion.ar import ar_loss_fn_mlx, cross_entropy
from telos.models.mlx_transformer import MLXTelosTransformer


def _reference_ar_loss(model, tokens, special):
    batch, sequence = tokens.shape
    logits = model(tokens)
    vocab = logits.shape[-1]
    targets = tokens[:, 1:]
    losses = cross_entropy(
        logits[:, :-1, :].reshape(-1, vocab),
        targets.reshape(-1),
        reduction="none",
    ).reshape(batch, sequence - 1)
    content = ~special[targets]
    losses = losses * content
    per_example = mx.sum(losses, axis=1) / mx.clip(
        mx.sum(content, axis=1), 1.0, float(sequence - 1)
    )
    return mx.mean(per_example)


@pytest.mark.parametrize("tied_embeddings", [True, False])
def test_next_token_projection_matches_full_logits_loss_and_gradients(tied_embeddings):
    mx.random.seed(23)
    kwargs = dict(
        vocab_size=97, d_model=64, n_layers=2, n_heads=1,
        tied_embeddings=tied_embeddings, precision="float32",
    )
    reference = MLXTelosTransformer(**kwargs, is_causal=True)
    optimized = MLXTelosTransformer(**kwargs, is_causal=True)
    optimized.load_weights(list(tree_flatten(reference.parameters())))
    tokens = mx.random.randint(0, kwargs["vocab_size"], (3, 13))
    special = mx.array([True, True, True, True] + [False] * (kwargs["vocab_size"] - 4))

    reference_vg = mx_nn.value_and_grad(
        reference, lambda model, ids: _reference_ar_loss(model, ids, special)
    )
    optimized_vg = mx_nn.value_and_grad(
        optimized, lambda model, ids: ar_loss_fn_mlx(model, ids, kwargs["vocab_size"], special)[0]
    )
    expected, expected_grads = reference_vg(reference, tokens)
    actual, actual_grads = optimized_vg(optimized, tokens)
    mx.eval(expected, expected_grads, actual, actual_grads)

    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=1e-6, atol=1e-6)
    for (expected_name, expected_grad), (actual_name, actual_grad) in zip(
        tree_flatten(expected_grads), tree_flatten(actual_grads)
    ):
        assert actual_name == expected_name
        np.testing.assert_allclose(
            np.array(actual_grad), np.array(expected_grad), rtol=2e-5, atol=2e-6
        )
