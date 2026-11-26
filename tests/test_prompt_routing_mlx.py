"""Tiny checks for prompt boundaries and the trainable MLX router."""

import pytest

from telos.training.core import metal_usable

pytestmark = pytest.mark.skipif(not metal_usable(), reason="MLX Metal required")


@pytest.mark.parametrize("mode", ["prompt", "token"])
def test_routed_training_is_causal_after_the_prompt(mode):
    import mlx.core as mx
    from mlx import nn

    from telos.models.prompt_routing_mlx import RoutedTransformer

    mx.random.seed(7)
    model = RoutedTransformer(routing=mode, vocab_size=40)
    a = mx.array([[37, 2, 5, 6, 38, 21, 23, 25]])
    b = mx.array([[37, 2, 5, 6, 38, 30, 31, 32]])
    logits, _ = model(a, 5)
    changed, _ = model(b, 5)
    assert mx.array_equal(logits[:, 4], changed[:, 4]).item()
    prefill, state = model.step(a[:, :5])
    assert mx.allclose(prefill, logits[0, 4], atol=1e-5, rtol=1e-5).item()
    _, next_state = model.step(a[:, 5:6], state)
    if mode == "prompt":
        assert all(first is not None and first == second for (_, first), (_, second) in zip(state, next_state, strict=True))
    else:
        assert all(fixed is None for _, fixed in next_state)

    def loss(net):
        predictions, balance = net(a, 5)
        return nn.losses.cross_entropy(predictions[:, 4:-1], a[:, 5:], reduction="mean") + 0.02*balance

    _, gradients = nn.value_and_grad(model, loss)(model)
    assert all(mx.any(layer["mlp"]["router"]["weight"] != 0).item() for layer in gradients["layers"])
