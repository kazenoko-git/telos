"""Tiny checks for prompt boundaries and the trainable MLX router."""

import pytest

from telos.training.core import metal_usable

pytestmark = pytest.mark.skipif(not metal_usable(), reason="MLX Metal required")


@pytest.mark.parametrize("mode", ["prompt", "token", "hybrid4", "hybrid8"])
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
    if mode != "token":
        assert all(first is not None and first == second
                   for (_, first), (_, second) in zip(state.layers, next_state.layers, strict=True))
    else:
        assert all(fixed is None for _, fixed in next_state.layers)

    def loss(net):
        predictions, balance = net(a, 5)
        return nn.losses.cross_entropy(predictions[:, 4:-1], a[:, 5:], reduction="mean") + 0.02*balance

    _, gradients = nn.value_and_grad(model, loss)(model)
    assert all(mx.any(layer["mlp"]["router"]["weight"] != 0).item() for layer in gradients["layers"])


@pytest.mark.parametrize("mode", ["prompt", "token", "hybrid4", "hybrid8"])
def test_variable_prompt_lengths_exclude_completion_padding_and_future_tokens(mode):
    import mlx.core as mx

    from telos.models.prompt_routing_mlx import RoutedTransformer
    from telos.training.routing_experiment import _completion

    mx.random.seed(9)
    model = RoutedTransformer(routing=mode, vocab_size=40)
    ids = mx.array([[37, 2, 38, 21, 23, 0, 0], [37, 2, 9, 8, 38, 25, 0]])
    changed_padding = mx.array([[37, 2, 38, 21, 23, 10, 30], [37, 2, 9, 8, 38, 25, 31]])
    prompts, ends = mx.array([3, 5]), mx.array([5, 6])
    logits, _, mask, balance = _completion(model, ids, prompts, ends)
    changed, _, _, changed_balance = _completion(model, changed_padding, prompts, ends)
    assert mask.tolist() == [[False, False, True, True, False, False],
                             [False, False, False, False, True, False]]
    assert mx.array_equal(mx.where(mask[..., None], logits, 0), mx.where(mask[..., None], changed, 0)).item()
    assert mx.array_equal(balance, changed_balance).item()

    for row, prompt in enumerate(prompts.tolist()):
        prefill, _ = model.step(ids[row:row+1, :prompt])
        assert mx.allclose(prefill, logits[row, prompt-1], atol=1e-5, rtol=1e-5).item()


@pytest.mark.parametrize(("mode", "interval"), [("hybrid4", 4), ("hybrid8", 8)])
def test_hybrid_refresh_boundaries_match_teacher_forcing(mode, interval):
    import mlx.core as mx
    from mlx import nn

    from telos.models.prompt_routing_mlx import RoutedTransformer

    class RouterSpy(nn.Module):
        def __init__(self, source):
            super().__init__()
            self.source = source
            self.calls = 0

        def __call__(self, features):
            self.calls += 1
            return self.source(features)

    mx.random.seed(14)
    model = RoutedTransformer(routing=mode, vocab_size=40)
    ids = mx.array([[37, 2, 5, 6, 38, *range(21, 39)]], dtype=mx.int32)
    prefix = 5
    logits, _ = model(ids, mx.array([prefix]), mx.array([ids.shape[1]]))
    spies = []
    for layer in model.layers:
        spy = RouterSpy(layer.mlp.router)
        layer.mlp.router = spy
        spies.append(spy)

    actual, state = model.step(ids[:, :prefix])
    assert all(spy.calls == 1 for spy in spies)
    assert mx.allclose(actual, logits[0, prefix-1], atol=3e-5, rtol=1e-5).item()
    for i in range(1, ids.shape[1]-prefix+1):
        pos = prefix+i-1
        actual, state = model.step(ids[:, pos:pos+1], state)
        assert state.decoded == i
        assert all(spy.calls == 1+i//interval for spy in spies)
        assert mx.allclose(actual, logits[0, pos], atol=3e-5, rtol=1e-5).item()
