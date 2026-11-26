"""Experimental causal transformer with top-one expert routing."""

from typing import NamedTuple

import mlx.core as mx
from mlx import nn

from .mlx_components import MLXBlock, MLXEmbedding, MLXRMSNorm

ROUTING_MODES = ("prompt", "token", "hybrid4", "hybrid8")
_REFRESH = {"hybrid4": 4, "hybrid8": 8}


class DecodeState(NamedTuple):
    layers: tuple
    decoded: int


def _expert(x, gate_up, down):
    gate, up = mx.split(x @ gate_up.T, 2, axis=-1)
    return (nn.silu(gate) * up) @ down.T


class RoutedFFN(nn.Module):
    def __init__(self, width, inner, experts, shared):
        super().__init__()
        self.shared = shared
        self.router = nn.Linear(width, experts, bias=False)
        self.count = experts
        scale = width**-0.5
        self.gate_up = mx.random.uniform(-scale, scale, (experts, 2 * inner, width))
        self.down = mx.random.uniform(-scale, scale, (experts, width, inner))

    def train_routes(self, x, features, valid=None, slots=None):
        probabilities = mx.softmax(self.router(features).astype(mx.float32), axis=-1)
        ids = mx.stop_gradient(mx.argmax(probabilities, axis=-1, keepdims=True))
        chosen = mx.take_along_axis(probabilities, ids, axis=-1)
        assignment = (ids == mx.arange(self.count)).astype(mx.float32)
        if valid is not None:
            weight = valid[..., None].astype(mx.float32)
            count = weight.sum()
            average = (probabilities * weight).sum(axis=(0, 1)) / count
            frequency = (assignment * weight).sum(axis=(0, 1)) / count
        else:
            average = probabilities.reshape(-1, self.count).mean(axis=0)
            frequency = assignment.reshape(-1, self.count).mean(axis=0)
        balance = self.count * (average * frequency).sum()
        if slots is not None:
            ids = mx.take_along_axis(ids, slots, axis=1)
            chosen = mx.take_along_axis(chosen, slots, axis=1)
        ids = mx.broadcast_to(ids, (*x.shape[:-1], 1)).squeeze(-1)
        gate_up = mx.gather_mm(x[..., None, :], self.gate_up.swapaxes(-1, -2), rhs_indices=ids).squeeze(-2)
        gate, up = mx.split(gate_up, 2, axis=-1)
        result = mx.gather_mm((nn.silu(gate) * up)[..., None, :], self.down.swapaxes(-1, -2),
                              rhs_indices=ids).squeeze(-2)
        # The router gets the selected probability's gradient with a unit forward weight.
        gate = 1 + (chosen - mx.stop_gradient(chosen))
        return self.shared(x) + gate.astype(x.dtype) * result, balance


class RoutedTransformer(nn.Module):
    def __init__(self, *, routing, vocab_size, width=64, inner=128, layers=2, experts=4):
        if routing not in ROUTING_MODES:
            raise ValueError(f"Routing must be one of {', '.join(ROUTING_MODES)}")
        super().__init__()
        self.routing = routing
        self.refresh_tokens = _REFRESH.get(routing)
        self.emb = MLXEmbedding(vocab_size, width)
        self.layers = [MLXBlock(width, 4, 4, is_causal=True) for _ in range(layers)]
        self.norm = MLXRMSNorm(width)
        for layer in self.layers:
            layer.mlp = RoutedFFN(width, inner, experts, layer.mlp)

    def __call__(self, ids, prompt_length, sequence_length=None):
        if isinstance(prompt_length, int) and not 1 <= prompt_length <= ids.shape[1]:
            raise ValueError("Prompt length must be within the input sequence")
        x = self.emb(ids)
        positions = mx.arange(ids.shape[1])[None, :]
        valid = None if sequence_length is None else positions < sequence_length[:, None]
        prompt = anchors = slots = None
        if self.routing != "token":
            if isinstance(prompt_length, int):
                prompt = x[:, :prompt_length].mean(axis=1, keepdims=True)
            else:
                mask = (positions < prompt_length[:, None])[..., None]
                prompt = mx.where(mask, x, 0).sum(axis=1, keepdims=True) / prompt_length[:, None, None]
            if self.refresh_tokens:
                prompts = mx.full((ids.shape[0],), prompt_length) if isinstance(prompt_length, int) else prompt_length
                first = prompts[:, None] - 1
                # The prompt predicts response token 1. Consuming response token N
                # selects the experts that start predicting at response token N+1.
                anchors = first + self.refresh_tokens * mx.arange(1, (ids.shape[1]-1)//self.refresh_tokens+1)
                stops = ids.shape[1] if sequence_length is None else sequence_length[:, None]
                valid = mx.concatenate((mx.ones_like(first, dtype=mx.bool_), anchors < stops), axis=1)
                anchors = mx.minimum(anchors, ids.shape[1]-1)[..., None]
                slots = (mx.maximum(positions-first, 0)//self.refresh_tokens)[..., None]
            else:
                valid = None
        balance = mx.array(0.0)
        for layer in self.layers:
            x, _ = layer.attention_residual(x)
            h = layer.norm2(x)
            features = h if prompt is None else layer.norm2(prompt)
            if anchors is not None:
                features = mx.concatenate((features, mx.take_along_axis(h, anchors, axis=1)), axis=1)
            output, auxiliary = layer.mlp.train_routes(h, features, valid, slots)
            x = x + output
            balance = balance + auxiliary
        return self.emb.as_linear(self.norm(x)), balance / len(self.layers)

    def step(self, ids, state=None, read_expert=None):
        """Decode one sequence, with the initial input containing only the prompt."""
        if ids.ndim != 2 or ids.shape[0] != 1 or not ids.shape[1] or (state is not None and ids.shape[1] != 1):
            raise ValueError("Routed decode expects one prompt or one cached token")
        x = self.emb(ids)
        pooled = x.mean(axis=1, keepdims=True) if state is None and self.routing != "token" else None
        decoded = 0 if state is None else state.decoded + 1
        refresh = self.refresh_tokens is not None and decoded % self.refresh_tokens == 0
        updated = []
        for index, layer in enumerate(self.layers):
            previous, fixed = (None, None) if state is None else state.layers[index]
            x, kv = layer.attention_residual(x, kv_cache=previous)
            h = layer.norm2(x)
            mlp = layer.mlp
            if self.routing != "token":
                if pooled is not None or refresh:
                    features = h if pooled is None else layer.norm2(pooled)
                    fixed = int(mx.argmax(mlp.router(features), axis=-1).item())
                selected = [fixed] * h.shape[1]
            else:
                selected = mx.argmax(mlp.router(h), axis=-1)[0].tolist()
            routed = mx.zeros_like(h)
            for expert in sorted(set(selected)):
                positions = mx.array([i for i, value in enumerate(selected) if expert == value])
                gate, down = ((mlp.gate_up[expert], mlp.down[expert]) if read_expert is None
                              else read_expert(index, expert))
                values = _expert(h[:, positions], gate, down)
                routed = routed.at[:, positions].add(values)
                mx.eval(routed)
                del gate, down, values
            x = x + (mlp.shared(h) + routed)
            updated.append((kv, fixed))
        logits = self.emb.as_linear(self.norm(x[:, -1:]))[0, 0]
        mx.eval(logits, updated)
        return logits, DecodeState(tuple(updated), decoded)
