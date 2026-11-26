"""Experimental causal transformer with prompt-fixed or token-level top-one routing."""

import mlx.core as mx
from mlx import nn

from .mlx_components import MLXBlock, MLXEmbedding, MLXRMSNorm


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

    def train_routes(self, x, prompt):
        probabilities = mx.softmax(self.router(x if prompt is None else prompt).astype(mx.float32), axis=-1)
        ids = mx.stop_gradient(mx.argmax(probabilities, axis=-1, keepdims=True))
        chosen = mx.take_along_axis(probabilities, ids, axis=-1)
        assignment = (ids == mx.arange(self.count)).astype(mx.float32)
        balance = self.count * (probabilities.reshape(-1, self.count).mean(axis=0)
                                * assignment.reshape(-1, self.count).mean(axis=0)).sum()
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
        if routing not in ("prompt", "token"):
            raise ValueError("Routing must be prompt or token")
        super().__init__()
        self.routing = routing
        self.emb = MLXEmbedding(vocab_size, width)
        self.layers = [MLXBlock(width, 4, 4, is_causal=True) for _ in range(layers)]
        self.norm = MLXRMSNorm(width)
        for layer in self.layers:
            layer.mlp = RoutedFFN(width, inner, experts, layer.mlp)

    def __call__(self, ids, prompt_length):
        if not 1 <= prompt_length <= ids.shape[1]:
            raise ValueError("Prompt length must be within the input sequence")
        x = self.emb(ids)
        prompt = x[:, :prompt_length].mean(axis=1, keepdims=True) if self.routing == "prompt" else None
        balance = mx.array(0.0)
        for layer in self.layers:
            x, _ = layer.attention_residual(x)
            pooled = None if prompt is None else layer.norm2(prompt)
            output, auxiliary = layer.mlp.train_routes(layer.norm2(x), pooled)
            x = x + output
            balance = balance + auxiliary
        return self.emb.as_linear(self.norm(x)), balance / len(self.layers)

    def step(self, ids, state=None, read_expert=None):
        """Decode one sequence, with the initial input containing only the prompt."""
        if ids.ndim != 2 or ids.shape[0] != 1 or not ids.shape[1] or (state is not None and ids.shape[1] != 1):
            raise ValueError("Routed decode expects one prompt or one cached token")
        x = self.emb(ids)
        pooled = x.mean(axis=1, keepdims=True) if state is None and self.routing == "prompt" else None
        updated = []
        for index, layer in enumerate(self.layers):
            previous, fixed = (None, None) if state is None else state[index]
            x, kv = layer.attention_residual(x, kv_cache=previous)
            h = layer.norm2(x)
            mlp = layer.mlp
            if self.routing == "prompt":
                if fixed is None:
                    fixed = int(mx.argmax(mlp.router(layer.norm2(pooled)), axis=-1).item())
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
        return logits, updated
