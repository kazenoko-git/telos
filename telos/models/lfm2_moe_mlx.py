"""Native MLX inference for the LFM2.5 text MoE hybrid backbone."""

import math
from typing import NamedTuple

import mlx.core as mx
from mlx import nn

from .qwen3_mlx import _LayerCache, _positive, _rope_parameters, _Weights
from .qwen3_moe_mlx import RoutedMLP
from .window_routing import WindowPinController, compute_route, sequential_ssd_loss_and_grad


def layer_types(config):
    kinds = config.get("layer_types")
    if (not isinstance(kinds, (list, tuple))
            or len(kinds) != _positive(config, "num_hidden_layers")
            or any(kind not in ("conv", "full_attention") for kind in kinds)):
        raise ValueError("LFM2-MoE requires one conv or full_attention type per layer")
    return kinds


class LFM2RoutedMLP(RoutedMLP):
    def __init__(self, weights, index, config, width, ranges, cache):
        super().__init__(weights, index, config, width, ranges, cache, module="feed_forward")
        experts = _positive(config, "num_experts")
        key = f"model.layers.{index}.feed_forward.expert_bias"
        self.bias = weights.take(key, (experts,)) if config.get("use_expert_bias", False) else None
        self.scaling = float(config.get("routed_scaling_factor", 1.0))
        if not math.isfinite(self.scaling) or self.scaling <= 0:
            raise ValueError("LFM2-MoE routed_scaling_factor must be positive and finite")

    def _route(self, x):
        return compute_route(
            x,
            self.router,
            layer=self.layer,
            top_k=self.top_k,
            route_scale=self.scaling,
            normalize=self.normalize,
            base_bias=self.bias,
            adapter=self.router_adapter,
            controller=self._window_controller,
            activation="sigmoid",
            eps=1e-6,
            output_dtype=x.dtype,
        )


class _ConvState(NamedTuple):
    history: mx.array
    length: int


class _ShortConv(nn.Module):
    def __init__(self, weights, prefix, width, kernel, bias):
        super().__init__()
        self.width = width
        self.history = kernel - 1
        self.input = weights.matrix(prefix + ".in_proj", width * 3, width, bias=bias)
        self.output = weights.matrix(prefix + ".out_proj", width, width, bias=bias)
        self.kernel = weights.take(prefix + ".conv.weight", (width, kernel, 1))
        self.bias = weights.take(prefix + ".conv.bias", (width,)) if bias else None

    def __call__(self, x, previous, _limit):
        key, gate, value = mx.split(self.input(x), 3, axis=-1)
        gated = key * value
        if self.history:
            history = (mx.zeros((x.shape[0], self.history, self.width), dtype=gated.dtype)
                       if previous is None else previous.history)
            gated = mx.concatenate((history, gated), axis=1)
            # Gather a compact tail so the short state does not retain the prefill buffer.
            positions = mx.arange(gated.shape[1] - self.history, gated.shape[1])
            state = mx.take(gated, positions, axis=1)
        else:
            state = gated[:, :0]
        convolved = mx.conv1d(gated, self.kernel, groups=self.width)
        if self.bias is not None:
            convolved = convolved + self.bias
        length = (0 if previous is None else previous.length) + x.shape[1]
        return self.output(gate * convolved), _ConvState(state, length)


class _Attention(nn.Module):
    def __init__(self, weights, prefix, width, heads, kv_heads, eps, rotary):
        super().__init__()
        self.head_dim = width // heads
        self.scale = self.head_dim ** -0.5
        self.eps = eps
        self._rotary = rotary
        kv_width = kv_heads * self.head_dim
        self.query = weights.matrix(prefix + ".q_proj", width, width)
        self.key = weights.matrix(prefix + ".k_proj", kv_width, width)
        self.value = weights.matrix(prefix + ".v_proj", kv_width, width)
        self.output = weights.matrix(prefix + ".out_proj", width, width)
        self.query_norm = weights.take(prefix + ".q_layernorm.weight", (self.head_dim,))
        self.key_norm = weights.take(prefix + ".k_layernorm.weight", (self.head_dim,))

    def __call__(self, x, previous, limit):
        batch, length, width = x.shape
        shape = (batch, length, -1, self.head_dim)
        query = mx.fast.rms_norm(self.query(x).reshape(shape), self.query_norm, self.eps)
        key = mx.fast.rms_norm(self.key(x).reshape(shape), self.key_norm, self.eps)
        value = self.value(x).reshape(shape).transpose(0, 2, 1, 3)
        offset = 0 if previous is None else previous.length
        base, freqs, amplitude = self._rotary
        options = {"traditional": False, "base": base, "freqs": freqs, "scale": 1.0, "offset": offset}
        query = mx.fast.rope(query.transpose(0, 2, 1, 3) * amplitude, self.head_dim, **options)
        key = mx.fast.rope(key.transpose(0, 2, 1, 3) * amplitude, self.head_dim, **options)
        if previous is None:
            present = _LayerCache(key, value, length)
        else:
            present = previous.append(key, value, limit)
            key = present.keys[:, :, :present.length]
            value = present.values[:, :, :present.length]
        output = mx.fast.scaled_dot_product_attention(
            query, key, value, scale=self.scale, mask="causal" if length > 1 else None,
        )
        return self.output(output.transpose(0, 2, 1, 3).reshape(batch, length, width)), present


class _DenseMLP(nn.Module):
    def __init__(self, weights, prefix, width, inner):
        super().__init__()
        self.gate = weights.matrix(prefix + ".gate_proj", inner, width)
        self.up = weights.matrix(prefix + ".up_proj", inner, width)
        self.down = weights.matrix(prefix + ".down_proj", width, inner)

    def __call__(self, x):
        return self.down(nn.silu(self.gate(x)) * self.up(x))


class _Layer(nn.Module):
    def __init__(self, weights, index, width, operator, mlp, eps):
        super().__init__()
        prefix = f"model.layers.{index}"
        self.operator = operator
        self.mlp = mlp
        self.eps = eps
        self.operator_norm = weights.take(prefix + ".operator_norm.weight", (width,))
        self.ffn_norm = weights.take(prefix + ".ffn_norm.weight", (width,))

    def __call__(self, x, previous, limit):
        mixed, state = self.operator(mx.fast.rms_norm(x, self.operator_norm, self.eps), previous, limit)
        x = x + mixed
        return x + self.mlp(mx.fast.rms_norm(x, self.ffn_norm, self.eps)), state


class LFM2MoEMLX(nn.Module):
    is_causal = True
    paradigm = "ar"
    # A short convolution tail has no state for arbitrary earlier prefixes.
    supports_prefix_cache = False

    def __init__(self, config, arrays, *, expert_factory):
        super().__init__()
        width = _positive(config, "hidden_size")
        heads = _positive(config, "num_attention_heads")
        kv_heads = _positive(config, "num_key_value_heads", heads)
        depth = _positive(config, "num_hidden_layers")
        dense = config.get("num_dense_layers", 2)
        if width % heads or width // heads % 2 or heads % kv_heads:
            raise ValueError("LFM2-MoE requires evenly grouped heads and an even head dimension")
        if type(dense) is not int or not 0 <= dense < depth:
            raise ValueError("Invalid number of dense LFM2-MoE layers")
        if config.get("hidden_act", "silu") != "silu":
            raise ValueError("Native LFM2-MoE requires SiLU")
        self.vocab_size = _positive(config, "vocab_size")
        self.max_seq_len = _positive(config, "max_position_embeddings", 128000)
        self.eps = float(config.get("norm_eps", 1e-5))
        kernel = _positive(config, "conv_L_cache", 3)
        rotary = _rope_parameters({"rope_theta": 1_000_000.0, **config}, width // heads, self.max_seq_len)
        weights = _Weights(arrays, config.get("quantization") or {})
        self.embedding = weights.matrix("model.embed_tokens", self.vocab_size, width)
        self.layers = []
        self._window_controller = WindowPinController()
        for index, kind in enumerate(layer_types(config)):
            prefix = f"model.layers.{index}"
            if kind == "conv":
                operator = _ShortConv(weights, prefix + ".conv", width, kernel, config.get("conv_bias", False))
            else:
                operator = _Attention(weights, prefix + ".self_attn", width, heads, kv_heads, self.eps, rotary)
            if index < dense:
                mlp = _DenseMLP(weights, prefix + ".feed_forward", width, _positive(config, "intermediate_size"))
            else:
                mlp = expert_factory(weights, index, config, width)
                mlp._window_controller = self._window_controller
            self.layers.append(_Layer(weights, index, width, operator, mlp, self.eps))
        self.norm = weights.take("model.embedding_norm.weight", (width,))
        self.head = (None if config.get("tie_word_embeddings", True)
                     else weights.matrix("lm_head", self.vocab_size, width))
        if weights.tensors:
            raise ValueError(f"Unexpected LFM2-MoE checkpoint tensor: {next(iter(weights.tensors))}")

    def set_window_size(self, window_size: int | None):
        self._window_controller.set_window_size(window_size)

    def hidden_states_with_cache(self, ids, kv_cache=None, *, is_causal=True):
        if not is_causal or ids.shape[1] < 1:
            raise ValueError("LFM2-MoE requires causal attention and at least one token per row")
        if self._window_controller.mode is None:
            if kv_cache is None:
                self._window_controller.begin_prefill()
            elif ids.shape[1] == 1:
                self._window_controller.begin_decode_step()
        offset = 0 if kv_cache is None else kv_cache[0].length
        if offset + ids.shape[1] > self.max_seq_len:
            raise ValueError(f"LFM2-MoE context exceeds the model limit of {self.max_seq_len} tokens")
        x = self.embedding.lookup(ids)
        updated = []
        for index, layer in enumerate(self.layers):
            x, state = layer(x, None if kv_cache is None else kv_cache[index], self.max_seq_len)
            updated.append(state)
        return mx.fast.rms_norm(x, self.norm, self.eps), updated

    def prefill_last_hidden(self, ids, kv_cache=None):
        hidden, cache = self.hidden_states_with_cache(ids, kv_cache)
        return hidden[:, -1:], cache

    def hidden_states(self, ids, *, is_causal=True):
        return self.hidden_states_with_cache(ids, is_causal=is_causal)[0]

    def logits_from_hidden(self, hidden):
        return (self.embedding if self.head is None else self.head)(hidden)

    def ssd_loss_and_grad(
        self,
        ids,
        loss_from_logits_fn,
        *,
        teacher_routes=None,
        prompt_lengths=None,
        response_mask=None,
        window_size=None,
        cov_weight=0.0,
    ):
        """Run NOGRAD forward + per-layer reverse-mode VJP to keep SSD expert memory bounded."""
        return sequential_ssd_loss_and_grad(
            self,
            ids,
            lambda x: self.embedding.lookup(x),
            lambda mod, x: mod(x, None, self.max_seq_len)[0],
            lambda x: loss_from_logits_fn(self.logits_from_hidden(mx.fast.rms_norm(x, self.norm, self.eps))[:, :-1]),
            controller=self._window_controller,
            teacher_routes=teacher_routes,
            prompt_lengths=prompt_lengths,
            response_mask=response_mask,
            window_size=window_size,
            cov_weight=cov_weight,
        )
