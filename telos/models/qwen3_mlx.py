"""Native MLX inference for full-attention Qwen3 causal checkpoints."""

import math
from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn

from .mlx_quantized import AffineWeights
from .window_routing import WindowPinController


def _positive(config, name, default=None):
    value = config.get(name, default)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"Qwen3 config requires a positive {name}")
    return value


def _rope_parameters(config, width, max_seq_len):
    """Return the MLX RoPE base or YaRN wavelengths and amplitude."""
    base = float(config.get("rope_theta", 10000.0))
    scaling = config.get("rope_parameters") or config.get("rope_scaling") or {}
    base = float(scaling.get("rope_theta", base))
    if not math.isfinite(base) or base <= 1:
        raise ValueError("Qwen3 rope_theta must be finite and greater than one")
    kind = scaling.get("rope_type", scaling.get("type", "default"))
    if kind == "default":
        return base, None, 1.0
    if kind == "linear":
        factor = float(scaling["factor"])
        if not math.isfinite(factor) or factor < 1:
            raise ValueError("Qwen3 RoPE factor must be finite and at least one")
        waves = base ** (mx.arange(0, width, 2, dtype=mx.float32) / width)
        waves = waves * factor
        mx.eval(waves)
        return None, waves, 1.0
    if kind != "yarn":
        raise ValueError(f"Unsupported Qwen3 RoPE scaling: {kind}")

    factor = float(scaling["factor"])
    original = _positive(scaling, "original_max_position_embeddings", max_seq_len)
    fast = float(scaling.get("beta_fast", 32))
    slow = float(scaling.get("beta_slow", 1))
    if not math.isfinite(factor) or factor < 1 or not 0 < slow < fast < math.inf:
        raise ValueError("Invalid Qwen3 YaRN factor or correction range")
    correction = width / (2 * math.log(base))
    low = correction * math.log(original / (2 * math.pi * fast))
    high = correction * math.log(original / (2 * math.pi * slow))
    if scaling.get("truncate", True):
        low, high = math.floor(low), math.ceil(high)
    low, high = max(low, 0), min(high, width - 1)
    blend = mx.clip((mx.arange(width // 2, dtype=mx.float32) - low) / (high - low or 0.001), 0, 1)
    waves = base ** (mx.arange(0, width, 2, dtype=mx.float32) / width)
    waves = waves / (1 - blend + blend / factor)
    log_factor = 0.1 * math.log(factor)
    amplitude = scaling.get("attention_factor")
    if amplitude is None:
        mscale, all_dims = scaling.get("mscale"), scaling.get("mscale_all_dim")
        amplitude = ((1 + mscale * log_factor) / (1 + all_dims * log_factor)
                     if mscale and all_dims else 1 + log_factor)
    # The frequency table is shared across requests and can cross threads.
    mx.eval(waves)
    return None, waves, float(amplitude)


class _Weights:
    def __init__(self, tensors, quantization):
        self.tensors = tensors
        self.quantization = quantization

    def take(self, key, shape=None):
        try:
            value = self.tensors.pop(key)
        except KeyError as exc:
            raise ValueError(f"Qwen3 checkpoint is missing {key}") from exc
        if shape is not None and (value.shape != shape or not mx.issubdtype(value.dtype, mx.floating)):
            raise ValueError(f"Invalid Qwen3 tensor {key}: expected a floating-point {shape} array")
        return value

    def matrix(self, key, rows, width, *, bias=False):
        offset = self.take(key + ".bias", (rows,)) if bias else None
        if key + ".scales" not in self.tensors:
            return _DenseWeights(self.take(key + ".weight", (rows, width)), offset)

        quant = self.quantization.get(key, self.quantization)
        if not isinstance(quant, dict) or quant.get("mode", "affine") != "affine":
            raise ValueError(f"Unsupported Qwen3 quantization for {key}")
        packed = AffineWeights(
            self.take(key + ".weight"), self.take(key + ".scales"), self.take(key + ".biases"),
            shape=(rows, width), group_size=_positive(quant, "group_size"), bits=_positive(quant, "bits"),
        )
        return _QuantizedProjection(packed, offset)


class _DenseWeights(nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        self.weight = weight
        self.bias = bias

    def __call__(self, x):
        return mx.matmul(x, self.weight.T) if self.bias is None else mx.addmm(self.bias, x, self.weight.T)

    def lookup(self, indices):
        return self.weight[indices]


class _QuantizedProjection(nn.Module):
    def __init__(self, packed, bias):
        super().__init__()
        self.packed = packed
        self.bias = bias

    def __call__(self, x):
        output = self.packed.as_linear(x)
        return output if self.bias is None else output + self.bias

    def lookup(self, indices):
        return self.packed.lookup(indices)


class _LayerCache(NamedTuple):
    keys: mx.array
    values: mx.array
    length: int

    def append(self, key, value, limit):
        length = self.length + key.shape[2]
        capacity = self.keys.shape[2]
        keys, values = self.keys, self.values
        if length > capacity:
            # Keep prefill compact. Reserve storage in chunks only once decode
            # needs room, and cap the reservation at the model's context.
            capacity_needed = min(limit, ((length + 255) // 256) * 256)
            padding = [(0, 0), (0, 0), (0, capacity_needed - capacity), (0, 0)]
            keys, values = mx.pad(keys, padding), mx.pad(values, padding)
        start = mx.array(self.length, dtype=mx.int32)
        return _LayerCache(
            mx.slice_update(keys, key, start, [2]),
            mx.slice_update(values, value, start, [2]),
            length,
        )

    def truncate(self, length):
        return _LayerCache(self.keys[:, :, :length], self.values[:, :, :length], length)


class _Block(nn.Module):
    def __init__(self, weights, index, config, width, heads, kv_heads, head_dim, mlp_width, *, expert_factory=None):
        super().__init__()
        prefix = f"model.layers.{index}"
        attn = prefix + ".self_attn"
        mlp = prefix + ".mlp"
        bias = config.get("attention_bias", False)
        self.query = weights.matrix(attn + ".q_proj", heads * head_dim, width, bias=bias)
        self.key = weights.matrix(attn + ".k_proj", kv_heads * head_dim, width, bias=bias)
        self.value = weights.matrix(attn + ".v_proj", kv_heads * head_dim, width, bias=bias)
        self.output = weights.matrix(attn + ".o_proj", width, heads * head_dim, bias=bias)
        self.query_norm = weights.take(attn + ".q_norm.weight", (head_dim,))
        self.key_norm = weights.take(attn + ".k_norm.weight", (head_dim,))
        self.input_norm = weights.take(prefix + ".input_layernorm.weight", (width,))
        self.post_norm = weights.take(prefix + ".post_attention_layernorm.weight", (width,))
        uses_experts = (expert_factory is not None and index not in config.get("mlp_only_layers", ())
                        and (index + 1) % _positive(config, "decoder_sparse_step", 1) == 0)
        if uses_experts:
            self.moe = expert_factory(weights, index, config, width)
        else:
            self.moe = None
            self.gate = weights.matrix(mlp + ".gate_proj", mlp_width, width)
            self.up = weights.matrix(mlp + ".up_proj", mlp_width, width)
            self.down = weights.matrix(mlp + ".down_proj", width, mlp_width)
        self.eps = float(config.get("rms_norm_eps", 1e-6))
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

    def __call__(self, x, previous, rotary, limit, *, last_only=False):
        batch, length, _ = x.shape
        hidden = mx.fast.rms_norm(x, self.input_norm, self.eps)
        shape = (batch, length, -1, self.head_dim)
        query_length = 1 if last_only else length
        query_input = hidden[:, -1:] if last_only else hidden
        query_shape = (batch, query_length, -1, self.head_dim)
        q = mx.fast.rms_norm(self.query(query_input).reshape(query_shape), self.query_norm, self.eps).transpose(0, 2, 1, 3)
        k = mx.fast.rms_norm(self.key(hidden).reshape(shape), self.key_norm, self.eps).transpose(0, 2, 1, 3)
        v = self.value(hidden).reshape(shape).transpose(0, 2, 1, 3)

        base, freqs, amplitude = rotary
        offset = previous.length if previous is not None else 0
        q = mx.fast.rope(q * amplitude, self.head_dim, traditional=False, base=base, freqs=freqs, scale=1.0,
                         offset=offset + length - query_length)
        k = mx.fast.rope(k * amplitude, self.head_dim, traditional=False, base=base, freqs=freqs, scale=1.0, offset=offset)
        if previous is None:
            present = _LayerCache(k, v, length)
        else:
            present = previous.append(k, v, limit)
            k, v = present.keys[:, :, :present.length], present.values[:, :, :present.length]
        # A lone final query can attend to every valid cache entry.
        mask = "causal" if query_length > 1 else None
        attention = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        attention = attention.transpose(0, 2, 1, 3).reshape(batch, query_length, -1)
        if last_only:
            x = x[:, -1:]
        x = x + self.output(attention)
        hidden = mx.fast.rms_norm(x, self.post_norm, self.eps)
        if self.moe is not None:
            return x + self.moe(hidden), present
        return x + self.down(nn.silu(self.gate(hidden)) * self.up(hidden)), present


class Qwen3MLX(nn.Module):
    """Load dense or affine-quantized Qwen3 without initializing dense weights."""

    is_causal = True
    paradigm = "ar"

    def __init__(self, config, arrays, *, expert_factory=None):
        super().__init__()
        width = _positive(config, "hidden_size")
        heads = _positive(config, "num_attention_heads")
        kv_heads = _positive(config, "num_key_value_heads", heads)
        head_dim = _positive(config, "head_dim", width // heads)
        depth = _positive(config, "num_hidden_layers")
        mlp_width = _positive(config, "intermediate_size")
        self.vocab_size = _positive(config, "vocab_size")
        self.max_seq_len = _positive(config, "max_position_embeddings", 32768)
        self.eps = float(config.get("rms_norm_eps", 1e-6))
        if heads % kv_heads or head_dim % 2:
            raise ValueError("Qwen3 requires evenly grouped heads and an even head dimension")
        if (config.get("use_sliding_window") or config.get("partial_rotary_factor", 1.0) != 1.0
                or config.get("hidden_act", "silu") != "silu"
                or any(kind != "full_attention" for kind in (config.get("layer_types") or ()))):
            raise ValueError("Native Qwen3 requires full attention, full rotary dimensions and SiLU")
        self._rotary = _rope_parameters(config, head_dim, self.max_seq_len)
        self._window_controller = WindowPinController()
        weights = _Weights(arrays, config.get("quantization") or {})
        self.embedding = weights.matrix("model.embed_tokens", self.vocab_size, width)
        self.layers = [_Block(weights, i, config, width, heads, kv_heads, head_dim, mlp_width,
                              expert_factory=expert_factory) for i in range(depth)]
        for layer in self.layers:
            if layer.moe is not None:
                layer.moe._window_controller = self._window_controller
        self.norm = weights.take("model.norm.weight", (width,))
        self.head = None if config.get("tie_word_embeddings", False) else weights.matrix("lm_head", self.vocab_size, width)
        if weights.tensors:
            raise ValueError(f"Unexpected Qwen3 checkpoint tensor: {next(iter(weights.tensors))}")

    def set_window_size(self, window_size: int | None):
        self._window_controller.set_window_size(window_size)

    def prefill_last_hidden(self, ids, kv_cache=None):
        """Return the final prompt position and the complete prompt KV cache."""
        return self._hidden_states_with_cache(ids, kv_cache, True, last_only=ids.shape[1] > 1)

    def hidden_states_with_cache(self, ids, kv_cache=None, *, is_causal=True):
        return self._hidden_states_with_cache(ids, kv_cache, is_causal)

    def _hidden_states_with_cache(self, ids, kv_cache, is_causal, *, last_only=False):
        if not is_causal:
            raise ValueError("Native Qwen3 supports causal attention")
        if ids.shape[1] < 1:
            raise ValueError("Native Qwen3 requires at least one token per row")
        if self._window_controller.mode is None:
            if kv_cache is None:
                self._window_controller.begin_prefill()
            elif ids.shape[1] == 1:
                self._window_controller.begin_decode_step()
        offset = kv_cache[0].length if kv_cache is not None else 0
        if offset + ids.shape[1] > self.max_seq_len:
            raise ValueError(f"Qwen3 context exceeds the model limit of {self.max_seq_len} tokens")
        x = self.embedding.lookup(ids)
        updated = []
        final_layer = len(self.layers) - 1
        for index, layer in enumerate(self.layers):
            previous = None if kv_cache is None else kv_cache[index]
            x, present = layer(x, previous, self._rotary, self.max_seq_len,
                               last_only=last_only and index == final_layer)
            updated.append(present)
        return mx.fast.rms_norm(x, self.norm, self.eps), updated

    def hidden_states(self, ids, *, is_causal=True):
        return self.hidden_states_with_cache(ids, is_causal=is_causal)[0]

    def logits_from_hidden(self, hidden):
        return (self.embedding if self.head is None else self.head)(hidden)
