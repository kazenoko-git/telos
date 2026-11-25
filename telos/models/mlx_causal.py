"""Native MLX causal inference for Llama, Mistral, Qwen2, and Phi-3/Phi-4 checkpoints."""

import math

from telos.training.core import metal_usable

if not metal_usable():
    raise ImportError("MLX is installed but Metal is unavailable in this session.")

import mlx.core as mx
import mlx.nn as nn

from .qwen3_mlx import _LayerCache, _Weights, _positive, _rope_parameters as _base_rope_parameters


def _causal_rope_parameters(config, width, max_seq_len):
    """Extend Qwen3 RoPE/YaRN support with Llama 3 smooth frequency scaling."""
    scaling = config.get("rope_parameters") or config.get("rope_scaling") or {}
    kind = scaling.get("rope_type", scaling.get("type", "default"))
    if kind != "llama3":
        return _base_rope_parameters(config, width, max_seq_len)

    base = float(scaling.get("rope_theta", config.get("rope_theta", 500000.0)))
    factor = float(scaling["factor"])
    low_freq_factor = float(scaling.get("low_freq_factor", 1.0))
    high_freq_factor = float(scaling.get("high_freq_factor", 4.0))
    old_ctx = float(_positive(scaling, "original_max_position_embeddings", 8192))
    if not math.isfinite(base) or base <= 1 or not math.isfinite(factor) or factor < 1:
        raise ValueError("Invalid Llama 3 RoPE base or scaling factor")
    if not 0 < low_freq_factor < high_freq_factor < math.inf:
        raise ValueError("Invalid Llama 3 RoPE frequency factors")

    freqs = base ** (mx.arange(0, width, 2, dtype=mx.float32) / width)
    wavelens = 2.0 * math.pi * freqs
    low_wavelen = old_ctx / low_freq_factor
    high_wavelen = old_ctx / high_freq_factor

    scaled = freqs * factor
    smooth = (old_ctx / wavelens - low_freq_factor) / (high_freq_factor - low_freq_factor)
    interpolated = freqs / ((1.0 - smooth) / factor + smooth)
    waves = mx.where(wavelens < high_wavelen, freqs, mx.where(wavelens > low_wavelen, scaled, interpolated))
    return None, waves, 1.0


class _StandardCausalBlock(nn.Module):
    """RMSNorm + GQA + RoPE + SwiGLU block for Llama, Mistral, and Qwen2 checkpoints."""

    def __init__(self, weights, index, config, width, heads, kv_heads, head_dim, mlp_width, *, qkv_bias, o_bias):
        super().__init__()
        prefix = f"model.layers.{index}"
        attn = prefix + ".self_attn"
        mlp = prefix + ".mlp"
        self.query = weights.matrix(attn + ".q_proj", heads * head_dim, width, bias=qkv_bias)
        self.key = weights.matrix(attn + ".k_proj", kv_heads * head_dim, width, bias=qkv_bias)
        self.value = weights.matrix(attn + ".v_proj", kv_heads * head_dim, width, bias=qkv_bias)
        self.output = weights.matrix(attn + ".o_proj", width, heads * head_dim, bias=o_bias)
        self.input_norm = weights.take(prefix + ".input_layernorm.weight", (width,))
        self.post_norm = weights.take(prefix + ".post_attention_layernorm.weight", (width,))
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

        q = self.query(query_input).reshape(query_shape).transpose(0, 2, 1, 3)
        k = self.key(hidden).reshape(shape).transpose(0, 2, 1, 3)
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
        mask = "causal" if query_length > 1 else None
        attention = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        attention = attention.transpose(0, 2, 1, 3).reshape(batch, query_length, -1)
        if last_only:
            x = x[:, -1:]
        x = x + self.output(attention)
        hidden = mx.fast.rms_norm(x, self.post_norm, self.eps)
        return x + self.down(nn.silu(self.gate(hidden)) * self.up(hidden)), present


class _Phi3Block(nn.Module):
    """RMSNorm + fused QKV GQA + RoPE + fused gate/up SwiGLU block for Phi-3/Phi-4."""

    def __init__(self, weights, index, config, width, heads, kv_heads, head_dim, mlp_width):
        super().__init__()
        prefix = f"model.layers.{index}"
        attn = prefix + ".self_attn"
        mlp = prefix + ".mlp"
        q_dim = heads * head_dim
        kv_dim = kv_heads * head_dim
        self.qkv = weights.matrix(attn + ".qkv_proj", q_dim + 2 * kv_dim, width)
        self.output = weights.matrix(attn + ".o_proj", width, q_dim)
        self.input_norm = weights.take(prefix + ".input_layernorm.weight", (width,))
        self.post_norm = weights.take(prefix + ".post_attention_layernorm.weight", (width,))
        self.gate_up = weights.matrix(mlp + ".gate_up_proj", 2 * mlp_width, width)
        self.down = weights.matrix(mlp + ".down_proj", width, mlp_width)
        self.eps = float(config.get("rms_norm_eps", 1e-5))
        self.head_dim = head_dim
        self.q_dim = q_dim
        self.kv_dim = kv_dim
        self.scale = head_dim ** -0.5

    def __call__(self, x, previous, rotary, limit, *, last_only=False):
        batch, length, _ = x.shape
        hidden = mx.fast.rms_norm(x, self.input_norm, self.eps)
        qkv = self.qkv(hidden)
        q, k, v = mx.split(qkv, [self.q_dim, self.q_dim + self.kv_dim], axis=-1)
        query_length = 1 if last_only else length
        if last_only:
            q = q[:, -1:]
        q = q.reshape(batch, query_length, -1, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch, length, -1, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, length, -1, self.head_dim).transpose(0, 2, 1, 3)

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
        mask = "causal" if query_length > 1 else None
        attention = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        attention = attention.transpose(0, 2, 1, 3).reshape(batch, query_length, -1)
        if last_only:
            x = x[:, -1:]
        x = x + self.output(attention)
        hidden = mx.fast.rms_norm(x, self.post_norm, self.eps)
        gate, up = mx.split(self.gate_up(hidden), 2, axis=-1)
        return x + self.down(nn.silu(gate) * up), present


class _BaseCausalMLX(nn.Module):
    """Shared causal forward, prefill, and KV-cache lifecycle for MLX architectures."""

    is_causal = True
    paradigm = "ar"

    def _init_common(self, config, arrays, *, default_qkv_bias=False, default_o_bias=False, packed_phi3=False):
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
            raise ValueError("Causal MLX model requires evenly grouped heads and an even head dimension")
        if (config.get("use_sliding_window") or config.get("sliding_window") is not None
                or config.get("partial_rotary_factor", 1.0) != 1.0
                or config.get("hidden_act", "silu") != "silu"):
            raise ValueError("Native causal MLX inference requires full attention, full rotary dimensions, and SiLU")

        self._rotary = _causal_rope_parameters(config, head_dim, self.max_seq_len)
        weights = _Weights(arrays, config.get("quantization") or {})
        self.embedding = weights.matrix("model.embed_tokens", self.vocab_size, width)
        if packed_phi3:
            self.layers = [_Phi3Block(weights, i, config, width, heads, kv_heads, head_dim, mlp_width) for i in range(depth)]
        else:
            qkv_bias = bool(config.get("attention_bias", default_qkv_bias))
            o_bias = bool(config.get("attention_bias", default_o_bias)) if default_qkv_bias == default_o_bias else default_o_bias
            self.layers = [
                _StandardCausalBlock(
                    weights, i, config, width, heads, kv_heads, head_dim, mlp_width,
                    qkv_bias=qkv_bias, o_bias=o_bias,
                )
                for i in range(depth)
            ]
        self.norm = weights.take("model.norm.weight", (width,))
        self.head = None if config.get("tie_word_embeddings", False) else weights.matrix("lm_head", self.vocab_size, width)
        if weights.tensors:
            raise ValueError(f"Unexpected checkpoint tensor: {next(iter(weights.tensors))}")

    def prefill_last_hidden(self, ids, kv_cache=None):
        """Return the final prompt position and the complete prompt KV cache."""
        return self._hidden_states_with_cache(ids, kv_cache, True, last_only=ids.shape[1] > 1)

    def hidden_states_with_cache(self, ids, kv_cache=None, *, is_causal=True):
        return self._hidden_states_with_cache(ids, kv_cache, is_causal)

    def _hidden_states_with_cache(self, ids, kv_cache, is_causal, *, last_only=False):
        if not is_causal:
            raise ValueError("Native causal MLX inference supports causal attention only")
        if ids.shape[1] < 1:
            raise ValueError("Native causal MLX inference requires at least one token per row")
        offset = kv_cache[0].length if kv_cache is not None else 0
        if offset + ids.shape[1] > self.max_seq_len:
            raise ValueError(f"Context exceeds the model limit of {self.max_seq_len} tokens")
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


class LlamaMLX(_BaseCausalMLX):
    """Native MLX inference for dense or affine-quantized Llama / Mistral checkpoints."""

    def __init__(self, config, arrays):
        super().__init__()
        self._init_common(config, arrays, default_qkv_bias=False, default_o_bias=False)


class Qwen2MLX(_BaseCausalMLX):
    """Native MLX inference for dense or affine-quantized Qwen2 / Qwen2.5 checkpoints."""

    def __init__(self, config, arrays):
        super().__init__()
        self._init_common(config, arrays, default_qkv_bias=True, default_o_bias=False)


class Phi3MLX(_BaseCausalMLX):
    """Native MLX inference for dense or affine-quantized Phi-3 / Phi-4 Mini checkpoints."""

    def __init__(self, config, arrays):
        super().__init__()
        self._init_common(config, arrays, packed_phi3=True)
