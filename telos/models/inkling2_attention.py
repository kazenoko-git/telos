"""Native Inkling text attention and causal convolution state."""

from functools import lru_cache
import math
from itertools import accumulate
from typing import NamedTuple

import mlx.core as mx
from mlx import nn

from .inkling2 import _integer, _positive_number
from .mlx_quantized import AffineWeights
from .qwen3_mlx import _LayerCache, _QuantizedProjection


class AttentionState(NamedTuple):
    kv: _LayerCache
    key_history: mx.array
    value_history: mx.array


def attention_dimensions(config, sliding):
    return tuple(_integer(config, key) for key in (
        ("swa_num_attention_heads", "swa_num_key_value_heads", "swa_head_dim") if sliding
        else ("num_attention_heads", "num_key_value_heads", "head_dim")
    ))


def _convolution_context(x, history, kernel):
    if history is None:
        history = mx.zeros((x.shape[0], kernel - 1, x.shape[-1]), dtype=mx.float32)
    context = mx.concatenate((history, x.astype(mx.float32)), axis=1)
    # Gather a compact tail without keeping the entire prefill window alive.
    tail = mx.take(context, mx.arange(context.shape[1] - kernel + 1, context.shape[1]), axis=1)
    return context, tail


@lru_cache(maxsize=4)
def _compiled_static_conv(kernel):
    scale = math.sqrt(kernel + 1)

    def _run(x, history, weight):
        context = mx.concatenate((history, x.astype(mx.float32)), axis=1)
        tail = context[:, 1:, :]
        result = mx.conv1d(context, weight.astype(mx.float32), groups=x.shape[-1])
        result = (result + x.astype(mx.float32)) / scale
        return result.astype(x.dtype), tail

    return mx.compile(_run)


class StaticConvolution(nn.Module):
    def __init__(self, weights, prefix, channels, kernel):
        super().__init__()
        self.kernel = kernel
        self.weight = weights.take(prefix + ".conv1d.weight", (channels, kernel, 1))

    def __call__(self, x, history, *, compiled=False):
        if compiled and history is not None and x.shape[1] == 1:
            return _compiled_static_conv(self.kernel)(x, history, self.weight)
        context, tail = _convolution_context(x, history, self.kernel)
        result = mx.conv1d(context, self.weight.astype(mx.float32), groups=x.shape[-1])
        result = (result + x.astype(mx.float32)) / math.sqrt(self.kernel + 1)
        return result.astype(x.dtype), tail


class DynamicConvolution(nn.Module):
    def __init__(self, weights, prefix, width, channels, heads, kernel):
        super().__init__()
        self.kernel = kernel
        self.heads = heads
        self.static = weights.take(prefix + ".static_weight", (kernel, channels))
        self.dynamic = weights.matrix(prefix + ".weight_proj", heads * kernel, width)

    def __call__(self, x, history, inputs):
        context, tail = _convolution_context(x, history, self.kernel)
        batch, length, channels = x.shape
        shape = batch, length, self.heads, channels // self.heads
        dynamic = self.dynamic(inputs).astype(mx.float32).reshape(batch, length, self.heads, 1, self.kernel)
        weights = dynamic + self.static.astype(mx.float32).T.reshape(self.heads, -1, self.kernel)
        result = mx.zeros(shape, dtype=mx.float32)
        for lag in range(self.kernel):
            start = self.kernel - 1 - lag
            result = result + context[:, start:start + length].reshape(shape) * weights[..., lag]
        result = (result.reshape(x.shape) + x.astype(mx.float32)) / math.sqrt(2 * self.kernel + 1)
        return result.astype(x.dtype), tail


@lru_cache(maxsize=8)
def _compiled_clip_heads(threshold, has_scale):
    if has_scale:
        def _run(x, scale):
            value = x.astype(mx.float32) * scale.astype(mx.float32)
            norm = mx.sqrt(mx.sum(mx.square(value), axis=-1, keepdims=True))
            return (value * mx.minimum(threshold / (norm + 1e-6), 1.0)).astype(x.dtype)

        return mx.compile(_run)

    def _run(x):
        value = x.astype(mx.float32)
        norm = mx.sqrt(mx.sum(mx.square(value), axis=-1, keepdims=True))
        return (value * mx.minimum(threshold / (norm + 1e-6), 1.0)).astype(x.dtype)

    return mx.compile(_run)


def _clip_heads(x, threshold, scale=None, *, compiled=False):
    if compiled:
        if scale is not None:
            return _compiled_clip_heads(threshold, True)(x, scale)
        return _compiled_clip_heads(threshold, False)(x)
    value = x.astype(mx.float32)
    if scale is not None:
        value = value * scale.astype(mx.float32)
    norm = mx.sqrt(mx.sum(mx.square(value), axis=-1, keepdims=True))
    return (value * mx.minimum(threshold / (norm + 1e-6), 1.0)).astype(x.dtype)


def _relative_attention_mask(relative, sink, keys, window, *, fused=False):
    batch, length, _, extent = relative.shape
    if fused and batch == length == 1 and relative.dtype == sink.dtype:
        from .inkling2_mask import fused_decode_mask

        return fused_decode_mask(relative, sink, keys, window)
    distance = mx.arange(keys - length, keys)[:, None] - mx.arange(keys)[None, :]
    bias = mx.take_along_axis(relative.transpose(0, 2, 1, 3), mx.clip(distance, 0, extent - 1)[None, None], axis=-1)
    bias = mx.where((distance >= 0) & (distance < extent), bias, 0)
    allowed = distance >= 0
    if window is not None:
        allowed = allowed & (distance < window)
    mask = mx.where(allowed, bias, -math.inf)
    # A zero key and value let a learned logit absorb probability without adding an output.
    return mx.concatenate((mask, sink.transpose(0, 2, 1)[..., None]), axis=-1)


class InklingAttention(nn.Module):
    def __init__(self, weights, prefix, config, *, sliding):
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = attention_dimensions(config, sliding)
        if self.heads % self.kv_heads:
            raise ValueError("Inkling 2 attention heads must be evenly grouped")
        self.window = _integer(config, "sliding_window_size") if sliding else None
        self.extent = self.window or _integer(config, "rel_extent")
        self.d_rel = _integer(config, "d_rel")
        self.eps = _positive_number(config, "rms_norm_eps")
        self.qk_clip = _positive_number(config, "qk_l2_clip_threshold")
        self.rel_clip = _positive_number(config, "rel_l2_clip_threshold")
        self.value_norm = config["use_value_norm_no_gamma"]
        self.log_floor = None if sliding else _integer(config, "log_scaling_n_floor")
        self.log_alpha = None if sliding else _positive_number(config, "log_scaling_alpha")
        width = _integer(config, "hidden_size")
        kernel = _integer(config, "sconv_kernel_size")
        query, kv = self.heads * self.head_dim, self.kv_heads * self.head_dim
        self.q = weights.matrix(prefix + ".q_proj", query, width, bias=config.get("q_bias", False))
        self.k = weights.matrix(prefix + ".k_proj", kv, width)
        self.v = weights.matrix(prefix + ".v_proj", kv, width)
        self.r = weights.matrix(prefix + ".r_proj", self.heads * self.d_rel, width)
        self.gate = weights.matrix(prefix + ".gate_proj", query, width)
        self.sink = weights.matrix(prefix + ".sink_proj", self.heads, width)
        self.output = weights.matrix(prefix + ".o_proj", width, query, bias=config.get("o_bias", False))
        self._query_group = None
        self.q_scale = weights.take(prefix + ".q_l2_clip.scale", (self.heads, self.head_dim))
        self.r_scale = weights.take(prefix + ".rel_l2_clip.scale", (self.heads, self.d_rel))
        self.relative = weights.take(prefix + ".rel_logits_proj.proj", (self.d_rel, self.extent))
        key_conv, value_conv = prefix + ".k_sconv", prefix + ".v_sconv"
        if sliding:
            self.key_conv = StaticConvolution(weights, key_conv, kv, kernel)
            self.value_conv = StaticConvolution(weights, value_conv, kv, kernel)
        else:
            self.key_conv = DynamicConvolution(weights, key_conv, width, kv, self.kv_heads, kernel)
            self.value_conv = DynamicConvolution(weights, value_conv, width, kv, self.kv_heads, kernel)

    def prepare_query_group(self):
        from telos.posttrain.mlx_adapters import LoRALinear

        if self._query_group is not None:
            return True
        projections = (self.q, self.r, self.sink, self.gate)
        bases = [layer.base if isinstance(layer, LoRALinear) else layer for layer in projections]
        if not all(isinstance(layer, _QuantizedProjection) for layer in bases):
            return False
        parts = [layer.packed for layer in bases]
        first = parts[0]
        if any((p.bits, p.group_size, p.weight.shape[1], p.scales.dtype)
               != (first.bits, first.group_size, first.weight.shape[1], first.scales.dtype) for p in parts[1:]):
            return False

        ends = list(accumulate(p.weight.shape[0] for p in parts))
        packed = AffineWeights(
            mx.concatenate([p.weight for p in parts]), mx.concatenate([p.scales for p in parts]),
            mx.concatenate([p.biases for p in parts]),
            shape=(ends[-1], first.weight.shape[1] * 32 // first.bits), group_size=first.group_size, bits=first.bits,
        )
        mx.eval(packed.parameters())
        # Original projection methods use row views of the same packed storage.
        for part, start, end in zip(parts, (0, *ends[:-1]), ends, strict=True):
            part.weight, part.scales, part.biases = packed.weight[start:end], packed.scales[start:end], packed.biases[start:end]
        self._query_group = (packed, ends[:-1], tuple(zip(projections, bases, strict=True)))
        return True

    def _query_projections(self, x, *, grouped):
        if not grouped or self._query_group is None or x.shape[:2] != (1, 1):
            return self.q(x), self.r(x), self.sink(x), self.gate(x)
        packed, boundaries, members = self._query_group
        results = []
        for out, (projection, base) in zip(mx.split(packed.as_linear(x), boundaries, axis=-1), members, strict=True):
            if base.bias is not None:
                out = out + base.bias
            if projection is not base:
                out = projection.add_delta(x, out)
            results.append(out)
        return results

    def __call__(self, q_input, k_input, v_input, previous, offset, limit, *, profile=None, grouped_query=False,
                 fused_mask=False, kv_cursor=None, compiled_helpers=False):
        batch, length, _ = q_input.shape
        k, v = self.k(k_input), self.v(v_input)
        if profile is not None:
            profile.evaluate(mx.eval, k, v, section="attention_projections")
        kh = None if previous is None else previous.key_history
        vh = None if previous is None else previous.value_history
        if self.window is None:
            k, kh = self.key_conv(k, kh, k_input)
            v, vh = self.value_conv(v, vh, v_input)
        else:
            k, kh = self.key_conv(k, kh, compiled=compiled_helpers)
            v, vh = self.value_conv(v, vh, compiled=compiled_helpers)
        if profile is not None:
            profile.evaluate(mx.eval, k, v, kh, vh, section="attention_convolutions")

        q, r, sink, gate = self._query_projections(q_input, grouped=grouped_query)
        if profile is not None:
            profile.evaluate(mx.eval, q, r, sink, gate, section="attention_projections")
        q = _clip_heads(q.reshape(batch, length, self.heads, self.head_dim), self.qk_clip, self.q_scale,
                        compiled=compiled_helpers)
        k = _clip_heads(k.reshape(batch, length, self.kv_heads, self.head_dim), self.qk_clip,
                        compiled=compiled_helpers)
        v = v.reshape(k.shape)
        if self.value_norm:
            v = mx.fast.rms_norm(v.astype(mx.float32), None, self.eps).astype(q.dtype)
        r = _clip_heads(r.reshape(batch, length, self.heads, self.d_rel), self.rel_clip, self.r_scale,
                        compiled=compiled_helpers)
        if self.log_floor is not None:
            pos = (mx.arange(offset + 1, offset + length + 1, dtype=mx.float32) if kv_cursor is None
                   else (kv_cursor + 1).reshape(1).astype(mx.float32))
            scale = (1 + self.log_alpha * mx.log(mx.maximum(pos / self.log_floor, 1)))[None, :, None, None]
            q = (q.astype(mx.float32) * scale).astype(q.dtype)
            r = (r.astype(mx.float32) * scale).astype(r.dtype)

        q, k, v = (mx.transpose(a, (0, 2, 1, 3)) for a in (q, k, v))
        if kv_cursor is not None:
            from .inkling2_mask import fixed_decode_mask

            k = mx.slice_update(previous.kv.keys, k, kv_cursor, [2])
            v = mx.slice_update(previous.kv.values, v, kv_cursor, [2])
            kv = _LayerCache(k, v, 0)  # The host associates the returned tensors with the decoded length.
            relative = r @ self.relative.astype(r.dtype)
            mask = fixed_decode_mask(relative, sink, kv_cursor, k.shape[2], self.window)
        elif previous is None:
            kv = _LayerCache(k, v, length)
        elif self.window is None:
            kv = previous.kv.append(k, v, limit)
        else:
            kv = _LayerCache(mx.concatenate((previous.kv.keys, k), axis=2),
                             mx.concatenate((previous.kv.values, v), axis=2), previous.kv.length + length)
        if kv_cursor is None:
            k, v = kv.keys[:, :, :kv.length], kv.values[:, :, :kv.length]
            relative = r @ self.relative.astype(r.dtype)
            mask = _relative_attention_mask(relative, sink, kv.length, self.window, fused=fused_mask)
            k, v = (mx.pad(a, ((0, 0), (0, 0), (0, 1), (0, 0))) for a in (k, v))
        if profile is not None:
            profile.evaluate(mx.eval, q, k, v, mask, section="attention_setup")
        output = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim ** -0.5, mask=mask)
        if profile is not None:
            profile.evaluate(mx.eval, output, section="attention_sdpa")
        output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        output = (output.astype(mx.float32) * mx.sigmoid(gate.astype(mx.float32))).astype(q.dtype)

        if kv_cursor is None and self.window is not None and kv.length >= self.window:
            # The current query sees up to window keys. Future queries need window - 1.
            tail = mx.arange(kv.length - self.window + 1, kv.length)
            kv = _LayerCache(mx.take(kv.keys, tail, axis=2), mx.take(kv.values, tail, axis=2), self.window - 1)
        if profile is not None:
            profile.evaluate(mx.eval, output, kv, section="attention_setup")
        output = self.output(output)
        if profile is not None:
            profile.evaluate(mx.eval, output, section="attention_projections")
        return output, AttentionState(kv, kh, vh)
