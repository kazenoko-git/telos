"""Native MLX inference and window-pinned distillation for Qwen 3.5 MoE checkpoints."""

from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn

from .qwen3_mlx import _LayerCache, _Weights, _positive, _rope_parameters
from .qwen3_moe_mlx import RoutedMLP
from .window_routing import WindowPinController, sequential_ssd_loss_and_grad


def _merge_text_config(config):
    text = config.get("text_config")
    if isinstance(text, dict):
        merged = {**config, **text}
        if "quantization" in config:
            merged["quantization"] = config["quantization"]
        return merged
    return config


def _checkpoint_prefix(arrays):
    for name in arrays:
        if name.startswith("language_model.model."):
            return "language_model.model", "language_model.lm_head"
    return "model", "lm_head"


def _layer_types(config, depth):
    kinds = config.get("layer_types")
    if isinstance(kinds, (list, tuple)) and len(kinds) == depth:
        return list(kinds)
    interval = _positive(config, "full_attention_interval", 4)
    return ["full_attention" if (i + 1) % interval == 0 else "linear_attention" for i in range(depth)]


def _l2_normalize(x, eps=1e-6):
    return x * mx.rsqrt(mx.sum(x * x, axis=-1, keepdims=True) + eps)


class _GatedDeltaState(NamedTuple):
    conv_history: mx.array
    recurrent: mx.array
    length: int


class Qwen3_5RoutedMLP(RoutedMLP):
    def __init__(self, weights, layer, config, width, ranges, cache, *, prefix_root="model"):
        super().__init__(
            weights, layer, config, width, ranges, cache,
            module="mlp", prefix_root=prefix_root, switch_name="switch_mlp", activation="silu",
        )
        prefix = f"{prefix_root}.layers.{layer}.mlp"
        self.shared_expert = None
        self.shared_expert_gate = None
        shared_inner = config.get("shared_expert_intermediate_size") or config.get("moe_intermediate_size")
        if shared_inner and f"{prefix}.shared_expert.gate_proj.weight" in weights.tensors:
            self.shared_expert = _DenseFFN(weights, f"{prefix}.shared_expert", width, int(shared_inner))
            if f"{prefix}.shared_expert_gate.weight" in weights.tensors:
                self.shared_expert_gate = weights.matrix(f"{prefix}.shared_expert_gate", 1, width)

    def __call__(self, hidden):
        routed = self._eval_routed(hidden)
        if self.shared_expert is None:
            return routed
        shared = self.shared_expert(hidden)
        if self.shared_expert_gate is not None:
            shared = mx.sigmoid(self.shared_expert_gate(hidden)) * shared
        return routed + shared.astype(routed.dtype)


class _DenseFFN(nn.Module):
    def __init__(self, weights, prefix, width, inner):
        super().__init__()
        self.gate = weights.matrix(prefix + ".gate_proj", inner, width)
        self.up = weights.matrix(prefix + ".up_proj", inner, width)
        self.down = weights.matrix(prefix + ".down_proj", width, inner)

    def __call__(self, x):
        return self.down(nn.silu(self.gate(x)) * self.up(x))


class _GatedDeltaAttention(nn.Module):
    """Causal conv1d + Gated DeltaNet linear recurrence + gated RMSNorm."""

    def __init__(self, weights, prefix, config, width, eps):
        super().__init__()
        self.num_k_heads = _positive(config, "linear_num_key_heads")
        self.num_v_heads = _positive(config, "linear_num_value_heads")
        self.head_k_dim = _positive(config, "linear_key_head_dim")
        self.head_v_dim = _positive(config, "linear_value_head_dim")
        if self.num_v_heads % self.num_k_heads:
            raise ValueError("Qwen3.5 linear_num_value_heads must be divisible by linear_num_key_heads")
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.kernel = _positive(config, "linear_conv_kernel_dim", 4)
        self.history = self.kernel - 1
        self.eps = eps

        self.in_proj_qkv = weights.matrix(prefix + ".in_proj_qkv", self.conv_dim, width)
        self.in_proj_z = weights.matrix(prefix + ".in_proj_z", self.value_dim, width)
        self.in_proj_b = weights.matrix(prefix + ".in_proj_b", self.num_v_heads, width)
        self.in_proj_a = weights.matrix(prefix + ".in_proj_a", self.num_v_heads, width)
        self.out_proj = weights.matrix(prefix + ".out_proj", width, self.value_dim)
        self.conv1d_weight = weights.take(prefix + ".conv1d.weight", (self.conv_dim, self.kernel, 1))
        self.A_log = weights.take(prefix + ".A_log", (self.num_v_heads,))
        self.dt_bias = weights.take(prefix + ".dt_bias", (self.num_v_heads,))
        self.norm = weights.take(prefix + ".norm.weight", (self.head_v_dim,))

    def __call__(self, x, previous, _rotary, _limit, *, last_only=False):
        batch, length, _ = x.shape
        mixed_qkv = self.in_proj_qkv(x)
        if self.history:
            hist = (mx.zeros((batch, self.history, self.conv_dim), dtype=mixed_qkv.dtype)
                    if previous is None else previous.conv_history)
            conv_in = mx.concatenate((hist, mixed_qkv), axis=1)
            positions = mx.arange(conv_in.shape[1] - self.history, conv_in.shape[1])
            next_hist = mx.take(conv_in, positions, axis=1)
        else:
            conv_in = mixed_qkv
            next_hist = mixed_qkv[:, :0]
        conved = nn.silu(mx.conv1d(conv_in, self.conv1d_weight, groups=self.conv_dim))

        q = conved[:, :, : self.key_dim].reshape(batch, length, self.num_k_heads, self.head_k_dim)
        k = conved[:, :, self.key_dim : 2 * self.key_dim].reshape(batch, length, self.num_k_heads, self.head_k_dim)
        v = conved[:, :, 2 * self.key_dim :].reshape(batch, length, self.num_v_heads, self.head_v_dim)
        q = _l2_normalize(q.astype(mx.float32)) * (self.head_k_dim ** -0.5)
        k = _l2_normalize(k.astype(mx.float32))
        v = v.astype(mx.float32)

        repeat = self.num_v_heads // self.num_k_heads
        if repeat > 1:
            q = mx.repeat(q, repeat, axis=2)
            k = mx.repeat(k, repeat, axis=2)

        decay = mx.exp(-mx.exp(self.A_log.astype(mx.float32)) * nn.softplus(
            self.in_proj_a(x).astype(mx.float32) + self.dt_bias.astype(mx.float32),
        ))
        beta = mx.sigmoid(self.in_proj_b(x).astype(mx.float32))
        state = (mx.zeros((batch, self.num_v_heads, self.head_v_dim, self.head_k_dim), dtype=mx.float32)
                 if previous is None else previous.recurrent)

        outputs = []
        for t in range(length):
            state = state * decay[:, t, :, None, None]
            k_t = k[:, t]
            v_t = v[:, t]
            pred = mx.sum(state * k_t[:, :, None, :], axis=-1)
            delta = (v_t - pred) * beta[:, t, :, None]
            state = state + delta[:, :, :, None] * k_t[:, :, None, :]
            outputs.append(mx.sum(state * q[:, t, :, None, :], axis=-1))

        out = mx.stack(outputs, axis=1) if length > 1 else outputs[0][:, None]
        z = self.in_proj_z(x).reshape(batch, length, self.num_v_heads, self.head_v_dim)
        gated = mx.fast.rms_norm(out.astype(x.dtype), self.norm, self.eps) * nn.silu(z)
        if last_only and length > 1:
            gated = gated[:, -1:]
        projected = self.out_proj(gated.reshape(batch, 1 if last_only else length, self.value_dim))
        prev_len = 0 if previous is None else previous.length
        return projected, _GatedDeltaState(next_hist, state, prev_len + length)


class _FullAttention(nn.Module):
    """GQA full attention with optional output gate and partial RoPE."""

    def __init__(self, weights, prefix, config, width, eps):
        super().__init__()
        self.heads = _positive(config, "num_attention_heads")
        self.kv_heads = _positive(config, "num_key_value_heads", self.heads)
        self.head_dim = _positive(config, "head_dim", width // self.heads)
        if self.heads % self.kv_heads or self.head_dim % 2:
            raise ValueError("Qwen3.5 requires evenly grouped attention heads and an even head_dim")
        self.eps = eps
        self.scale = self.head_dim ** -0.5
        self.attn_output_gate = bool(config.get("attn_output_gate", True))
        bias = bool(config.get("attention_bias", False))
        q_out = self.heads * self.head_dim * (2 if self.attn_output_gate else 1)
        kv_out = self.kv_heads * self.head_dim

        self.query = weights.matrix(prefix + ".q_proj", q_out, width, bias=bias)
        self.key = weights.matrix(prefix + ".k_proj", kv_out, width, bias=bias)
        self.value = weights.matrix(prefix + ".v_proj", kv_out, width, bias=bias)
        self.output = weights.matrix(prefix + ".o_proj", width, self.heads * self.head_dim, bias=bias)
        self.q_norm = weights.take(prefix + ".q_norm.weight", (self.head_dim,))
        self.k_norm = weights.take(prefix + ".k_norm.weight", (self.head_dim,))

        rope_cfg = config.get("rope_parameters") or config.get("rope_scaling") or {}
        partial = float(rope_cfg.get("partial_rotary_factor", config.get("partial_rotary_factor", 1.0)))
        self.rotary_dims = max(2, int(self.head_dim * partial))
        self.rotary_dims -= self.rotary_dims % 2

    def __call__(self, x, previous, rotary, limit, *, last_only=False):
        batch, length, _ = x.shape
        q_len = 1 if last_only else length
        q_in = x[:, -1:] if last_only else x
        q_raw = self.query(q_in)
        if self.attn_output_gate:
            q_part, gate = mx.split(q_raw.reshape(batch, q_len, self.heads, 2 * self.head_dim), 2, axis=-1)
            gate = gate.reshape(batch, q_len, self.heads * self.head_dim)
        else:
            q_part = q_raw.reshape(batch, q_len, self.heads, self.head_dim)
            gate = None

        q = mx.fast.rms_norm(q_part, self.q_norm, self.eps).transpose(0, 2, 1, 3)
        k = mx.fast.rms_norm(self.key(x).reshape(batch, length, self.kv_heads, self.head_dim), self.k_norm, self.eps).transpose(0, 2, 1, 3)
        v = self.value(x).reshape(batch, length, self.kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        base, freqs, amplitude = rotary
        offset = previous.length if previous is not None else 0
        q = mx.fast.rope(q * amplitude, self.rotary_dims, traditional=False, base=base, freqs=freqs, scale=1.0,
                         offset=offset + length - q_len)
        k = mx.fast.rope(k * amplitude, self.rotary_dims, traditional=False, base=base, freqs=freqs, scale=1.0, offset=offset)
        if previous is None:
            present = _LayerCache(k, v, length)
        else:
            present = previous.append(k, v, limit)
            k, v = present.keys[:, :, : present.length], present.values[:, :, : present.length]
        mask = "causal" if q_len > 1 else None
        attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        attn = attn.transpose(0, 2, 1, 3).reshape(batch, q_len, self.heads * self.head_dim)
        if gate is not None:
            attn = attn * mx.sigmoid(gate)
        return self.output(attn), present


class _Block(nn.Module):
    def __init__(self, weights, index, kind, config, width, eps, *, expert_factory, prefix_root="model"):
        super().__init__()
        prefix = f"{prefix_root}.layers.{index}"
        self.input_norm = weights.take(prefix + ".input_layernorm.weight", (width,))
        self.post_norm = weights.take(prefix + ".post_attention_layernorm.weight", (width,))
        self.eps = eps
        if kind == "linear_attention":
            self.operator = _GatedDeltaAttention(weights, prefix + ".linear_attn", config, width, eps)
        else:
            self.operator = _FullAttention(weights, prefix + ".self_attn", config, width, eps)
        self.mlp = expert_factory(weights, index, config, width)

    def __call__(self, x, previous, rotary, limit, *, last_only=False):
        mixed, present = self.operator(mx.fast.rms_norm(x, self.input_norm, self.eps), previous, rotary, limit, last_only=last_only)
        base = x[:, -1:] if last_only and x.shape[1] > 1 else x
        h = base + mixed
        return h + self.mlp(mx.fast.rms_norm(h, self.post_norm, self.eps)), present


class Qwen3_5MoEMLX(nn.Module):
    """Native MLX Qwen 3.5 MoE engine with hybrid DeltaNet recurrence and SSD expert routing."""

    is_causal = True
    paradigm = "ar"
    # Recurrent GatedDeltaNet layers do not keep arbitrary earlier prefix states.
    supports_prefix_cache = False

    def __init__(self, config, arrays, *, expert_factory):
        super().__init__()
        text = _merge_text_config(config)
        width = _positive(text, "hidden_size")
        depth = _positive(text, "num_hidden_layers")
        self.vocab_size = _positive(text, "vocab_size")
        self.max_seq_len = _positive(text, "max_position_embeddings", 32768)
        self.eps = float(text.get("rms_norm_eps", 1e-6))

        heads = _positive(text, "num_attention_heads")
        head_dim = _positive(text, "head_dim", width // heads)
        rope_cfg = text.get("rope_parameters") or text.get("rope_scaling") or {}
        partial = float(rope_cfg.get("partial_rotary_factor", text.get("partial_rotary_factor", 1.0)))
        rotary_dims = max(2, int(head_dim * partial))
        rotary_dims -= rotary_dims % 2
        self._rotary = _rope_parameters(text, rotary_dims, self.max_seq_len)

        root, head_key = _checkpoint_prefix(arrays)
        weights = _Weights(arrays, text.get("quantization") or {})
        self.embedding = weights.matrix(f"{root}.embed_tokens", self.vocab_size, width)
        self._window_controller = WindowPinController()
        kinds = _layer_types(text, depth)
        self.layers = [
            _Block(weights, i, kind, text, width, self.eps, expert_factory=expert_factory, prefix_root=root)
            for i, kind in enumerate(kinds)
        ]
        for layer in self.layers:
            layer.mlp._window_controller = self._window_controller
        self.norm = weights.take(f"{root}.norm.weight", (width,))
        tied = text.get("tie_word_embeddings", False) and (head_key + ".weight") not in weights.tensors
        self.head = None if tied else weights.matrix(head_key, self.vocab_size, width)
        if weights.tensors:
            raise ValueError(f"Unexpected Qwen3.5 MoE checkpoint tensor: {next(iter(weights.tensors))}")

    def set_window_size(self, window_size: int | None):
        self._window_controller.set_window_size(window_size)

    def prefill_last_hidden(self, ids, kv_cache=None):
        return self._hidden_states_with_cache(ids, kv_cache, True, last_only=ids.shape[1] > 1)

    def hidden_states_with_cache(self, ids, kv_cache=None, *, is_causal=True):
        return self._hidden_states_with_cache(ids, kv_cache, is_causal)

    def _hidden_states_with_cache(self, ids, kv_cache, is_causal, *, last_only=False):
        if not is_causal:
            raise ValueError("Native Qwen3.5 MoE supports causal attention")
        if ids.shape[1] < 1:
            raise ValueError("Native Qwen3.5 MoE requires at least one token per row")
        if self._window_controller.mode is None:
            if kv_cache is None:
                self._window_controller.begin_prefill()
            elif ids.shape[1] == 1:
                self._window_controller.begin_decode_step()
        offset = kv_cache[0].length if kv_cache is not None else 0
        if offset + ids.shape[1] > self.max_seq_len:
            raise ValueError(f"Qwen3.5 MoE context exceeds the model limit of {self.max_seq_len} tokens")
        x = self.embedding.lookup(ids)
        updated = []
        final_layer = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            prev = None if kv_cache is None else kv_cache[i]
            x, present = layer(x, prev, self._rotary, self.max_seq_len, last_only=last_only and i == final_layer)
            updated.append(present)
        return mx.fast.rms_norm(x, self.norm, self.eps), updated

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
            lambda mod, x: mod(x, None, self._rotary, self.max_seq_len)[0],
            lambda x: loss_from_logits_fn(self.logits_from_hidden(mx.fast.rms_norm(x, self.norm, self.eps))[:, :-1]),
            controller=self._window_controller,
            teacher_routes=teacher_routes,
            prompt_lengths=prompt_lengths,
            response_mask=response_mask,
            window_size=window_size,
            cov_weight=cov_weight,
        )
