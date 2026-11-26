"""Native MLX inference and window-pinned distillation for Gemma 4 (26B-A4B MoE and dense)."""

import math

import mlx.core as mx
import mlx.nn as nn

from .qwen3_mlx import _LayerCache, _Weights, _positive, _rope_parameters
from .qwen3_moe_mlx import RoutedMLP
from .window_routing import WindowPinController, compute_route, sequential_ssd_loss_and_grad


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


def _gemma4_rope(text, layer_type, head_dim, max_seq_len):
    rope_map = text.get("rope_parameters") or {}
    cfg = rope_map.get(layer_type) if isinstance(rope_map.get(layer_type), dict) else rope_map
    partial = float(cfg.get("partial_rotary_factor", 0.25 if layer_type == "full_attention" else 1.0))
    dims = max(2, int(head_dim * partial))
    dims -= dims % 2
    if cfg.get("rope_type") == "proportional":
        theta = float(cfg.get("rope_theta", 1_000_000.0))
        waves = theta ** (mx.arange(0, dims, 2, dtype=mx.float32) / head_dim)
        mx.eval(waves)
        return dims, (None, waves, 1.0)
    merged = {**text, "rope_parameters": cfg}
    return dims, _rope_parameters(merged, dims, max_seq_len)


def _ffn_activate(x, kind):
    if kind in ("gelu_pytorch_tanh", "gelu_approx", "gelu"):
        return nn.gelu_approx(x)
    return nn.silu(x)


class _DenseMLP(nn.Module):
    def __init__(self, weights, prefix, width, inner, act_kind):
        super().__init__()
        self.act_kind = act_kind
        self.gate = weights.matrix(prefix + ".gate_proj", inner, width)
        self.up = weights.matrix(prefix + ".up_proj", inner, width)
        self.down = weights.matrix(prefix + ".down_proj", width, inner)

    def __call__(self, x):
        return self.down(_ffn_activate(self.gate(x), self.act_kind) * self.up(x))


class Gemma4RoutedMLP(RoutedMLP):
    """Expert bank wrapper for Gemma 4 `experts.switch_glu` with router-supplied routes."""

    def __init__(self, weights, layer, config, width, ranges, cache, *, prefix_root="model"):
        super().__init__(
            weights, layer, config, width, ranges, cache,
            module="experts", prefix_root=prefix_root, switch_name="switch_glu",
            load_router=False,
            activation=config.get("hidden_activation", config.get("hidden_act", "gelu_pytorch_tanh")),
            inner_size=_positive(config, "moe_intermediate_size"),
            count=_positive(config, "num_experts"),
            top_k=_positive(config, "top_k_experts", config.get("num_experts_per_tok")),
        )


class _Gemma4Attention(nn.Module):
    """Hybrid sliding/full GQA with q/k/v RMSNorm, optional k_eq_v, and proportional RoPE."""

    def __init__(self, weights, prefix, kind, text, width, max_seq_len, eps):
        super().__init__()
        self.is_sliding = kind == "sliding_attention"
        self.heads = _positive(text, "num_attention_heads")
        base_kv = _positive(text, "num_key_value_heads", self.heads)
        base_dim = _positive(text, "head_dim", width // self.heads)
        global_dim = text.get("global_head_dim") or base_dim
        self.head_dim = int(base_dim if self.is_sliding else global_dim)
        global_kv = text.get("num_global_key_value_heads") or base_kv
        self.use_k_eq_v = bool(text.get("attention_k_eq_v", False) and not self.is_sliding)
        self.kv_heads = int(global_kv if self.use_k_eq_v else base_kv)
        self.sliding_window = int(text.get("sliding_window", 512))
        self.eps = eps
        bias = bool(text.get("attention_bias", False))
        self.rotary_dims, self.rotary = _gemma4_rope(text, kind, self.head_dim, max_seq_len)

        self.query = weights.matrix(prefix + ".q_proj", self.heads * self.head_dim, width, bias=bias)
        self.key = weights.matrix(prefix + ".k_proj", self.kv_heads * self.head_dim, width, bias=bias)
        has_v = not self.use_k_eq_v and (prefix + ".v_proj.weight") in weights.tensors
        self.value = weights.matrix(prefix + ".v_proj", self.kv_heads * self.head_dim, width, bias=bias) if has_v else None
        self.output = weights.matrix(prefix + ".o_proj", width, self.heads * self.head_dim, bias=bias)
        self.q_norm = weights.take(prefix + ".q_norm.weight", (self.head_dim,))
        self.k_norm = weights.take(prefix + ".k_norm.weight", (self.head_dim,))

    def __call__(self, x, previous, limit, *, last_only=False):
        batch, length, _ = x.shape
        q_len = 1 if last_only else length
        q_in = x[:, -1:] if last_only else x
        q = mx.fast.rms_norm(self.query(q_in).reshape(batch, q_len, self.heads, self.head_dim), self.q_norm, self.eps).transpose(0, 2, 1, 3)

        k_proj = self.key(x).reshape(batch, length, self.kv_heads, self.head_dim)
        k = mx.fast.rms_norm(k_proj, self.k_norm, self.eps).transpose(0, 2, 1, 3)
        v_raw = k_proj if self.value is None else self.value(x).reshape(batch, length, self.kv_heads, self.head_dim)
        v = mx.fast.rms_norm(v_raw, None, self.eps).transpose(0, 2, 1, 3)

        base, freqs, amplitude = self.rotary
        offset = previous.length if previous is not None else 0
        q = mx.fast.rope(q * amplitude, self.rotary_dims, traditional=False, base=base, freqs=freqs, scale=1.0,
                         offset=offset + length - q_len)
        k = mx.fast.rope(k * amplitude, self.rotary_dims, traditional=False, base=base, freqs=freqs, scale=1.0, offset=offset)

        if previous is None:
            present = _LayerCache(k, v, length)
        else:
            present = previous.append(k, v, limit)
            k, v = present.keys[:, :, : present.length], present.values[:, :, : present.length]

        total_len = k.shape[2]
        if self.is_sliding and total_len > self.sliding_window and q_len == 1:
            start = total_len - self.sliding_window
            k, v = k[:, :, start:], v[:, :, start:]
            mask = None
        elif q_len > 1:
            if self.is_sliding:
                q_pos = offset + length - q_len + mx.arange(q_len, dtype=mx.int32)[:, None]
                k_pos = mx.arange(total_len, dtype=mx.int32)[None, :]
                allowed = (k_pos <= q_pos) & ((q_pos - k_pos) < self.sliding_window)
                mask = mx.where(allowed, 0.0, -1e9).astype(q.dtype)
            else:
                mask = "causal"
        else:
            mask = None

        attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0, mask=mask)
        return self.output(attn.transpose(0, 2, 1, 3).reshape(batch, q_len, self.heads * self.head_dim)), present


class _DecoderLayer(nn.Module):
    def __init__(self, weights, index, kind, text, width, ranges, max_seq_len, eps, *, expert_factory, prefix_root="model"):
        super().__init__()
        prefix = f"{prefix_root}.layers.{index}"
        self.layer = index
        self.eps = eps
        self.enable_moe = bool(text.get("enable_moe_block", False))
        act_kind = text.get("hidden_activation", text.get("hidden_act", "gelu_pytorch_tanh"))
        mlp_width = _positive(text, "intermediate_size")

        self.input_norm = weights.take(prefix + ".input_layernorm.weight", (width,))
        self.post_attn_norm = weights.take(prefix + ".post_attention_layernorm.weight", (width,))
        self.self_attn = _Gemma4Attention(weights, prefix + ".self_attn", kind, text, width, max_seq_len, eps)

        self.pre_ffn_norm = weights.take(prefix + ".pre_feedforward_layernorm.weight", (width,))
        self.post_ffn_norm = weights.take(prefix + ".post_feedforward_layernorm.weight", (width,))
        self.mlp = _DenseMLP(weights, prefix + ".mlp", width, mlp_width, act_kind)

        self.router_adapter = None
        self._window_controller = None
        if self.enable_moe:
            count = _positive(text, "num_experts")
            self.top_k = _positive(text, "top_k_experts", text.get("num_experts_per_tok"))
            self.root_size = width ** -0.5
            self.router = weights.matrix(prefix + ".router.proj", count, width)
            self.router_scale = weights.take(prefix + ".router.scale")
            self.per_expert_scale = weights.take(prefix + ".router.per_expert_scale", (count,))
            self.pre_ffn_norm_2 = weights.take(prefix + ".pre_feedforward_layernorm_2.weight", (width,))
            self.post_ffn_norm_1 = weights.take(prefix + ".post_feedforward_layernorm_1.weight", (width,))
            self.post_ffn_norm_2 = weights.take(prefix + ".post_feedforward_layernorm_2.weight", (width,))
            self.experts = expert_factory(weights, index, text, width)
        self.layer_scalar = (weights.take(prefix + ".layer_scalar")
                             if prefix + ".layer_scalar" in weights.tensors else None)

    def __call__(self, x, previous, limit, *, last_only=False):
        h_attn, present = self.self_attn(
            mx.fast.rms_norm(x, self.input_norm, self.eps), previous, limit, last_only=last_only,
        )
        base = x[:, -1:] if last_only and x.shape[1] > 1 else x
        h = base + mx.fast.rms_norm(h_attn, self.post_attn_norm, self.eps)

        h1 = self.mlp(mx.fast.rms_norm(h, self.pre_ffn_norm, self.eps))
        if self.enable_moe:
            h1 = mx.fast.rms_norm(h1, self.post_ffn_norm_1, self.eps)
            router_in = mx.fast.rms_norm(h, None, self.eps) * self.router_scale * self.root_size
            indices, scores = compute_route(
                router_in.reshape(-1, router_in.shape[-1]),
                self.router,
                layer=self.layer,
                top_k=self.top_k,
                route_scale=1.0,
                normalize=True,
                adapter=self.router_adapter,
                controller=self._window_controller,
                activation="softmax",
                eps=0.0,
                output_dtype=mx.float32,
            )
            scores = (scores * self.per_expert_scale[indices].astype(mx.float32)).astype(h.dtype)
            h2_in = mx.fast.rms_norm(h, self.pre_ffn_norm_2, self.eps)
            h2 = mx.fast.rms_norm(self.experts._eval_routed(h2_in, indices=indices, scores=scores), self.post_ffn_norm_2, self.eps)
            ffn_out = h1 + h2
        else:
            ffn_out = h1

        out = h + mx.fast.rms_norm(ffn_out, self.post_ffn_norm, self.eps)
        if self.layer_scalar is not None:
            out = out * self.layer_scalar
        return out, present


class Gemma4MLX(nn.Module):
    """Native MLX Gemma 4 engine (26B-A4B MoE with SSD expert routing and dense variants)."""

    is_causal = True
    paradigm = "ar"
    supports_prefix_cache = True

    def __init__(self, config, arrays, *, expert_factory=None):
        super().__init__()
        text = _merge_text_config(config)
        width = _positive(text, "hidden_size")
        depth = _positive(text, "num_hidden_layers")
        self.vocab_size = _positive(text, "vocab_size")
        self.max_seq_len = _positive(text, "max_position_embeddings", 32768)
        self.eps = float(text.get("rms_norm_eps", 1e-6))
        self.embed_scale = math.sqrt(width)
        self.final_softcap = text.get("final_logit_softcapping")

        kinds = text.get("layer_types")
        if not isinstance(kinds, (list, tuple)) or len(kinds) != depth:
            kinds = ["full_attention" if (i + 1) % 6 == 0 else "sliding_attention" for i in range(depth)]

        root, head_key = _checkpoint_prefix(arrays)
        weights = _Weights(arrays, text.get("quantization") or {})
        self.embedding = weights.matrix(f"{root}.embed_tokens", self.vocab_size, width)
        self._window_controller = WindowPinController()
        self.layers = [
            _DecoderLayer(
                weights, i, kinds[i], text, width, arrays, self.max_seq_len, self.eps,
                expert_factory=expert_factory, prefix_root=root,
            )
            for i in range(depth)
        ]
        for layer in self.layers:
            layer._window_controller = self._window_controller
        self.norm = weights.take(f"{root}.norm.weight", (width,))
        tied = text.get("tie_word_embeddings", True) and (head_key + ".weight") not in weights.tensors
        self.head = None if tied else weights.matrix(head_key, self.vocab_size, width)
        if weights.tensors:
            raise ValueError(f"Unexpected Gemma 4 checkpoint tensor: {next(iter(weights.tensors))}")

    def set_window_size(self, window_size: int | None):
        self._window_controller.set_window_size(window_size)

    def prefill_last_hidden(self, ids, kv_cache=None):
        return self._hidden_states_with_cache(ids, kv_cache, True, last_only=ids.shape[1] > 1)

    def hidden_states_with_cache(self, ids, kv_cache=None, *, is_causal=True):
        return self._hidden_states_with_cache(ids, kv_cache, is_causal)

    def _hidden_states_with_cache(self, ids, kv_cache, is_causal, *, last_only=False):
        if not is_causal:
            raise ValueError("Native Gemma 4 supports causal attention")
        if ids.shape[1] < 1:
            raise ValueError("Native Gemma 4 requires at least one token per row")
        if self._window_controller.mode is None:
            if kv_cache is None:
                self._window_controller.begin_prefill()
            elif ids.shape[1] == 1:
                self._window_controller.begin_decode_step()
        offset = kv_cache[0].length if kv_cache is not None else 0
        if offset + ids.shape[1] > self.max_seq_len:
            raise ValueError(f"Gemma 4 context exceeds the model limit of {self.max_seq_len} tokens")
        x = self.embedding.lookup(ids) * self.embed_scale
        updated = []
        final_layer = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            prev = None if kv_cache is None else kv_cache[i]
            x, present = layer(x, prev, self.max_seq_len, last_only=last_only and i == final_layer)
            updated.append(present)
        return mx.fast.rms_norm(x, self.norm, self.eps), updated

    def hidden_states(self, ids, *, is_causal=True):
        return self.hidden_states_with_cache(ids, is_causal=is_causal)[0]

    def logits_from_hidden(self, hidden):
        logits = (self.embedding if self.head is None else self.head)(hidden)
        if self.final_softcap:
            cap = float(self.final_softcap)
            logits = cap * mx.tanh(logits.astype(mx.float32) / cap)
        return logits

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
            lambda x: self.embedding.lookup(x) * self.embed_scale,
            lambda mod, x: mod(x, None, self.max_seq_len)[0],
            lambda x: loss_from_logits_fn(self.logits_from_hidden(mx.fast.rms_norm(x, self.norm, self.eps))[:, :-1]),
            controller=self._window_controller,
            teacher_routes=teacher_routes,
            prompt_lengths=prompt_lengths,
            response_mask=response_mask,
            window_size=window_size,
            cov_weight=cov_weight,
        )
