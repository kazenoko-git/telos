"""Shared MLX window-pinned MoE routing and distilled window adapter loader."""

from __future__ import annotations

import json
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

_HF_TO_MLX_ATTR = {
    "q_proj": ("q", "query", "q_proj"),
    "k_proj": ("k", "key", "k_proj"),
    "v_proj": ("v", "value", "v_proj"),
    "o_proj": ("output", "out", "o_proj"),
    "gate_proj": ("gate", "gate_proj"),
    "up_proj": ("up", "up_proj"),
    "down_proj": ("down", "down_proj"),
    "r_proj": ("r", "r_proj"),
    "sink_proj": ("sink", "sink_proj"),
    "in_proj": ("input", "in_proj"),
    "out_proj": ("output", "out_proj"),
}


class RouterAdapter(nn.Module):
    """Optional low-rank selector and token-router deltas for window-pinned MoE routing."""

    def __init__(
        self,
        *,
        selector_a: mx.array | None = None,
        selector_b: mx.array | None = None,
        selector_scale: float = 1.0,
        selector_bias: mx.array | None = None,
        token_a: mx.array | None = None,
        token_b: mx.array | None = None,
        token_scale: float = 2.0,
    ):
        super().__init__()
        self.selector_a = selector_a
        self.selector_b = selector_b
        self.selector_scale = float(selector_scale)
        self.selector_bias = selector_bias
        self.token_a = token_a
        self.token_b = token_b
        self.token_scale = float(token_scale)

    def add_selector_delta(self, base_logits: mx.array, x: mx.array) -> mx.array:
        if self.selector_a is None or self.selector_b is None:
            return base_logits
        xf = x.astype(mx.float32)
        delta = (xf @ self.selector_a.astype(mx.float32).T) @ self.selector_b.astype(mx.float32).T
        return base_logits + delta * self.selector_scale

    def add_token_delta(self, base_logits: mx.array, x: mx.array) -> mx.array:
        if self.token_a is None or self.token_b is None:
            return base_logits
        xf = x.astype(mx.float32)
        delta = (xf @ self.token_a.astype(mx.float32).T) @ self.token_b.astype(mx.float32).T
        return base_logits + delta * self.token_scale


class WindowPinController:
    """Tracks per-request decode window boundaries and caches pinned expert IDs per layer."""

    def __init__(self, window_size: int | None = None):
        if window_size is not None and (type(window_size) is not int or window_size < 1):
            raise ValueError("window_size must be None or a positive integer")
        self.window_size = window_size
        self._active = False
        self._decode_step = 0
        self._pinned: dict[int, mx.array] = {}

    def set_window_size(self, window_size: int | None):
        if window_size is not None and (type(window_size) is not int or window_size < 1):
            raise ValueError("window_size must be None or a positive integer")
        self.window_size = window_size
        self.begin_prefill()

    def begin_prefill(self):
        self._active = False
        self._decode_step = 0
        self._pinned.clear()

    def begin_decode_step(self):
        if self.window_size is None:
            self._active = False
            return
        self._active = True
        if self._decode_step % self.window_size == 0:
            self._pinned.clear()
        self._decode_step += 1

    @property
    def active(self) -> bool:
        return self.window_size is not None and self._active

    def get(self, layer: int) -> mx.array | None:
        return self._pinned.get(layer)

    def set(self, layer: int, indices: mx.array):
        self._pinned[layer] = indices


def compute_route(
    x: mx.array,
    base_logits_fn,
    *,
    layer: int,
    top_k: int,
    route_scale: float = 1.0,
    normalize: bool = True,
    base_bias: mx.array | None = None,
    adapter: RouterAdapter | None = None,
    controller: WindowPinController | None = None,
    activation: str = "sigmoid",
    eps: float = 1e-20,
    output_dtype=mx.float32,
):
    """Select top-k experts (or reuse window-pinned IDs in decode) and score them per token."""
    flat = x.reshape(-1, x.shape[-1])
    base_logits = base_logits_fn(flat).astype(mx.float32)
    tok_logits = base_logits if adapter is None else adapter.add_token_delta(base_logits, flat)
    probs = mx.softmax(tok_logits, axis=-1) if activation == "softmax" else mx.sigmoid(tok_logits)

    if controller is not None and controller.active:
        pinned = controller.get(layer)
        if pinned is None:
            sel_logits = base_logits if adapter is None else adapter.add_selector_delta(base_logits, flat)
            sel_probs = mx.softmax(sel_logits, axis=-1) if activation == "softmax" else mx.sigmoid(sel_logits)
            bias = adapter.selector_bias if (adapter is not None and adapter.selector_bias is not None) else base_bias
            ranking = sel_probs if bias is None else sel_probs + bias.astype(mx.float32)
            pinned = mx.argsort(-ranking, axis=-1)[..., :top_k]
            controller.set(layer, pinned)
        indices = mx.broadcast_to(pinned[:1], (flat.shape[0], top_k)) if pinned.shape[0] == 1 else pinned
    else:
        ranking = probs if base_bias is None else probs + base_bias.astype(mx.float32)
        indices = mx.argsort(-ranking, axis=-1)[..., :top_k]

    selected = mx.take_along_axis(probs, mx.stop_gradient(indices), axis=-1)
    if normalize:
        selected = selected / (selected.sum(axis=-1, keepdims=True) + eps)
    return indices, (selected * route_scale).astype(output_dtype)


def _resolve_layer_module(modules: dict, layer_idx: int, proj_name: str) -> tuple[str, nn.Module] | None:
    """Resolve HF or Telos projection names on layers.{layer_idx}."""
    candidates = _HF_TO_MLX_ATTR.get(proj_name, (proj_name,))
    prefix = f"layers.{layer_idx}."
    for sub in ("attention", "operator", "self_attn", "mlp", ""):
        for attr in candidates:
            full = f"{prefix}{sub}.{attr}" if sub else f"{prefix}{attr}"
            if full in modules:
                return full, modules[full]
    return None


def _attach_router_adapter(model, layer_idx: int, tensors: dict[str, mx.array], *, selector_alpha: float, token_scale: float):
    if layer_idx >= len(model.layers):
        raise ValueError(f"Adapter layer {layer_idx} exceeds model depth {len(model.layers)}")
    sel_a = tensors.get("selector_a")
    sel_b = tensors.get("selector_b")
    sel_scale = selector_alpha / sel_a.shape[0] if sel_a is not None else 1.0
    adapter = RouterAdapter(
        selector_a=sel_a,
        selector_b=sel_b,
        selector_scale=sel_scale,
        selector_bias=tensors.get("bias"),
        token_a=tensors.get("token_a"),
        token_b=tensors.get("token_b"),
        token_scale=token_scale,
    )
    layer = model.layers[layer_idx]
    for target in (layer, getattr(layer, "mlp", None), getattr(layer, "moe", None)):
        if target is not None and hasattr(target, "router_adapter"):
            target.router_adapter = adapter
            return
    raise ValueError(f"Layer {layer_idx} does not expose a compatible MoE router_adapter slot")


def apply_window_adapter(
    model,
    path: str | Path,
    *,
    window_size: int | None = None,
    lora_alpha: float | None = None,
    selector_alpha: float = 32.0,
    token_scale: float = 2.0,
):
    """Load an H100 or native distilled window-routing + LoRA adapter onto an MLX MoE model."""
    from telos.posttrain.mlx_adapters import LoRALinear

    path = Path(path)
    meta = {}
    if path.with_suffix(".json").is_file():
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    raw = mx.load(str(path))
    if not raw:
        raise ValueError(f"Empty window adapter file: {path}")

    router_by_layer: dict[int, dict[str, mx.array]] = {}
    lora_pairs: dict[tuple[int, str], dict[str, mx.array]] = {}

    for key, tensor in raw.items():
        parts = key.split(".")
        if "layers" not in parts:
            continue
        li = parts.index("layers")
        if li + 1 >= len(parts) or not parts[li + 1].isdigit():
            continue
        layer_idx = int(parts[li + 1])
        tail = parts[li + 2 :]
        if tail[-1] in ("selector_a", "selector_b", "bias", "token_a", "token_b"):
            router_by_layer.setdefault(layer_idx, {})[tail[-1]] = tensor
        elif tail[-1] in ("lora_a", "lora_b"):
            # Supports both `...self_attn.q_proj.lora_a` and `...self_attn.gate_proj.gate_proj.lora_a`.
            proj_name = tail[-2]
            if len(tail) >= 3 and tail[-2] == tail[-3]:
                proj_name = tail[-3]
            lora_pairs.setdefault((layer_idx, proj_name), {})[tail[-1]] = tensor

    for layer_idx, r_tensors in sorted(router_by_layer.items()):
        _attach_router_adapter(
            model, layer_idx, r_tensors,
            selector_alpha=float(meta.get("selector_alpha", selector_alpha)),
            token_scale=float(meta.get("token_router_scale", token_scale)),
        )

    modules = dict(model.named_modules())
    replacements = []
    for (layer_idx, proj_name), pair in sorted(lora_pairs.items()):
        if set(pair) != {"lora_a", "lora_b"}:
            raise ValueError(f"Incomplete LoRA pair for layer {layer_idx} projection {proj_name}")
        resolved = _resolve_layer_module(modules, layer_idx, proj_name)
        if resolved is None:
            raise ValueError(f"Cannot map adapter projection layers.{layer_idx}.{proj_name} onto {type(model).__name__}")
        mod_path, base = resolved
        if isinstance(base, LoRALinear):
            base = base.base
        rank = int(pair["lora_a"].shape[0])
        alpha = float(lora_alpha if lora_alpha is not None else meta.get("lora_alpha", 2.0 * rank))
        wrapper = LoRALinear(base, rank=rank, alpha=alpha, dropout=0.0)
        wrapper.lora_a = pair["lora_a"].astype(mx.float32)
        wrapper.lora_b = pair["lora_b"].astype(mx.float32)
        wrapper.scale = alpha / rank
        replacements.append((mod_path, wrapper))

    if replacements:
        model.update_modules(tree_unflatten(replacements))

    effective_window = window_size if window_size is not None else meta.get("window_size", 16)
    if effective_window is not None and hasattr(model, "set_window_size"):
        model.set_window_size(int(effective_window))
    mx.eval(model.parameters())
    return model
