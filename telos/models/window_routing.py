"""Shared MLX window-pinned MoE routing, distillation controllers, and adapter loader."""

from __future__ import annotations

import hashlib
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
        self.adapter_disabled = False

    def add_selector_delta(self, base_logits: mx.array, x: mx.array) -> mx.array:
        if self.adapter_disabled or self.selector_a is None or self.selector_b is None:
            return base_logits
        xf = x.astype(mx.float32)
        delta = (xf @ self.selector_a.astype(mx.float32).T) @ self.selector_b.astype(mx.float32).T
        return base_logits + delta * self.selector_scale

    def add_token_delta(self, base_logits: mx.array, x: mx.array) -> mx.array:
        if self.adapter_disabled or self.token_a is None or self.token_b is None:
            return base_logits
        xf = x.astype(mx.float32)
        delta = (xf @ self.token_a.astype(mx.float32).T) @ self.token_b.astype(mx.float32).T
        return base_logits + delta * self.token_scale


class WindowPinController:
    """Tracks decode-step or sequence-level window pinning per layer."""

    def __init__(self, window_size: int | None = None):
        if window_size is not None and (type(window_size) is not int or window_size < 1):
            raise ValueError("window_size must be None or a positive integer")
        self.window_size = window_size
        self._active = False
        self._decode_step = 0
        self._pinned: dict[int, mx.array] = {}
        self.reuse_host_routes = False
        self.batch_nvfp4_experts = False
        self.compile_nvfp4_experts = False
        self.combine_nvfp4_experts = False
        self.host_routes = {}
        # Sequence-level distillation state.
        self.mode: str | None = None
        self.seq_window_size: int = 16
        self.prompt_lengths: list[int] = []
        self.chunk_offset: int = 0
        self.seq_pinned: dict[tuple[int, int], tuple[int, mx.array, mx.array]] = {}
        self.seq_last_chunk: dict[int, int] = {}
        self.seq_selector_logits: dict[int, list[mx.array]] = {}
        self.seq_selector_windows: dict[int, list[mx.array]] = {}
        self.seq_anchor_cache: dict[int, list[tuple[mx.array, mx.array, mx.array]]] = {}
        self.teacher_routes: dict[int, list[list[mx.array]]] = {}

    def set_window_size(self, window_size: int | None):
        if window_size is not None and (type(window_size) is not int or window_size < 1):
            raise ValueError("window_size must be None or a positive integer")
        self.window_size = window_size
        self.begin_prefill()

    def begin_prefill(self):
        self._active = False
        self._decode_step = 0
        self._pinned.clear()
        self.host_routes.clear()

    def begin_decode_step(self):
        if self.window_size is None:
            self._active = False
            return
        self._active = True
        if self._decode_step % self.window_size == 0:
            self._pinned.clear()
            self.host_routes.clear()
        self._decode_step += 1

    def reset_sequence_teacher(self, prompt_lengths: list[int]):
        self.mode = "teacher"
        self.prompt_lengths = list(prompt_lengths)
        self.chunk_offset = 0
        self.teacher_routes.clear()

    def reset_sequence_student(self, window_size: int, prompt_lengths: list[int]):
        self.mode = "student"
        self.seq_window_size = int(window_size)
        self.prompt_lengths = list(prompt_lengths)
        self.chunk_offset = 0
        self.seq_pinned.clear()
        self.seq_last_chunk.clear()
        self.seq_selector_logits.clear()
        self.seq_selector_windows.clear()
        self.seq_anchor_cache.clear()

    def clear_sequence(self):
        self.mode = None
        self.chunk_offset = 0
        self.seq_pinned.clear()
        self.seq_last_chunk.clear()
        self.seq_selector_logits.clear()
        self.seq_selector_windows.clear()
        self.seq_anchor_cache.clear()
        self.teacher_routes.clear()

    @property
    def active(self) -> bool:
        return self.mode is None and self.window_size is not None and self._active

    def get(self, layer: int) -> mx.array | None:
        return self._pinned.get(layer)

    def set(self, layer: int, indices: mx.array):
        self._pinned[layer] = indices


def _route_sequence(
    flat: mx.array,
    base_logits_fn,
    *,
    layer: int,
    top_k: int,
    route_scale: float,
    normalize: bool,
    base_bias: mx.array | None,
    adapter: RouterAdapter | None,
    controller: WindowPinController,
    activation: str,
    eps: float,
    output_dtype,
):
    batch = max(1, len(controller.prompt_lengths))
    chunk_len = flat.shape[0] // batch
    width = flat.shape[-1]
    base_logits = base_logits_fn(flat).astype(mx.float32)
    tok_logits = base_logits if adapter is None else adapter.add_token_delta(base_logits, flat)
    tok_probs = mx.softmax(tok_logits, axis=-1) if activation == "softmax" else mx.sigmoid(tok_logits)

    if controller.mode == "teacher":
        ranking = tok_probs if base_bias is None else tok_probs + base_bias.astype(mx.float32)
        indices = mx.stop_gradient(mx.argsort(-ranking, axis=-1)[..., :top_k])
        routes = controller.teacher_routes.setdefault(layer, [])
        if controller.chunk_offset == 0:
            routes.append([indices.reshape(batch, chunk_len, top_k)])
        else:
            routes[-1].append(indices.reshape(batch, chunk_len, top_k))
        selected = mx.take_along_axis(tok_probs, indices, axis=-1)
        if normalize:
            selected = selected / (selected.sum(axis=-1, keepdims=True) + eps)
        return indices, (selected * route_scale).astype(output_dtype)

    h_seq = flat.reshape(batch, chunk_len, width)
    win_size = controller.seq_window_size
    prompt_lens = mx.array(controller.prompt_lengths, dtype=mx.int32)[:, None]
    start = controller.chunk_offset
    positions = mx.arange(start, start + chunk_len, dtype=mx.int32)[None, :]
    is_resp = positions >= prompt_lens
    rel_pos = mx.maximum(positions - prompt_lens, 0)
    win_idx_global = rel_pos // win_size
    w_min = int(mx.min(win_idx_global).item())
    w_max = int(mx.max(win_idx_global).item())
    local_wins = max(1, w_max - w_min + 1)
    win_ids = mx.arange(w_min, w_min + local_wins, dtype=mx.int32)[None, :]
    anchors_global = prompt_lens + win_ids * win_size
    anchors_local = mx.clip(anchors_global - start, 0, chunk_len - 1)
    anchor_hidden = mx.stop_gradient(mx.take_along_axis(h_seq, mx.stop_gradient(anchors_local)[:, :, None], axis=1))
    anchor_flat = anchor_hidden.reshape(-1, width)
    sel_base = base_logits_fn(anchor_flat).astype(mx.float32)
    sel_logits = (sel_base if adapter is None else adapter.add_selector_delta(sel_base, anchor_flat)).reshape(batch, local_wins, -1)
    bias = adapter.selector_bias if (adapter is not None and adapter.selector_bias is not None and not adapter.adapter_disabled) else base_bias
    sel_logits_biased = sel_logits if bias is None else sel_logits + bias.astype(mx.float32)
    sel_probs = mx.softmax(sel_logits, axis=-1) if activation == "softmax" else mx.sigmoid(sel_logits)

    pin_key = (layer, start)
    cached = controller.seq_pinned.get(pin_key)
    continued = anchors_global < start
    if cached is None:
        sel_rank_probs = sel_probs if bias is None else sel_probs + bias.astype(mx.float32)
        pinned = mx.stop_gradient(mx.argsort(-sel_rank_probs, axis=-1)[..., :top_k])
        previous_start = controller.seq_last_chunk.get(layer)
        if previous_start is not None:
            prev_min, prev_pins, prev_probs = controller.seq_pinned[layer, previous_start]
            prev_idx = mx.clip(win_ids - prev_min, 0, prev_pins.shape[1] - 1)[:, :, None]
            pinned = mx.where(continued[..., None], mx.take_along_axis(prev_pins, prev_idx, axis=1), pinned)
            sel_probs = mx.where(continued[..., None], mx.take_along_axis(prev_probs, prev_idx, axis=1), sel_probs)
        controller.seq_pinned[pin_key] = (w_min, pinned, sel_probs)
        controller.seq_last_chunk[layer] = start
        # Only real anchors contribute to coverage, once per row and window.
        anchored_windows = mx.where((anchors_global >= start) & (anchors_global < start + chunk_len), win_ids, -1)
        controller.seq_selector_logits.setdefault(layer, []).append(sel_logits_biased)
        controller.seq_selector_windows.setdefault(layer, []).append(anchored_windows)
        controller.seq_anchor_cache.setdefault(layer, []).append(
            (mx.stop_gradient(anchor_hidden), mx.stop_gradient(sel_base.reshape(batch, local_wins, -1)), anchored_windows)
        )
    else:
        _, pinned, first_visit_probs = cached
        sel_probs = mx.where(continued[..., None], first_visit_probs, sel_probs)

    win_idx_local = mx.clip(win_idx_global - w_min, 0, local_wins - 1)
    pinned_tok = mx.take_along_axis(pinned, mx.stop_gradient(win_idx_local)[:, :, None], axis=1)
    sel_tok_probs = mx.take_along_axis(sel_probs, mx.stop_gradient(win_idx_local)[:, :, None], axis=1)

    prompt_ranking = tok_probs if base_bias is None else tok_probs + base_bias.astype(mx.float32)
    prompt_indices = mx.stop_gradient(mx.argsort(-prompt_ranking, axis=-1)[..., :top_k]).reshape(batch, chunk_len, top_k)
    indices = mx.where(is_resp[:, :, None], pinned_tok, prompt_indices).reshape(-1, top_k)

    selected = mx.take_along_axis(tok_probs, mx.stop_gradient(indices), axis=-1)
    if normalize:
        selected = selected / (selected.sum(axis=-1, keepdims=True) + eps)

    sel_gathered = mx.take_along_axis(sel_tok_probs.reshape(-1, sel_tok_probs.shape[-1]), mx.stop_gradient(indices), axis=-1)
    st_gate = 1.0 + (sel_gathered - mx.stop_gradient(sel_gathered))
    weights = mx.where(is_resp.reshape(-1, 1), selected * st_gate, selected)
    return indices, (weights * route_scale).astype(output_dtype)


def zero_grad_tree(tree):
    from mlx.utils import tree_map

    return tree_map(lambda arr: mx.zeros_like(arr), tree)


def add_grad_trees(dst, src):
    if isinstance(dst, dict):
        return {k: add_grad_trees(dst[k], src[k]) if k in src else dst[k] for k in dst}
    if isinstance(dst, list):
        return [add_grad_trees(d, s) for d, s in zip(dst, src, strict=True)]
    if isinstance(dst, tuple):
        return tuple(add_grad_trees(d, s) for d, s in zip(dst, src, strict=True))
    return dst + src


def _layer_router_adapter(layer_mod):
    if hasattr(layer_mod, "router_adapter"):
        return layer_mod, "router_adapter", layer_mod.router_adapter
    for sub_name in ("mlp", "moe"):
        sub = getattr(layer_mod, sub_name, None)
        if sub is not None and hasattr(sub, "router_adapter"):
            return sub, f"{sub_name}.router_adapter", sub.router_adapter
    return None, "", None


def compute_window_coverage_loss_and_grads(
    model,
    controller: WindowPinController,
    teacher_routes: dict[int, list[mx.array]],
    prompt_lengths: list[int],
    response_mask: mx.array,
) -> tuple[mx.array, dict[int, tuple[str, dict]]]:
    """Compute window selector CE loss and gradients directly from cached anchor states."""
    if not controller.seq_anchor_cache or not teacher_routes:
        return mx.array(0.0, dtype=mx.float32), {}
    batch, seq_len = response_mask.shape
    win_size = controller.seq_window_size
    prompt_lens = mx.array(prompt_lengths, dtype=mx.int32)[:, None]
    positions = mx.arange(seq_len, dtype=mx.int32)[None, :]
    valid_f = response_mask.astype(mx.float32)
    win_idx = mx.maximum(positions - prompt_lens, 0) // win_size

    losses = []
    grads_by_layer: dict[int, tuple[str, dict]] = {}
    for layer_idx, passes in teacher_routes.items():
        if layer_idx not in controller.seq_anchor_cache or not passes:
            continue
        _, attr_path, adapter = _layer_router_adapter(model.layers[layer_idx])
        if adapter is None:
            continue
        anchors = controller.seq_anchor_cache[layer_idx]
        anchor_h = anchors[0][0] if len(anchors) == 1 else mx.concatenate([a[0] for a in anchors], axis=1)
        base_logits = anchors[0][1] if len(anchors) == 1 else mx.concatenate([a[1] for a in anchors], axis=1)
        win_ids = anchors[0][2] if len(anchors) == 1 else mx.concatenate([a[2] for a in anchors], axis=1)
        num_experts = base_logits.shape[-1]
        num_wins = win_ids.shape[1]
        eye = mx.eye(num_experts, dtype=mx.float32)
        counts = mx.zeros((batch, num_wins, num_experts), dtype=mx.float32)
        for pass_indices in passes:
            one_hot = mx.sum(eye[pass_indices], axis=-2) * valid_f[:, :, None]
            stacked = mx.stack(
                [mx.sum(one_hot * (win_idx == win_ids[:, w : w + 1]).astype(mx.float32)[:, :, None], axis=1)
                 for w in range(num_wins)],
                axis=1,
            )
            counts = counts + stacked
        win_totals = mx.sum(counts, axis=-1, keepdims=True)
        has_tokens = (mx.squeeze(win_totals, axis=-1) > 0).astype(mx.float32)
        target_dist = counts / mx.maximum(win_totals, 1.0)
        flat_h = anchor_h.reshape(-1, anchor_h.shape[-1])
        flat_base = base_logits.reshape(-1, num_experts)

        def _layer_ce(r_params):
            orig = adapter.trainable_parameters()
            adapter.update(r_params)
            try:
                logits = adapter.add_selector_delta(flat_base, flat_h).reshape(batch, num_wins, num_experts)
                if adapter.selector_bias is not None and not adapter.adapter_disabled:
                    logits = logits + adapter.selector_bias.astype(mx.float32)
                log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
                ce = -mx.sum(target_dist * log_probs, axis=-1)
                return mx.sum(ce * has_tokens) / mx.maximum(mx.sum(has_tokens), 1.0)
            finally:
                adapter.update(orig)

        l_val, g_val = mx.value_and_grad(_layer_ce)(adapter.trainable_parameters())
        mx.eval(l_val, g_val)
        losses.append(l_val)
        grads_by_layer[layer_idx] = (attr_path, g_val)
    if not losses:
        return mx.array(0.0, dtype=mx.float32), {}
    scale = 1.0 / len(losses)
    scaled_grads = {
        layer_idx: (attr_path, {k: v * scale for k, v in g_val.items()})
        for layer_idx, (attr_path, g_val) in grads_by_layer.items()
    }
    return sum(losses) * scale, scaled_grads


def sequential_ssd_loss_and_grad(
    model,
    ids: mx.array,
    embed_fn,
    layer_call_fn,
    tail_loss_fn,
    *,
    controller: WindowPinController | None = None,
    teacher_routes: dict | None = None,
    prompt_lengths: list[int] | None = None,
    response_mask: mx.array | None = None,
    window_size: int | None = None,
    cov_weight: float = 0.0,
):
    """Differentiate sequential block models one layer at a time to bound SSD weight memory."""
    accum = zero_grad_tree(model.trainable_parameters())
    if controller is not None and window_size is not None and prompt_lengths is not None:
        controller.reset_sequence_student(window_size, prompt_lengths)
    try:
        x = embed_fn(ids)
        mx.eval(x)
        activations = [x]
        for mod in model.layers:
            x = layer_call_fn(mod, x)
            mx.eval(x)
            activations.append(x)

        loss_val, d_x = mx.value_and_grad(tail_loss_fn)(x)
        mx.eval(loss_val, d_x)

        if controller is not None and cov_weight > 0.0 and teacher_routes and prompt_lengths is not None and response_mask is not None:
            cov_loss, cov_grads = compute_window_coverage_loss_and_grads(
                model, controller, teacher_routes, prompt_lengths, response_mask,
            )
            loss_val = loss_val + cov_weight * cov_loss
            for layer_idx, (attr_path, g_adapt) in cov_grads.items():
                parts = attr_path.split(".")
                dst = accum["layers"][layer_idx]
                for p in parts[:-1]:
                    dst = dst[p]
                dst[parts[-1]] = add_grad_trees(dst[parts[-1]], {k: cov_weight * v for k, v in g_adapt.items()})
            mx.eval(loss_val)

        from mlx.utils import tree_flatten, tree_unflatten

        for idx in range(len(model.layers) - 1, -1, -1):
            mod = model.layers[idx]
            orig = mod.trainable_parameters()
            flat_items = tree_flatten(orig)
            keys = [k for k, _ in flat_items]
            vals = [v for _, v in flat_items]
            n_p = len(vals)
            x_in = activations[idx]

            def _one_layer(*args):
                l_params = tree_unflatten(list(zip(keys, args[:n_p], strict=True)))
                x_a = args[n_p]
                mod.update(l_params)
                try:
                    return [layer_call_fn(mod, x_a)]
                finally:
                    mod.update(orig)

            _, out_grads = mx.vjp(_one_layer, [*vals, x_in], [d_x])
            mx.eval(out_grads)
            g_layer = tree_unflatten(list(zip(keys, out_grads[:n_p], strict=True))) if n_p else {}
            d_x = out_grads[n_p]
            accum["layers"][idx] = add_grad_trees(accum["layers"][idx], g_layer)
        return loss_val, accum
    finally:
        if controller is not None:
            controller.clear_sequence()


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
    """Select top-k experts (per-token, decode-window, or sequence-distill) and score them."""
    flat = x.reshape(-1, x.shape[-1])
    if controller is not None and controller.mode in {"teacher", "student"}:
        return _route_sequence(
            flat,
            base_logits_fn,
            layer=layer,
            top_k=top_k,
            route_scale=route_scale,
            normalize=normalize,
            base_bias=base_bias,
            adapter=adapter,
            controller=controller,
            activation=activation,
            eps=eps,
            output_dtype=output_dtype,
        )

    base_logits = base_logits_fn(flat).astype(mx.float32)
    tok_logits = base_logits if adapter is None else adapter.add_token_delta(base_logits, flat)
    probs = mx.softmax(tok_logits, axis=-1) if activation == "softmax" else mx.sigmoid(tok_logits)

    if controller is not None and controller.active:
        pinned = controller.get(layer)
        if pinned is None:
            sel_logits = base_logits if adapter is None else adapter.add_selector_delta(base_logits, flat)
            sel_probs = mx.softmax(sel_logits, axis=-1) if activation == "softmax" else mx.sigmoid(sel_logits)
            bias = adapter.selector_bias if (adapter is not None and adapter.selector_bias is not None and not adapter.adapter_disabled) else base_bias
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


def compute_window_coverage_loss(
    controller: WindowPinController,
    teacher_routes: dict[int, list[mx.array]],
    prompt_lengths: list[int],
    response_mask: mx.array,
) -> mx.array:
    """Train RouterAdapter window selectors against empirical teacher expert distributions."""
    if not controller.seq_selector_logits or not teacher_routes:
        return mx.array(0.0, dtype=mx.float32)
    batch, seq_len = response_mask.shape
    win_size = controller.seq_window_size
    prompt_lens = mx.array(prompt_lengths, dtype=mx.int32)[:, None]
    positions = mx.arange(seq_len, dtype=mx.int32)[None, :]
    valid_f = response_mask.astype(mx.float32)
    win_idx = mx.maximum(positions - prompt_lens, 0) // win_size

    losses = []
    for layer, passes in teacher_routes.items():
        if layer not in controller.seq_selector_logits or not passes:
            continue
        logits_chunks = controller.seq_selector_logits[layer]
        sel_logits = logits_chunks[0] if len(logits_chunks) == 1 else mx.concatenate(logits_chunks, axis=1)
        win_chunks = controller.seq_selector_windows[layer]
        win_ids = win_chunks[0] if len(win_chunks) == 1 else mx.concatenate(win_chunks, axis=1)
        num_experts = sel_logits.shape[-1]
        num_wins = sel_logits.shape[1]
        eye = mx.eye(num_experts, dtype=mx.float32)
        counts = mx.zeros((batch, num_wins, num_experts), dtype=mx.float32)
        for pass_indices in passes:
            one_hot = mx.sum(eye[pass_indices], axis=-2) * valid_f[:, :, None]
            stacked = mx.stack(
                [mx.sum(one_hot * (win_idx == win_ids[:, w : w + 1]).astype(mx.float32)[:, :, None], axis=1)
                 for w in range(num_wins)],
                axis=1,
            )
            counts = counts + stacked
        win_totals = mx.sum(counts, axis=-1, keepdims=True)
        has_tokens = (mx.squeeze(win_totals, axis=-1) > 0).astype(mx.float32)
        target_dist = counts / mx.maximum(win_totals, 1.0)
        log_probs = sel_logits - mx.logsumexp(sel_logits, axis=-1, keepdims=True)
        ce = -mx.sum(target_dist * log_probs, axis=-1)
        losses.append(mx.sum(ce * has_tokens) / mx.maximum(mx.sum(has_tokens), 1.0))
    if not losses:
        return mx.array(0.0, dtype=mx.float32)
    return sum(losses) / len(losses)


def init_window_adapters(
    model,
    *,
    selector_rank: int = 32,
    selector_alpha: float = 32.0,
    token_router_rank: int = 16,
    token_scale: float = 2.0,
) -> list[str]:
    """Attach trainable RouterAdapter modules onto all MoE layers of an MLX model."""
    from telos.posttrain.mlx_adapters import _linear_shape

    if selector_rank < 1 or token_router_rank < 1 or selector_alpha <= 0:
        raise ValueError("Window adapter ranks and alpha must be positive")
    attached = []
    for idx, layer in enumerate(model.layers):
        if hasattr(layer, "router_adapter") and hasattr(layer, "router"):
            base_w, base_b = layer.router
            num_experts, width = base_w.shape
            target = layer
        else:
            target = next((sub for sub in (getattr(layer, "mlp", None), getattr(layer, "moe", None))
                           if sub is not None and hasattr(sub, "router_adapter")), None)
            if target is None:
                continue
            num_experts, width = _linear_shape(target.router)
            base_b = getattr(target, "bias", None)
        bias = (mx.zeros((num_experts,), dtype=mx.float32) if base_b is None
                else base_b.astype(mx.float32))
        adapter = RouterAdapter(
            selector_a=mx.random.uniform(-1 / math.sqrt(width), 1 / math.sqrt(width),
                                         (selector_rank, width), dtype=mx.float32),
            selector_b=mx.zeros((num_experts, selector_rank), dtype=mx.float32),
            selector_scale=selector_alpha / selector_rank,
            selector_bias=bias,
            token_a=mx.random.uniform(-1 / math.sqrt(width), 1 / math.sqrt(width),
                                      (token_router_rank, width), dtype=mx.float32),
            token_b=mx.zeros((num_experts, token_router_rank), dtype=mx.float32),
            token_scale=token_scale,
        )
        adapter.unfreeze()
        target.router_adapter = adapter
        attached.append(f"layers.{idx}.router_adapter")
    if not attached:
        raise ValueError("Window-pinned training requires a model with MoE router layers")
    return attached


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
        selector_bias=tensors.get("selector_bias", tensors.get("bias")),
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
    base_checkpoint: str | Path | None = None,
):
    """Load an H100 or native distilled window-routing + LoRA adapter onto an MLX MoE model."""
    from telos.models.checkpoint import resolve_checkpoint
    from telos.posttrain.mlx_adapters import LoRALinear

    path = Path(path)
    meta = {}
    if path.with_suffix(".json").is_file():
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if base_checkpoint is not None and "base_sha256" in meta:
        base_file = resolve_checkpoint(base_checkpoint)
        digest = hashlib.sha256()
        with open(base_file, "rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != meta["base_sha256"]:
            raise ValueError("Adapter base checkpoint hash mismatch")
    raw = mx.load(str(path))
    if not raw:
        raise ValueError(f"Empty window adapter file: {path}")

    win_meta = meta.get("window") if isinstance(meta.get("window"), dict) else {}
    eff_selector_alpha = float(win_meta.get("selector_alpha", meta.get("selector_alpha", selector_alpha)))
    eff_token_scale = float(win_meta.get("token_router_scale", meta.get("token_router_scale", token_scale)))

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
        if tail[-1] in ("selector_a", "selector_b", "selector_bias", "bias", "token_a", "token_b"):
            router_by_layer.setdefault(layer_idx, {})[tail[-1]] = tensor
        elif tail[-1] in ("lora_a", "lora_b"):
            # Supports Telos nested paths (`attention.q.lora_a`) and HF paths (`self_attn.gate_proj.gate_proj.lora_a`).
            mod_prefix = ".".join(tail[:-1])
            lora_pairs.setdefault((layer_idx, mod_prefix), {})[tail[-1]] = tensor

    for layer_idx, r_tensors in sorted(router_by_layer.items()):
        _attach_router_adapter(
            model, layer_idx, r_tensors,
            selector_alpha=eff_selector_alpha,
            token_scale=eff_token_scale,
        )

    modules = dict(model.named_modules())
    replacements = []
    for (layer_idx, mod_prefix), pair in sorted(lora_pairs.items()):
        if set(pair) != {"lora_a", "lora_b"}:
            raise ValueError(f"Incomplete LoRA pair for layer {layer_idx} projection {mod_prefix}")
        exact_path = f"layers.{layer_idx}.{mod_prefix}"
        if exact_path in modules:
            mod_path, base = exact_path, modules[exact_path]
        else:
            parts = mod_prefix.split(".")
            proj_name = parts[-2] if len(parts) >= 2 and parts[-1] == parts[-2] else parts[-1]
            resolved = _resolve_layer_module(modules, layer_idx, proj_name)
            if resolved is None:
                raise ValueError(f"Cannot map adapter projection {exact_path} onto {type(model).__name__}")
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

    effective_window = window_size if window_size is not None else win_meta.get("window_size", meta.get("window_size", 16))
    if effective_window is not None and hasattr(model, "set_window_size"):
        model.set_window_size(int(effective_window))
    mx.eval(model.parameters())
    return model
