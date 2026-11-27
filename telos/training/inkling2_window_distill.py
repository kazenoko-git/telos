"""PyTorch window-pinned routing and LoRA distillation across MoE and dense models."""

from contextlib import contextmanager
import json
import math
from pathlib import Path
from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

DEFAULT_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "out_proj",
    "gate_proj",
    "in_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "qkv_proj",
    "r_proj",
    "sink_proj",
)
_MOE_BLOCK_ATTRS = ("mlp", "feed_forward", "moe")
_OPERATOR_ATTRS = ("self_attn", "linear_attn", "conv", "attention", "operator")
_MLP_LORA_PROJS = ("gate_proj", "up_proj", "down_proj")
_SOFTMAX_FAMILIES = ("qwen", "mixtral", "gemma4")



class LoRALinear(nn.Module):
    """Low-rank adapter wrapping a frozen linear projection."""

    def __init__(self, base: nn.Linear, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        self.base = base
        for param in self.base.parameters():
            param.requires_grad_(False)
        self.rank = rank
        self.scale = alpha / rank
        dtype = base.weight.dtype
        device = base.weight.device
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32, device=device))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32, device=device))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self._base_dtype = dtype
        self.adapter_disabled = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.adapter_disabled:
            return out
        delta = F.linear(F.linear(x.float(), self.lora_a), self.lora_b) * self.scale
        return out + delta.to(out.dtype)


@contextmanager
def _disabled_lora(model: nn.Module):
    modules = [m for m in model.modules() if isinstance(m, LoRALinear)]
    for module in modules:
        module.adapter_disabled = True
    try:
        yield
    finally:
        for module in modules:
            module.adapter_disabled = False


class WindowSelector(nn.Module):
    """Select window experts with a low-rank or full-size weight update and learned bias."""

    def __init__(self, base_weight: torch.Tensor, base_bias: torch.Tensor, rank: int = 32, alpha: float = 32.0, router_training: str = "adapter"):
        super().__init__()
        num_experts, hidden_dim = base_weight.shape
        device = base_weight.device
        self.register_buffer("base_weight", base_weight.detach().float().clone())
        self.selector_a = nn.Parameter(torch.empty(rank, hidden_dim, dtype=torch.float32, device=device))
        self.selector_b = nn.Parameter(torch.zeros(num_experts, rank, dtype=torch.float32, device=device))
        nn.init.kaiming_uniform_(self.selector_a, a=math.sqrt(5))
        self.selector_weight = None
        if router_training == "full":
            self.selector_a = None
            self.selector_b = None
            self.selector_weight = nn.Parameter(torch.zeros_like(self.base_weight))
        self.scale = alpha / rank
        self.bias = nn.Parameter(base_bias.detach().float().clone())

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        base = F.linear(hidden.float(), self.base_weight)
        delta = (F.linear(hidden.float(), self.selector_weight) if self.selector_weight is not None
                 else F.linear(F.linear(hidden.float(), self.selector_a), self.selector_b) * self.scale)
        return base + delta


class DistillContext:
    """Shared per-forward routing state compatible with gradient checkpointing."""

    def __init__(self, window_size: int = 16):
        self.window_size = window_size
        self.mode = "teacher"
        self.prompt_lengths: torch.Tensor | None = None
        self.pinned: dict[int, torch.Tensor] = {}
        self.anchor_hidden: dict[int, torch.Tensor] = {}
        self.selectors: dict[int, WindowSelector] = {}
        self.teacher_routes: dict[int, list[torch.Tensor]] = {}
        self.record_teacher = False

    def reset_forward(self, mode: str, prompt_lengths: torch.Tensor, *, record_teacher: bool = False):
        self.mode = mode
        self.prompt_lengths = prompt_lengths
        self.pinned.clear()
        self.anchor_hidden.clear()
        self.teacher_routes.clear()
        self.record_teacher = record_teacher


class WindowPinnedRouter(nn.Module):
    """Wraps any MoE gate/router for teacher or W-token window-pinned student passes."""

    def __init__(
        self,
        base_gate: nn.Module,
        layer_idx: int,
        context: DistillContext,
        *,
        top_k: int | None = None,
        num_experts: int | None = None,
        hidden_dim: int | None = None,
        base_bias: torch.Tensor | None = None,
        routed_scaling_factor: float | None = None,
        activation: str = "sigmoid",
        normalize: bool = True,
        eps: float = 1e-20,
        return_probs: bool = False,
        per_expert_scale: torch.Tensor | None = None,
        selector_rank: int = 32,
        selector_alpha: float = 32.0,
        token_router_rank: int = 16,
        token_scale: float = 2.0,
        router_training: str = "adapter",
    ):
        super().__init__()
        self.base_gate = base_gate
        for param in self.base_gate.parameters():
            param.requires_grad_(False)
        self.layer_idx = layer_idx
        self.context = context
        proj_mod = getattr(base_gate, "proj", base_gate)
        w_experts, w_hidden = proj_mod.weight.shape
        self.top_k = int(top_k if top_k is not None else getattr(base_gate, "top_k"))
        self.num_experts = int(num_experts if num_experts is not None else getattr(base_gate, "num_experts", w_experts))
        self.hidden_dim = int(hidden_dim if hidden_dim is not None else getattr(base_gate, "hidden_dim", w_hidden))
        self.routed_scaling_factor = float(
            routed_scaling_factor if routed_scaling_factor is not None
            else getattr(base_gate, "routed_scaling_factor", 1.0)
        )
        self.activation = activation
        self.normalize = bool(normalize)
        self.eps = float(eps)
        self._return_probs = bool(return_probs)
        self._has_base_preprocess = hasattr(base_gate, "norm") or hasattr(base_gate, "scalar_root_size")
        self.register_buffer("_proj_weight", proj_mod.weight.detach())

        expert_scale = per_expert_scale if per_expert_scale is not None else getattr(base_gate, "per_expert_scale", None)
        if expert_scale is not None:
            self.register_buffer("per_expert_scale", expert_scale.detach().float().clone())
        else:
            self.per_expert_scale = None

        detected_bias = base_bias
        if detected_bias is None:
            detected_bias = getattr(base_gate, "e_score_correction_bias", getattr(base_gate, "expert_bias", None))
        if detected_bias is not None:
            self.register_buffer("base_bias", detected_bias.detach().float().clone())
            init_bias = self.base_bias
        else:
            self.base_bias = None
            init_bias = torch.zeros(self.num_experts, dtype=torch.float32, device=proj_mod.weight.device)

        self.selector = WindowSelector(
            proj_mod.weight, init_bias, rank=selector_rank, alpha=selector_alpha, router_training=router_training,
        )
        context.selectors[layer_idx] = self.selector
        self.token_a = nn.Parameter(torch.empty(token_router_rank, self.hidden_dim, dtype=torch.float32,
                                                device=proj_mod.weight.device))
        self.token_b = nn.Parameter(torch.zeros(self.num_experts, token_router_rank, dtype=torch.float32,
                                                device=proj_mod.weight.device))
        nn.init.kaiming_uniform_(self.token_a, a=math.sqrt(5))
        self.token_scale = float(token_scale)
        self.token_weight = None
        if router_training == "full":
            self.token_a = None
            self.token_b = None
            self.token_weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim,
                                                        device=proj_mod.weight.device, dtype=torch.float32))

    def _prepare_input(self, flat: torch.Tensor) -> torch.Tensor:
        if not self._has_base_preprocess:
            return flat.float()
        x = flat
        norm = getattr(self.base_gate, "norm", None)
        if norm is not None:
            x = norm(x)
        scale = getattr(self.base_gate, "scale", None)
        if scale is not None:
            x = x * scale
        root = getattr(self.base_gate, "scalar_root_size", None)
        if root is not None:
            x = x * root
        return x.float()

    def _activate(self, logits: torch.Tensor) -> torch.Tensor:
        if self.activation == "softmax":
            return F.softmax(logits, dim=-1)
        return torch.sigmoid(logits)

    def _select_and_score(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        probs = self._activate(logits)
        ranking = probs if self.base_bias is None else probs + self.base_bias
        indices = torch.topk(ranking, self.top_k, dim=-1, sorted=False).indices
        weights = probs.gather(-1, indices)
        if self.normalize:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + self.eps)
        weights = weights * self.routed_scaling_factor
        if self.per_expert_scale is not None:
            weights = weights * self.per_expert_scale[indices]
        return weights, indices

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = hidden_states.reshape(-1, self.hidden_dim)
        if self.context.mode == "teacher":
            gate_out = self.base_gate(flat)
            if isinstance(gate_out, tuple):
                logits, weights, indices = gate_out
            else:
                logits = gate_out
                weights, indices = self._select_and_score(logits.float())
                weights = weights.to(hidden_states.dtype)
            if self.context.record_teacher:
                self.context.teacher_routes.setdefault(self.layer_idx, []).append(indices.detach())
            return logits, weights, indices

        prompt_lengths = self.context.prompt_lengths
        assert prompt_lengths is not None
        batch_size = prompt_lengths.shape[0]
        seq_len = flat.shape[0] // batch_size
        h_seq = self._prepare_input(flat).reshape(batch_size, seq_len, self.hidden_dim)

        # Base + adapted token logits across the sequence.
        tok_logits = (
            F.linear(h_seq, self._proj_weight.float())
            + (F.linear(h_seq, self.token_weight) if self.token_weight is not None
               else F.linear(F.linear(h_seq, self.token_a), self.token_b) * self.token_scale)
        )
        tok_probs = self._activate(tok_logits)

        positions = torch.arange(seq_len, device=h_seq.device)[None, :]
        is_resp = positions >= prompt_lengths[:, None]
        rel_pos = (positions - prompt_lengths[:, None]).clamp_min(0)
        win_size = self.context.window_size
        win_idx = rel_pos // win_size
        max_windows = max(1, (seq_len + win_size - 1) // win_size)

        # Anchor position for each window w is prompt_len + w * win_size (clamped inside seq_len).
        anchors = prompt_lengths[:, None] + torch.arange(max_windows, device=h_seq.device)[None, :] * win_size
        anchors_clamped = anchors.clamp_max(seq_len - 1)
        anchor_hidden = h_seq.gather(1, anchors_clamped.unsqueeze(-1).expand(-1, -1, self.hidden_dim))

        # On pass 0 (first call for this layer in this forward), select pinned experts per window.
        sel_logits = self.selector.logits(anchor_hidden)
        if self.layer_idx not in self.context.pinned:
            with torch.no_grad():
                sel_probs = self._activate(sel_logits)
                pinned = torch.topk(sel_probs + self.selector.bias, self.top_k, dim=-1, sorted=False).indices
            self.context.pinned[self.layer_idx] = pinned
            self.context.anchor_hidden[self.layer_idx] = anchor_hidden.detach()
        else:
            pinned = self.context.pinned[self.layer_idx]

        # Broadcast each token's window's pinned indices: (B, L, top_k).
        pinned_tok = pinned.gather(1, win_idx.unsqueeze(-1).expand(-1, -1, self.top_k))
        sel_tok_logits = sel_logits.gather(1, win_idx.unsqueeze(-1).expand(-1, -1, self.num_experts))
        sel_tok_probs = self._activate(sel_tok_logits)

        # Unpinned routing for prompt tokens; pinned routing + straight-through gate for response tokens.
        prompt_ranking = tok_probs if self.base_bias is None else tok_probs + self.base_bias
        prompt_indices = torch.topk(prompt_ranking, self.top_k, dim=-1, sorted=False).indices
        indices = torch.where(is_resp.unsqueeze(-1), pinned_tok, prompt_indices)

        gathered_tok = tok_probs.gather(-1, indices)
        if self.normalize:
            weights = gathered_tok / (gathered_tok.sum(dim=-1, keepdim=True) + self.eps)
        else:
            weights = gathered_tok
        weights = weights * self.routed_scaling_factor

        # Straight-through gate into the window selector probabilities for response tokens.
        sel_gathered = sel_tok_probs.gather(-1, indices)
        st_gate = 1.0 + (sel_gathered - sel_gathered.detach())
        weights = torch.where(is_resp.unsqueeze(-1), weights * st_gate, weights)
        if self.per_expert_scale is not None:
            weights = weights * self.per_expert_scale[indices]

        head_out = tok_probs if self._return_probs else tok_logits
        return (
            head_out.reshape(-1, self.num_experts),
            weights.to(hidden_states.dtype).reshape(-1, self.top_k),
            indices.reshape(-1, self.top_k),
        )


def _linear_gate_moe_forward(self, hidden_states: torch.Tensor):
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    flat = hidden_states.reshape(-1, hidden_dim)
    router_logits, routing_weights, selected_experts = self.gate(flat)
    routing_weights = routing_weights.to(hidden_states.dtype)

    if callable(self.experts) and not isinstance(self.experts, nn.ModuleList):
        final_hidden = self.experts(flat, selected_experts, routing_weights)
    else:
        final_hidden = torch.zeros_like(flat)
        expert_mask = F.one_hot(selected_experts, num_classes=self.gate.num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero(as_tuple=False).flatten()
        for expert_idx_tensor in expert_hit:
            expert_idx = int(expert_idx_tensor.item())
            slot_idx, token_idx = torch.where(expert_mask[expert_idx])
            current_state = flat[token_idx]
            current_hidden = self.experts[expert_idx](current_state) * routing_weights[token_idx, slot_idx, None]
            final_hidden.index_add_(0, token_idx, current_hidden.to(final_hidden.dtype))

    if hasattr(self, "shared_expert") and self.shared_expert is not None:
        shared_out = self.shared_expert(flat)
        if hasattr(self, "shared_expert_gate") and self.shared_expert_gate is not None:
            shared_out = F.sigmoid(self.shared_expert_gate(flat)) * shared_out
        final_hidden = final_hidden + shared_out.to(final_hidden.dtype)

    out = final_hidden.reshape(batch_size, sequence_length, hidden_dim)
    if getattr(self, "_returns_router_logits", False):
        return out, router_logits
    return out


def _find_text_layers(model: nn.Module):
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise ValueError("Cannot locate decoder layers on model")


def _has_router_proj(module: nn.Module | None) -> bool:
    if module is None:
        return False
    if isinstance(module, WindowPinnedRouter):
        return True
    proj = getattr(module, "proj", module)
    return hasattr(proj, "weight")


def _find_moe_router_slot(layer: nn.Module):
    for attr in _MOE_BLOCK_ATTRS:
        block = getattr(layer, attr, None)
        gate = getattr(block, "gate", None) if block is not None else None
        if _has_router_proj(gate):
            return block, "gate", gate
    router = getattr(layer, "router", None)
    if _has_router_proj(router):
        return layer, "router", router
    return None


def _find_moe_block(layer: nn.Module):
    slot = _find_moe_router_slot(layer)
    if slot is None:
        return None
    return slot[0]


def _text_config(model: nn.Module):
    cfg = getattr(model, "config", None)
    if cfg is not None and hasattr(cfg, "get_text_config"):
        return cfg.get_text_config()
    if cfg is not None and hasattr(cfg, "text_config") and cfg.text_config is not None:
        return cfg.text_config
    return cfg


def _moe_router_spec(model: nn.Module, moe_block: nn.Module, gate: nn.Module) -> dict:
    cfg = _text_config(model)
    gate_cfg = getattr(gate, "config", None)
    model_type = str(getattr(cfg, "model_type", getattr(getattr(model, "config", None), "model_type", ""))).lower()
    base_bias = getattr(gate, "e_score_correction_bias", getattr(gate, "expert_bias", None))
    if base_bias is None and moe_block is not gate:
        base_bias = getattr(moe_block, "expert_bias", None)
    use_softmax = any(family in model_type for family in _SOFTMAX_FAMILIES) and base_bias is None
    proj_mod = getattr(gate, "proj", gate)
    w_experts, w_hidden = proj_mod.weight.shape
    top_k = getattr(gate, "top_k", getattr(gate_cfg, "top_k_experts", getattr(moe_block, "top_k", None)))
    if top_k is None:
        top_k = getattr(cfg, "num_experts_per_tok", getattr(cfg, "top_k_experts", None))
    return {
        "top_k": top_k,
        "num_experts": getattr(gate, "num_experts", getattr(moe_block, "num_experts", getattr(cfg, "num_experts", w_experts))),
        "hidden_dim": getattr(gate, "hidden_dim", getattr(gate_cfg, "hidden_size", getattr(cfg, "hidden_size", w_hidden))),
        "base_bias": base_bias,
        "routed_scaling_factor": getattr(
            gate, "routed_scaling_factor",
            getattr(moe_block, "routed_scaling_factor", getattr(cfg, "routed_scaling_factor", 1.0)),
        ),
        "activation": "softmax" if use_softmax else "sigmoid",
        "normalize": getattr(
            gate, "norm_topk_prob",
            getattr(moe_block, "norm_topk_prob", getattr(cfg, "norm_topk_prob", True)),
        ),
        "eps": 1e-6 if "lfm2" in model_type else 1e-20,
        "return_probs": hasattr(gate, "per_expert_scale") or "gemma4" in model_type,
        "per_expert_scale": getattr(gate, "per_expert_scale", None),
    }


def _wrap_container_lora(container: nn.Module | None, targets: tuple[str, ...], *, rank: int, alpha: float):
    if container is None:
        return
    for proj_name in targets:
        module = getattr(container, proj_name, None)
        if isinstance(module, nn.Linear):
            setattr(container, proj_name, LoRALinear(module, rank=rank, alpha=alpha))
        elif module is not None:
            for sub_name in ("linear", "gate_proj"):
                sub = getattr(module, sub_name, None)
                if isinstance(sub, nn.Linear):
                    setattr(module, sub_name, LoRALinear(sub, rank=rank, alpha=alpha))


def _attach_layer_lora(
    layer: nn.Module,
    lora_targets: tuple[str, ...],
    *,
    rank: int,
    alpha: float,
    adapt_resident_mlp: bool = False,
):
    containers = [getattr(layer, attr, None) for attr in _OPERATOR_ATTRS]
    slot = _find_moe_router_slot(layer)
    if slot is None:
        containers.append(getattr(layer, "mlp", getattr(layer, "feed_forward", None)))
    for container in containers:
        _wrap_container_lora(container, lora_targets, rank=rank, alpha=alpha)
    if adapt_resident_mlp and slot is not None:
        moe_block = slot[0]
        if moe_block is layer:
            # Gemma4-style blocks keep a dense parallel MLP alongside layer.router.
            _wrap_container_lora(getattr(layer, "mlp", None), _MLP_LORA_PROJS, rank=rank, alpha=alpha)
        for shared_attr in ("shared_expert", "shared_experts"):
            shared = getattr(moe_block, shared_attr, None)
            if shared is not None:
                _wrap_container_lora(shared, _MLP_LORA_PROJS, rank=rank, alpha=alpha)


def patch_moe_for_window_distill(
    model: nn.Module,
    *,
    window_size: int | None = 16,
    lora_rank: int = 16,
    lora_alpha: float = 32.0,
    selector_rank: int = 32,
    selector_alpha: float = 32.0,
    token_router_rank: int = 16,
    token_scale: float = 2.0,
    lora_targets: tuple[str, ...] = DEFAULT_LORA_TARGETS,
    require_moe: bool = True,
    router_training: str = "adapter",
    adapt_resident_mlp: bool = False,
) -> DistillContext:
    """Freeze base weights and install WindowPinnedRouter + LoRA across Inkling2, Qwen3/3.5-MoE, Gemma4, LFM2-MoE, and dense models."""
    if router_training not in {"adapter", "full"}:
        raise ValueError("router_training must be adapter or full")
    if router_training == "full" and window_size is None:
        raise ValueError("Full-router distillation requires window_size")
    for param in model.parameters():
        param.requires_grad_(False)

    effective_alpha = float(lora_alpha)
    context = DistillContext(window_size=window_size if window_size is not None else 16)
    layers = _find_text_layers(model)
    moe_count = 0
    for layer_idx, layer in enumerate(layers):
        if window_size is not None:
            slot = _find_moe_router_slot(layer)
            if slot is not None:
                moe_block, attr_name, orig_gate = slot
                spec = _moe_router_spec(model, moe_block, orig_gate)
                if spec["top_k"] is None:
                    raise ValueError(f"Cannot infer top_k for MoE layer {layer_idx}")
                pinned_gate = WindowPinnedRouter(
                    orig_gate,
                    layer_idx,
                    context,
                    top_k=spec["top_k"],
                    num_experts=spec["num_experts"],
                    hidden_dim=spec["hidden_dim"],
                    base_bias=spec["base_bias"],
                    routed_scaling_factor=spec["routed_scaling_factor"],
                    activation=spec["activation"],
                    normalize=spec["normalize"],
                    eps=spec["eps"],
                    return_probs=spec["return_probs"],
                    per_expert_scale=spec["per_expert_scale"],
                    selector_rank=selector_rank,
                    selector_alpha=selector_alpha,
                    token_router_rank=token_router_rank,
                    token_scale=token_scale,
                    router_training=router_training,
                )
                setattr(moe_block, attr_name, pinned_gate)
                if isinstance(orig_gate, nn.Linear) and attr_name == "gate" and hasattr(moe_block, "experts"):
                    # Plain nn.Linear gates (e.g., HF 4.57 Qwen3MoeSparseMoeBlock) compute routing in the block.
                    moe_block._returns_router_logits = "Qwen3Moe" in type(moe_block).__name__
                    moe_block.forward = MethodType(_linear_gate_moe_forward, moe_block)
                moe_count += 1
        _attach_layer_lora(
            layer, lora_targets, rank=lora_rank, alpha=effective_alpha,
            adapt_resident_mlp=adapt_resident_mlp,
        )
    if window_size is not None and require_moe and moe_count == 0:
        raise ValueError("Window-pinned distillation requires a model with MoE router layers")
    return context


def patch_inkling_for_window_distill(
    model: nn.Module,
    *,
    window_size: int = 16,
    lora_rank: int = 16,
    lora_alpha: float = 32.0,
    selector_rank: int = 32,
    token_router_rank: int = 16,
    lora_targets: tuple[str, ...] = DEFAULT_LORA_TARGETS,
    router_training: str = "adapter",
) -> DistillContext:
    """Backward-compatible entry point for patching MoE models for window distillation."""
    return patch_moe_for_window_distill(
        model,
        window_size=window_size,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        selector_rank=selector_rank,
        token_router_rank=token_router_rank,
        lora_targets=lora_targets,
        require_moe=True,
        router_training=router_training,
    )


def compute_window_coverage_loss(
    context: DistillContext,
    prompt_lengths: torch.Tensor,
    seq_len: int,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Supervise WindowSelector logits with empirical teacher expert frequency per window."""
    if not context.anchor_hidden or not context.teacher_routes:
        return torch.tensor(0.0, device=prompt_lengths.device)
    batch_size = prompt_lengths.shape[0]
    win_size = context.window_size
    max_windows = max(1, (seq_len + win_size - 1) // win_size)
    positions = torch.arange(seq_len, device=prompt_lengths.device)[None, :]
    valid = positions >= prompt_lengths[:, None]
    if attention_mask is not None:
        valid = valid & attention_mask.bool()
    rel_pos = (positions - prompt_lengths[:, None]).clamp_min(0)
    win_idx = rel_pos // win_size

    losses = []
    for layer_idx, anchor_h in context.anchor_hidden.items():
        passes = context.teacher_routes.get(layer_idx, [])
        if not passes:
            continue
        selector = context.selectors[layer_idx]
        sel_logits = selector.logits(anchor_h) + selector.bias
        num_experts = sel_logits.shape[-1]
        # Accumulate teacher expert hit counts per (batch, window, expert) across passes.
        counts = torch.zeros(batch_size, max_windows, num_experts, device=sel_logits.device, dtype=torch.float32)
        for pass_indices in passes:
            # pass_indices: (B * L, top_k)
            idx_seq = pass_indices.reshape(batch_size, seq_len, -1)
            for k_col in range(idx_seq.shape[-1]):
                valid_mask = valid.float()
                src = valid_mask.unsqueeze(-1)
                flat_idx = idx_seq[..., k_col:k_col + 1]
                one_hot = torch.zeros(batch_size, seq_len, num_experts, device=sel_logits.device)
                one_hot.scatter_add_(2, flat_idx, src)
                for w in range(max_windows):
                    w_mask = (win_idx == w).float().unsqueeze(-1)
                    counts[:, w, :] += (one_hot * w_mask).sum(dim=1)
        win_totals = counts.sum(dim=-1, keepdim=True)
        has_tokens = (win_totals > 0).squeeze(-1)
        if not has_tokens.any():
            continue
        target_dist = counts / win_totals.clamp_min(1.0)
        log_probs = F.log_softmax(sel_logits, dim=-1)
        ce = -(target_dist * log_probs).sum(dim=-1)
        losses.append(ce[has_tokens].mean())
    if not losses:
        return torch.tensor(0.0, device=prompt_lengths.device)
    return torch.stack(losses).mean()


def distill_step(
    model: nn.Module,
    context: DistillContext,
    input_ids: torch.Tensor,
    prompt_lengths: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    *,
    kl_weight: float = 1.0,
    ce_weight: float = 0.1,
    cov_weight: float = 0.25,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Run optional teacher no-grad pass + student pass and return total loss + metrics."""
    batch_size, seq_len = input_ids.shape
    positions = torch.arange(seq_len, device=input_ids.device)[None, :]
    resp_mask = positions[:, 1:] >= prompt_lengths[:, None]
    if attention_mask is not None:
        resp_mask = resp_mask & attention_mask[:, 1:].bool()

    teacher_logits = None
    if kl_weight > 0.0 or cov_weight > 0.0:
        # 1. Teacher pass (unpinned, no LoRA, no grad).
        context.reset_forward("teacher", prompt_lengths, record_teacher=cov_weight > 0.0)
        with torch.no_grad(), _disabled_lora(model):
            teacher_out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            teacher_logits = teacher_out.logits[:, :-1, :].float()

    # Retain this teacher pass across reset_forward, which clears per-forward state.
    teacher_routes = dict(context.teacher_routes) if teacher_logits is not None else {}
    # 2. Student pass (W-token window-pinned when router is active).
    context.reset_forward("student", prompt_lengths, record_teacher=False)
    context.teacher_routes.update(teacher_routes)
    student_out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    student_logits = student_out.logits[:, :-1, :].float()

    mask_f = resp_mask.float()
    denom = mask_f.sum().clamp_min(1.0)
    if kl_weight > 0.0 and teacher_logits is not None:
        t_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)
        s_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        t_probs = t_log_probs.exp()
        kl_loss = ((t_probs * (t_log_probs - s_log_probs)).sum(dim=-1) * (temperature ** 2) * mask_f).sum() / denom
    else:
        kl_loss = student_logits.new_zeros(())

    targets = input_ids[:, 1:]
    ce_per_tok = F.cross_entropy(
        student_logits.reshape(-1, student_logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).reshape(batch_size, seq_len - 1)
    ce_loss = (ce_per_tok * mask_f).sum() / denom

    cov_loss = compute_window_coverage_loss(context, prompt_lengths, seq_len, attention_mask) if cov_weight > 0.0 else student_logits.new_zeros(())
    total = kl_weight * kl_loss + ce_weight * ce_loss + cov_weight * cov_loss

    with torch.no_grad():
        top1_match = (
            ((student_logits.argmax(dim=-1) == teacher_logits.argmax(dim=-1)).float() * mask_f).sum() / denom
            if teacher_logits is not None else student_logits.new_zeros(())
        )

    return total, {
        "loss": float(total.item()),
        "kl": float(kl_loss.item()),
        "ce": float(ce_loss.item()),
        "cov": float(cov_loss.item()),
        "top1_match": float(top1_match.item()),
    }


def export_adapter(model: nn.Module, path: str, *, metadata: dict | None = None):
    """Save trainable WindowPinnedRouter and LoRA tensors (plus optional JSON metadata) to Safetensors."""
    tensors = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            tensors[name] = param.detach().cpu()
    save_file(tensors, path)
    if metadata is not None:
        Path(path).with_suffix(".json").write_text(
            json.dumps(metadata, indent=2, default=str), encoding="utf-8",
        )
    return list(tensors.keys())


def load_adapter(model: nn.Module, path: str):
    """Load saved adapter tensors into a patched model."""
    tensors = load_file(path)
    missing, unexpected = model.load_state_dict(tensors, strict=False)
    if unexpected:
        raise ValueError(f"Unexpected adapter keys: {unexpected[:5]}")
    return len(tensors), missing
