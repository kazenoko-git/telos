"""PyTorch window-pinned routing and LoRA distillation for Inkling 2 Mini."""

import math
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        delta = F.linear(F.linear(x.float(), self.lora_a), self.lora_b) * self.scale
        return out + delta.to(out.dtype)


class WindowSelector(nn.Module):
    """Selects top-k experts per window using base router + low-rank delta and learned bias."""

    def __init__(self, base_weight: torch.Tensor, base_bias: torch.Tensor, rank: int = 32, alpha: float = 32.0):
        super().__init__()
        num_experts, hidden_dim = base_weight.shape
        device = base_weight.device
        self.register_buffer("base_weight", base_weight.detach().float().clone())
        self.selector_a = nn.Parameter(torch.empty(rank, hidden_dim, dtype=torch.float32, device=device))
        self.selector_b = nn.Parameter(torch.zeros(num_experts, rank, dtype=torch.float32, device=device))
        nn.init.kaiming_uniform_(self.selector_a, a=math.sqrt(5))
        self.scale = alpha / rank
        self.bias = nn.Parameter(base_bias.detach().float().clone())

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        base = F.linear(hidden.float(), self.base_weight)
        delta = F.linear(F.linear(hidden.float(), self.selector_a), self.selector_b) * self.scale
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
    """Wraps an Inkling TopK router for teacher or W-token window-pinned student passes."""

    def __init__(self, base_gate: nn.Module, layer_idx: int, context: DistillContext,
                 *, selector_rank: int = 32, token_router_rank: int = 16):
        super().__init__()
        self.base_gate = base_gate
        for param in self.base_gate.parameters():
            param.requires_grad_(False)
        self.layer_idx = layer_idx
        self.context = context
        self.top_k = base_gate.top_k
        self.num_experts = base_gate.num_experts
        self.hidden_dim = base_gate.hidden_dim
        self.routed_scaling_factor = base_gate.routed_scaling_factor

        bias = getattr(base_gate, "e_score_correction_bias",
                       torch.zeros(self.num_experts, dtype=torch.float32, device=base_gate.weight.device))
        self.selector = WindowSelector(base_gate.weight, bias, rank=selector_rank)
        context.selectors[layer_idx] = self.selector
        self.token_a = nn.Parameter(torch.empty(token_router_rank, self.hidden_dim, dtype=torch.float32,
                                                device=base_gate.weight.device))
        self.token_b = nn.Parameter(torch.zeros(self.num_experts, token_router_rank, dtype=torch.float32,
                                                device=base_gate.weight.device))
        nn.init.kaiming_uniform_(self.token_a, a=math.sqrt(5))
        self.token_scale = 2.0

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.context.mode == "teacher":
            logits, weights, indices = self.base_gate(hidden_states)
            if self.context.record_teacher:
                self.context.teacher_routes.setdefault(self.layer_idx, []).append(indices.detach())
            return logits, weights, indices

        prompt_lengths = self.context.prompt_lengths
        assert prompt_lengths is not None
        batch_size = prompt_lengths.shape[0]
        seq_len = hidden_states.shape[0] // batch_size
        h_seq = hidden_states.float().reshape(batch_size, seq_len, self.hidden_dim)

        # Base + adapted token logits across the sequence.
        tok_logits = (
            F.linear(h_seq, self.base_gate.weight)
            + F.linear(F.linear(h_seq, self.token_a), self.token_b) * self.token_scale
        )
        tok_scores = torch.sigmoid(tok_logits)

        positions = torch.arange(seq_len, device=h_seq.device)[None, :]
        is_resp = positions >= prompt_lengths[:, None]
        rel_pos = (positions - prompt_lengths[:, None]).clamp_min(0)
        win_size = self.context.window_size
        win_idx = rel_pos // win_size
        max_windows = max(1, (seq_len + win_size - 1) // win_size)

        # Anchor position for each window w is prompt_len + w * win_size (clamped inside seq_len).
        anchors = (prompt_lengths[:, None] + torch.arange(max_windows, device=h_seq.device)[None, :] * win_size)
        anchors_clamped = anchors.clamp_max(seq_len - 1)
        anchor_hidden = h_seq.gather(1, anchors_clamped.unsqueeze(-1).expand(-1, -1, self.hidden_dim))

        # On pass 0 (first call for this layer in this forward), select pinned experts per window.
        sel_logits = self.selector.logits(anchor_hidden)
        if self.layer_idx not in self.context.pinned:
            with torch.no_grad():
                sel_scores = torch.sigmoid(sel_logits) + self.selector.bias
                pinned = torch.topk(sel_scores, self.top_k, dim=-1, sorted=False).indices
            self.context.pinned[self.layer_idx] = pinned
            self.context.anchor_hidden[self.layer_idx] = anchor_hidden.detach()
        else:
            pinned = self.context.pinned[self.layer_idx]

        # Broadcast each token's window's pinned indices: (B, L, top_k).
        pinned_tok = pinned.gather(1, win_idx.unsqueeze(-1).expand(-1, -1, self.top_k))
        sel_tok_logits = sel_logits.gather(1, win_idx.unsqueeze(-1).expand(-1, -1, self.num_experts))
        sel_tok_probs = torch.sigmoid(sel_tok_logits)

        # Unpinned routing for prompt tokens; pinned routing + straight-through gate for response tokens.
        bias = getattr(self.base_gate, "e_score_correction_bias", 0.0)
        prompt_indices = torch.topk(tok_scores + bias, self.top_k, dim=-1, sorted=False).indices
        indices = torch.where(is_resp.unsqueeze(-1), pinned_tok, prompt_indices)

        gathered_tok = tok_scores.gather(-1, indices)
        weights = gathered_tok / (gathered_tok.sum(dim=-1, keepdim=True) + 1e-20) * self.routed_scaling_factor

        # Straight-through gate into the window selector probabilities for response tokens.
        sel_gathered = sel_tok_probs.gather(-1, indices)
        st_gate = 1.0 + (sel_gathered - sel_gathered.detach())
        weights = torch.where(is_resp.unsqueeze(-1), weights * st_gate, weights)

        return tok_logits.reshape(-1, self.num_experts), weights.reshape(-1, self.top_k), indices.reshape(-1, self.top_k)


def _find_text_layers(model: nn.Module):
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise ValueError("Cannot locate decoder layers on Inkling model")


def patch_inkling_for_window_distill(
    model: nn.Module,
    *,
    window_size: int = 16,
    lora_rank: int = 16,
    selector_rank: int = 32,
    token_router_rank: int = 16,
    lora_targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"),
) -> DistillContext:
    """Freeze base weights and install WindowPinnedRouter + attention LoRA adapters."""
    for param in model.parameters():
        param.requires_grad_(False)

    context = DistillContext(window_size=window_size)
    layers = _find_text_layers(model)
    for layer_idx, layer in enumerate(layers):
        layer.mlp.gate = WindowPinnedRouter(
            layer.mlp.gate, layer_idx, context,
            selector_rank=selector_rank, token_router_rank=token_router_rank,
        )
        attn = getattr(layer, "self_attn", None)
        if attn is not None:
            for proj_name in lora_targets:
                module = getattr(attn, proj_name, None)
                if isinstance(module, nn.Linear):
                    setattr(attn, proj_name, LoRALinear(module, rank=lora_rank))
                elif module is not None:
                    for sub_name in ("linear", "gate_proj"):
                        sub = getattr(module, sub_name, None)
                        if isinstance(sub, nn.Linear):
                            setattr(module, sub_name, LoRALinear(sub, rank=lora_rank))
    return context


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
    ce_weight: float = 0.1,
    cov_weight: float = 0.25,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Run teacher no-grad pass + student W-token pinned pass and return total loss + metrics."""
    batch_size, seq_len = input_ids.shape
    positions = torch.arange(seq_len, device=input_ids.device)[None, :]
    resp_mask = positions[:, 1:] >= prompt_lengths[:, None]
    if attention_mask is not None:
        resp_mask = resp_mask & attention_mask[:, 1:].bool()

    # 1. Teacher pass (unpinned, no grad).
    context.reset_forward("teacher", prompt_lengths, record_teacher=True)
    with torch.no_grad():
        teacher_out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        teacher_logits = teacher_out.logits[:, :-1, :].float()

    # 2. Student pass (W-token window-pinned).
    context.reset_forward("student", prompt_lengths, record_teacher=False)
    student_out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    student_logits = student_out.logits[:, :-1, :].float()

    # Masked KL divergence over response positions.
    t_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)
    s_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    t_probs = t_log_probs.exp()
    kl_per_tok = (t_probs * (t_log_probs - s_log_probs)).sum(dim=-1) * (temperature ** 2)
    mask_f = resp_mask.float()
    denom = mask_f.sum().clamp_min(1.0)
    kl_loss = (kl_per_tok * mask_f).sum() / denom

    # Optional hard next-token cross-entropy on response tokens.
    targets = input_ids[:, 1:]
    ce_per_tok = F.cross_entropy(
        student_logits.reshape(-1, student_logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).reshape(batch_size, seq_len - 1)
    ce_loss = (ce_per_tok * mask_f).sum() / denom

    cov_loss = compute_window_coverage_loss(context, prompt_lengths, seq_len, attention_mask)
    total = kl_loss + ce_weight * ce_loss + cov_weight * cov_loss

    with torch.no_grad():
        top1_match = ((student_logits.argmax(dim=-1) == teacher_logits.argmax(dim=-1)).float() * mask_f).sum() / denom

    return total, {
        "loss": float(total.item()),
        "kl": float(kl_loss.item()),
        "ce": float(ce_loss.item()),
        "cov": float(cov_loss.item()),
        "top1_match": float(top1_match.item()),
    }


def export_adapter(model: nn.Module, path: str):
    """Save only trainable WindowPinnedRouter and LoRA tensors to Safetensors."""
    tensors = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            tensors[name] = param.detach().cpu()
    save_file(tensors, path)
    return list(tensors.keys())


def load_adapter(model: nn.Module, path: str):
    """Load saved adapter tensors into a patched model."""
    tensors = load_file(path)
    missing, unexpected = model.load_state_dict(tensors, strict=False)
    if unexpected:
        raise ValueError(f"Unexpected adapter keys: {unexpected[:5]}")
    return len(tensors), missing
