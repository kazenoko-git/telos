"""Unit tests for Inkling 2 window-pinned router & LoRA distillation."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from telos.training.inkling2_window_distill import (
    DistillContext,
    WindowPinnedRouter,
    distill_step,
    export_adapter,
    load_adapter,
    patch_inkling_for_window_distill,
)


class _ToyRouter(nn.Module):
    def __init__(self, hidden_dim=16, num_experts=6, top_k=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.routed_scaling_factor = 4.0
        self.weight = nn.Parameter(torch.randn(num_experts, hidden_dim) * 0.2)
        self.register_buffer("e_score_correction_bias", torch.zeros(num_experts))

    def forward(self, hidden_states):
        flat = hidden_states.float().reshape(-1, self.hidden_dim)
        logits = F.linear(flat, self.weight)
        scores = torch.sigmoid(logits)
        indices = torch.topk(scores + self.e_score_correction_bias, self.top_k, dim=-1, sorted=False).indices
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20) * self.routed_scaling_factor
        return logits, weights, indices


class _ToyMoE(nn.Module):
    def __init__(self, hidden_dim=16, inner_dim=24, num_experts=6, top_k=2):
        super().__init__()
        self.gate = _ToyRouter(hidden_dim, num_experts, top_k)
        self.gate_up = nn.Parameter(torch.randn(num_experts, 2 * inner_dim, hidden_dim) * 0.15)
        self.down = nn.Parameter(torch.randn(num_experts, hidden_dim, inner_dim) * 0.15)

    def forward(self, hidden_states):
        b, seq, d = hidden_states.shape
        flat = hidden_states.reshape(-1, d)
        _, weights, indices = self.gate(flat)
        out = torch.zeros_like(flat)
        for k in range(indices.shape[-1]):
            idx = indices[:, k]
            gu = torch.bmm(flat.unsqueeze(1), self.gate_up[idx].transpose(-1, -2)).squeeze(1)
            g, u = gu.chunk(2, dim=-1)
            act = F.silu(g) * u
            dn = torch.bmm(act.unsqueeze(1), self.down[idx].transpose(-1, -2)).squeeze(1)
            out = out + dn * weights[:, k:k + 1]
        return out.reshape(b, seq, d)


class _ToyLayer(nn.Module):
    def __init__(self, hidden_dim=16, num_experts=6, top_k=2):
        super().__init__()
        self.self_attn = nn.Module()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"):
            setattr(self.self_attn, name, nn.Linear(hidden_dim, hidden_dim, bias=False))
        self.mlp = _ToyMoE(hidden_dim, 24, num_experts, top_k)

    def forward(self, x, use_checkpoint=False):
        def _inner(h):
            attn = self.self_attn.o_proj(
                self.self_attn.q_proj(h) * torch.sigmoid(self.self_attn.gate_proj(h))
                + self.self_attn.v_proj(h) + 0.1 * self.self_attn.k_proj(h)
            )
            h = h + 0.1 * attn
            return h + self.mlp(h)

        if use_checkpoint and x.requires_grad:
            return torch.utils.checkpoint.checkpoint(_inner, x, use_reentrant=False)
        return _inner(x)


class _ToyInklingModel(nn.Module):
    def __init__(self, vocab_size=32, hidden_dim=16, num_layers=3, loop_passes=3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        layers = [_ToyLayer(hidden_dim) for _ in range(num_layers)]
        self.model = SimpleNamespace(language_model=SimpleNamespace(layers=nn.ModuleList(layers)))
        self.layers = self.model.language_model.layers
        self.loop_passes = loop_passes
        self.head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, use_cache=False, use_checkpoint=False):
        h = self.embed(input_ids)
        h = self.layers[0](h, use_checkpoint=use_checkpoint)
        for _ in range(self.loop_passes):
            h = self.layers[1](h, use_checkpoint=use_checkpoint)
        h = self.layers[2](h, use_checkpoint=use_checkpoint)
        return SimpleNamespace(logits=self.head(h))


def test_window_pinned_distill_gradients_checkpointing_and_export(tmp_path):
    torch.manual_seed(7)
    model = _ToyInklingModel()
    ctx = patch_inkling_for_window_distill(model, window_size=4, lora_rank=4, selector_rank=4, token_router_rank=4)

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    frozen_experts = [p for n, p in model.named_parameters() if "gate_up" in n or "down" in n]
    assert trainable
    assert all(not p.requires_grad for p in frozen_experts)
    assert any("selector.selector_b" in n for n in trainable)
    assert any("self_attn.o_proj.lora_b" in n for n in trainable)

    input_ids = torch.randint(0, 32, (2, 12))
    prompt_lengths = torch.tensor([4, 5])

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-2)
    initial_loss = None
    for step in range(8):
        optimizer.zero_grad()
        loss, metrics = distill_step(model, ctx, input_ids, prompt_lengths)
        if initial_loss is None:
            initial_loss = metrics["loss"]
        loss.backward()
        if step == 0:
            # Verify gradients flow to both WindowSelector and LoRA adapters across loop passes.
            assert model.layers[1].mlp.gate.selector.selector_b.grad is not None
            assert model.layers[1].mlp.gate.selector.selector_b.grad.abs().sum() > 0
            assert model.layers[1].self_attn.o_proj.lora_b.grad is not None
            assert model.layers[1].self_attn.o_proj.lora_b.grad.abs().sum() > 0
        optimizer.step()

    assert metrics["loss"] < initial_loss

    # Verify gradient checkpointing across looped layers works cleanly.
    orig_forward = model.forward
    model.forward = lambda input_ids, attention_mask=None, use_cache=False: orig_forward(
        input_ids, attention_mask=attention_mask, use_cache=use_cache, use_checkpoint=True,
    )
    optimizer.zero_grad()
    ckpt_loss, _ = distill_step(model, ctx, input_ids, prompt_lengths)
    ckpt_loss.backward()
    assert model.layers[1].mlp.gate.selector.selector_b.grad.abs().sum() > 0
    assert model.layers[1].self_attn.o_proj.lora_b.grad.abs().sum() > 0
    model.forward = orig_forward

    # Verify adapter export and reload round-trip.
    adapter_path = tmp_path / "adapter.safetensors"
    keys = export_adapter(model, str(adapter_path))
    assert len(keys) == len(trainable)

    fresh = _ToyInklingModel()
    patch_inkling_for_window_distill(fresh, window_size=4, lora_rank=4, selector_rank=4, token_router_rank=4)
    loaded_count, _ = load_adapter(fresh, str(adapter_path))
    assert loaded_count == len(keys)
    for n, p in model.named_parameters():
        if p.requires_grad:
            assert torch.equal(p, dict(fresh.named_parameters())[n])
