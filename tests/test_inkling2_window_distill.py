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
    patch_moe_for_window_distill,
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


@pytest.mark.parametrize("router_training", ["adapter", "full"])
def test_window_pinned_distill_gradients_checkpointing_and_export(tmp_path, router_training):
    torch.manual_seed(7)
    model = _ToyInklingModel()
    ctx = patch_inkling_for_window_distill(model, window_size=4, lora_rank=4, selector_rank=4, token_router_rank=4, router_training=router_training)

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    frozen_experts = [p for n, p in model.named_parameters() if "gate_up" in n or "down" in n]
    assert trainable
    assert all(not p.requires_grad for p in frozen_experts)
    assert any(("selector.selector_weight" if router_training == "full" else "selector.selector_b") in n for n in trainable)
    assert any("self_attn.o_proj.lora_b" in n for n in trainable)

    input_ids = torch.randint(0, 32, (2, 12))
    prompt_lengths = torch.tensor([4, 5])

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-4)
    initial_loss = None
    for step in range(8):
        optimizer.zero_grad()
        loss, metrics = distill_step(model, ctx, input_ids, prompt_lengths)
        if initial_loss is None:
            initial_loss = metrics["loss"]
        assert metrics["cov"] > 0
        loss.backward()
        if step == 0:
            # Verify gradients flow to both WindowSelector and LoRA adapters across loop passes.
            assert getattr(model.layers[1].mlp.gate.selector, "selector_weight" if router_training == "full" else "selector_b").grad is not None
            assert getattr(model.layers[1].mlp.gate.selector, "selector_weight" if router_training == "full" else "selector_b").grad.abs().sum() > 0
            assert model.layers[1].self_attn.o_proj.lora_b.grad is not None
            assert model.layers[1].self_attn.o_proj.lora_b.grad.abs().sum() > 0
        optimizer.step()

    assert metrics["loss"] < initial_loss
    if router_training == "full":
        for router in (m for m in model.modules() if isinstance(m, WindowPinnedRouter)):
            assert router.token_weight.grad is not None
            assert torch.isfinite(router.token_weight.grad).all()
            assert router.token_weight.grad.abs().sum() > 0
            assert all(not p.requires_grad and p.grad is None for p in router.base_gate.parameters())

    # Verify gradient checkpointing across looped layers works cleanly.
    orig_forward = model.forward
    model.forward = lambda input_ids, attention_mask=None, use_cache=False: orig_forward(
        input_ids, attention_mask=attention_mask, use_cache=use_cache, use_checkpoint=True,
    )
    optimizer.zero_grad()
    ckpt_loss, _ = distill_step(model, ctx, input_ids, prompt_lengths)
    ckpt_loss.backward()
    assert getattr(model.layers[1].mlp.gate.selector, "selector_weight" if router_training == "full" else "selector_b").grad.abs().sum() > 0
    assert model.layers[1].self_attn.o_proj.lora_b.grad.abs().sum() > 0
    model.forward = orig_forward

    # Verify adapter export and reload round-trip.
    adapter_path = tmp_path / "adapter.safetensors"
    keys = export_adapter(model, str(adapter_path))
    assert len(keys) == len(trainable)

    fresh = _ToyInklingModel()
    patch_inkling_for_window_distill(fresh, window_size=4, lora_rank=4, selector_rank=4, token_router_rank=4, router_training=router_training)
    loaded_count, _ = load_adapter(fresh, str(adapter_path))
    assert loaded_count == len(keys)
    for n, p in model.named_parameters():
        if p.requires_grad:
            assert torch.equal(p, dict(fresh.named_parameters())[n])


def test_qwen3_moe_hf_window_distill(tmp_path):
    pytest.importorskip("transformers")
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    torch.manual_seed(11)
    cfg = Qwen3MoeConfig(
        vocab_size=32, hidden_size=16, intermediate_size=24, moe_intermediate_size=16,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        num_experts=4, num_experts_per_tok=2, norm_topk_prob=True,
    )
    model = Qwen3MoeForCausalLM(cfg)
    ctx = patch_moe_for_window_distill(model, window_size=4, lora_rank=4, selector_rank=4, token_router_rank=4)

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert any("mlp.gate.selector.selector_b" in n for n in trainable)
    assert any("self_attn.o_proj.lora_b" in n for n in trainable)
    assert model.model.layers[0].mlp.gate.activation == "softmax"

    input_ids = torch.randint(0, 32, (2, 10))
    attn_mask = torch.ones_like(input_ids)
    prompt_lengths = torch.tensor([3, 4])
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-2)
    first_loss = None
    for step in range(4):
        opt.zero_grad()
        loss, metrics = distill_step(model, ctx, input_ids, prompt_lengths, attn_mask)
        if first_loss is None:
            first_loss = metrics["loss"]
        loss.backward()
        if step == 0:
            assert model.model.layers[0].mlp.gate.selector.selector_b.grad.abs().sum() > 0
            assert model.model.layers[0].self_attn.q_proj.lora_b.grad.abs().sum() > 0
        opt.step()
    assert metrics["loss"] <= first_loss

    adapter_path = tmp_path / "qwen3_moe_w4.safetensors"
    export_adapter(model, str(adapter_path), metadata={"model_type": "qwen3_moe", "lora_alpha": 8.0, "window": {"window_size": 4}})
    assert adapter_path.with_suffix(".json").is_file()


class _ToyLfm2Layer(nn.Module):
    def __init__(self, hidden_dim=16, *, kind="conv", moe=True):
        super().__init__()
        if kind == "conv":
            self.conv = nn.Module()
            self.conv.in_proj = nn.Linear(hidden_dim, hidden_dim * 2, bias=False)
            self.conv.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        else:
            self.self_attn = nn.Module()
            for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
                setattr(self.self_attn, name, nn.Linear(hidden_dim, hidden_dim, bias=False))
        self.feed_forward = nn.Module()
        if moe:
            self.feed_forward.gate = _ToyRouter(hidden_dim, num_experts=4, top_k=2)
            self.feed_forward.expert_bias = nn.Parameter(torch.zeros(4), requires_grad=False)
            self.feed_forward.experts = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(4)])
        else:
            self.feed_forward.gate_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x):
        if hasattr(self, "conv"):
            h = x + 0.1 * self.conv.out_proj(torch.relu(self.conv.in_proj(x)[..., : x.shape[-1]]))
        else:
            h = x + 0.1 * self.self_attn.out_proj(self.self_attn.q_proj(x) + self.self_attn.k_proj(x) + self.self_attn.v_proj(x))
        if hasattr(self.feed_forward, "gate"):
            b, s, d = h.shape
            _, weights, indices = self.feed_forward.gate(h)
            out = torch.zeros(b * s, d, dtype=h.dtype, device=h.device)
            flat = h.reshape(-1, d)
            for k in range(indices.shape[-1]):
                for e_idx in range(4):
                    mask = indices[:, k] == e_idx
                    if mask.any():
                        out[mask] = out[mask] + self.feed_forward.experts[e_idx](flat[mask]) * weights[mask, k : k + 1]
            return h + out.reshape(b, s, d)
        return h + self.feed_forward.gate_proj(h)


def test_lfm2_moe_hybrid_window_distill():
    torch.manual_seed(13)
    model = nn.Module()
    model.config = SimpleNamespace(model_type="lfm2_moe", hidden_size=16, num_experts=4, num_experts_per_tok=2)
    model.embed = nn.Embedding(32, 16)
    model.model = SimpleNamespace(layers=nn.ModuleList([
        _ToyLfm2Layer(16, kind="conv", moe=False),
        _ToyLfm2Layer(16, kind="full_attention", moe=True),
        _ToyLfm2Layer(16, kind="conv", moe=True),
    ]))
    model.layers = model.model.layers
    model.head = nn.Linear(16, 32, bias=False)
    model.forward = lambda input_ids, attention_mask=None, use_cache=False: SimpleNamespace(
        logits=model.head(model.model.layers[2](model.model.layers[1](model.model.layers[0](model.embed(input_ids)))))
    )

    ctx = patch_moe_for_window_distill(model, window_size=4, lora_rank=4, selector_rank=4, token_router_rank=4)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert any("layers.1.feed_forward.gate.selector.selector_b" in n for n in trainable)
    assert any("layers.2.conv.in_proj.lora_b" in n for n in trainable)
    assert any("layers.1.self_attn.out_proj.lora_b" in n for n in trainable)
    assert not hasattr(model.model.layers[0].feed_forward, "selector")

    input_ids = torch.randint(0, 32, (2, 8))
    prompt_lengths = torch.tensor([3, 4])
    loss, _ = distill_step(model, ctx, input_ids, prompt_lengths)
    loss.backward()
    assert model.model.layers[1].feed_forward.gate.selector.selector_b.grad.abs().sum() > 0
    assert model.model.layers[2].conv.out_proj.lora_b.grad.abs().sum() > 0


def test_external_torch_posttrain_dense_and_rejects_mlx_quantized(tmp_path):
    import json
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from telos.posttrain.trainer import posttrain

    ckpt = tmp_path / "hf_qwen3"
    ckpt.mkdir()
    cfg = Qwen3Config(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
    )
    hf_model = Qwen3ForCausalLM(cfg)
    hf_model.save_pretrained(ckpt, safe_serialization=True)
    tok = Tokenizer(models.WordLevel({"[UNK]": 0, "<|im_end|>": 1, **{f"t{i}": i for i in range(2, 20)}}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.save(str(ckpt / "tokenizer.json"))
    (ckpt / "generation_config.json").write_text(json.dumps({"eos_token_id": 1, "pad_token_id": 0}))

    data = tmp_path / "data.jsonl"
    data.write_text(json.dumps({"prompt": "t2 t3", "response": "t4 t5"}) + "\n")
    out = posttrain(
        ckpt, data, tmp_path / "out_qwen3",
        method="distill", hardware="cpu", finetune="lora",
        lora_rank=4, lora_alpha=8.0, max_steps=2, batch_size=1, grad_accum=2, max_length=16,
    )
    assert out.is_file()
    assert out.with_suffix(".json").is_file()

    cfg_path = ckpt / "config.json"
    bad_cfg = json.loads(cfg_path.read_text())
    bad_cfg["quantization"] = {"bits": 4, "group_size": 64}
    cfg_path.write_text(json.dumps(bad_cfg))
    with pytest.raises(ValueError, match="unpacked HuggingFace checkpoint"):
        posttrain(ckpt, data, tmp_path / "out_bad", method="sft", hardware="cpu", finetune="lora", max_steps=1)


class _ToyQwen35SparseBlock(nn.Module):
    def __init__(self, hidden_dim=16, num_experts=4, top_k=2):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self.top_k = top_k
        self.num_experts = num_experts
        self.norm_topk_prob = True
        self.experts = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_experts)])
        self.shared_expert = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.shared_expert_gate = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden_states):
        raise AssertionError("Should be replaced by _linear_gate_moe_forward")


@pytest.mark.parametrize("router_training", ["adapter", "full"])
def test_qwen3_5_moe_linear_attn_and_shared_expert_window_distill(tmp_path, router_training):
    torch.manual_seed(17)
    model = nn.Module()
    model.config = SimpleNamespace(
        model_type="qwen3_5_moe",
        text_config=SimpleNamespace(model_type="qwen3_5_moe_text", hidden_size=16, num_experts=4, num_experts_per_tok=2),
    )
    model.embed = nn.Embedding(32, 16)

    lin_layer = nn.Module()
    lin_layer.linear_attn = nn.Module()
    lin_layer.linear_attn.in_proj_qkv = nn.Linear(16, 16, bias=False)
    lin_layer.linear_attn.out_proj = nn.Linear(16, 16, bias=False)
    lin_layer.mlp = _ToyQwen35SparseBlock(16, num_experts=4, top_k=2)

    attn_layer = nn.Module()
    attn_layer.self_attn = nn.Module()
    attn_layer.self_attn.q_proj = nn.Linear(16, 16, bias=False)
    attn_layer.self_attn.o_proj = nn.Linear(16, 16, bias=False)
    attn_layer.mlp = _ToyQwen35SparseBlock(16, num_experts=4, top_k=2)

    def _run_model(input_ids, attention_mask=None, use_cache=False):
        h = model.embed(input_ids)
        h = h + lin_layer.linear_attn.out_proj(lin_layer.linear_attn.in_proj_qkv(h))
        h = h + lin_layer.mlp(h)
        h = h + attn_layer.self_attn.o_proj(attn_layer.self_attn.q_proj(h))
        h = h + attn_layer.mlp(h)
        return SimpleNamespace(logits=model.head(h))

    model.model = SimpleNamespace(language_model=SimpleNamespace(layers=nn.ModuleList([lin_layer, attn_layer])))
    model.layers = model.model.language_model.layers
    model.head = nn.Linear(16, 32, bias=False)
    model.forward = _run_model

    ctx = patch_moe_for_window_distill(model, window_size=4, lora_rank=4, selector_rank=4, token_router_rank=4, router_training=router_training)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert any("layers.0.linear_attn.in_proj_qkv.lora_b" in n for n in trainable)
    assert any("layers.1.self_attn.o_proj.lora_b" in n for n in trainable)
    assert any(("selector.selector_weight" if router_training == "full" else "selector.selector_b") in n for n in trainable)
    assert not lin_layer.mlp.shared_expert.weight.requires_grad

    input_ids = torch.randint(0, 32, (2, 10))
    prompt_lengths = torch.tensor([3, 4])
    loss, _ = distill_step(model, ctx, input_ids, prompt_lengths)
    loss.backward()
    assert getattr(lin_layer.mlp.gate.selector, "selector_weight" if router_training == "full" else "selector_b").grad.abs().sum() > 0
    assert lin_layer.linear_attn.in_proj_qkv.lora_b.grad.abs().sum() > 0



    if router_training == "full":
        routers = [m for m in model.modules() if isinstance(m, WindowPinnedRouter)]
        for router in routers:
            assert router.token_weight.grad is not None
            assert torch.isfinite(router.token_weight.grad).all()
            assert router.token_weight.grad.abs().sum() > 0
            assert all(not p.requires_grad and p.grad is None for p in router.base_gate.parameters())
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        optimizer.step()
        path = tmp_path / "full.safetensors"
        keys = export_adapter(model, str(path))
        assert any(k.endswith("selector_weight") for k in keys)
        assert any(k.endswith("token_weight") for k in keys)
        expected = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in expected:
                    p.zero_()
        load_adapter(model, str(path))
        for n, p in model.named_parameters():
            if n in expected:
                assert torch.equal(p, expected[n])

class _ToyGemma4Router(nn.Module):
    def __init__(self, hidden_dim=16, num_experts=6, top_k=2):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_dim, top_k_experts=top_k)
        self.scalar_root_size = hidden_dim ** -0.5
        self.norm = nn.RMSNorm(hidden_dim, eps=1e-6, elementwise_affine=False)
        self.proj = nn.Linear(hidden_dim, num_experts, bias=False)
        self.scale = nn.Parameter(torch.full((hidden_dim,), 1.25))
        self.per_expert_scale = nn.Parameter(torch.linspace(0.8, 1.3, num_experts))

    def forward(self, hidden_states):
        h = self.norm(hidden_states) * self.scale * self.scalar_root_size
        probs = F.softmax(self.proj(h), dim=-1, dtype=torch.float32)
        weights, indices = torch.topk(probs, k=self.config.top_k_experts, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights.to(hidden_states.dtype) * self.per_expert_scale[indices]
        return probs, weights, indices


@pytest.mark.parametrize("router_training", ["adapter", "full"])
def test_gemma4_moe_router_and_exact_unpinned_equivalence(tmp_path, router_training):
    torch.manual_seed(19)
    model = nn.Module()
    model.config = SimpleNamespace(
        model_type="gemma4",
        text_config=SimpleNamespace(model_type="gemma4_text", hidden_size=16, num_experts=6, top_k_experts=2, enable_moe_block=True),
    )
    model.embed = nn.Embedding(32, 16)

    layer = nn.Module()
    layer.self_attn = nn.Module()
    layer.self_attn.q_proj = nn.Linear(16, 16, bias=False)
    layer.self_attn.o_proj = nn.Linear(16, 16, bias=False)
    layer.mlp = nn.Module()
    layer.mlp.gate_proj = nn.Linear(16, 16, bias=False)
    layer.router = _ToyGemma4Router(16, num_experts=6, top_k=2)
    layer.experts = nn.ModuleList([nn.Linear(16, 16, bias=False) for _ in range(6)])

    def _run_layer(h):
        h = h + 0.1 * layer.self_attn.o_proj(layer.self_attn.q_proj(h))
        dense_out = layer.mlp.gate_proj(h)
        _, weights, indices = layer.router(h.reshape(-1, 16))
        flat = h.reshape(-1, 16)
        moe_out = torch.zeros_like(flat)
        for k in range(indices.shape[-1]):
            for e_idx in range(6):
                mask = indices[:, k] == e_idx
                if mask.any():
                    moe_out[mask] = moe_out[mask] + layer.experts[e_idx](flat[mask]) * weights[mask, k : k + 1]
        return h + dense_out + moe_out.reshape_as(h)

    model.model = SimpleNamespace(language_model=SimpleNamespace(layers=nn.ModuleList([layer])))
    model.layers = model.model.language_model.layers
    model.head = nn.Linear(16, 32, bias=False)
    model.forward = lambda input_ids, attention_mask=None, use_cache=False: SimpleNamespace(
        logits=model.head(_run_layer(model.embed(input_ids)))
    )

    # At window_size=1 before any updates, student matches Gemma4TextRouter identically (KL == 0).
    ctx = patch_moe_for_window_distill(model, window_size=1, lora_rank=4, selector_rank=4, token_router_rank=4, router_training=router_training)
    input_ids = torch.randint(0, 32, (2, 8))
    prompt_lengths = torch.tensor([3, 4])
    loss0, metrics0 = distill_step(model, ctx, input_ids, prompt_lengths)
    assert metrics0["kl"] < 1e-6
    assert metrics0["top1_match"] == 1.0

    # At W=4, gradients flow cleanly through WindowSelector, token adapter, and self_attn LoRA while mlp stays frozen.
    ctx.window_size = 4
    loss, _ = distill_step(model, ctx, input_ids, prompt_lengths)
    loss.backward()
    assert getattr(layer.router.selector, "selector_weight" if router_training == "full" else "selector_b").grad.abs().sum() > 0
    assert getattr(layer.router, "token_weight" if router_training == "full" else "token_b").grad.abs().sum() > 0
    assert layer.self_attn.q_proj.lora_b.grad.abs().sum() > 0
    assert not layer.mlp.gate_proj.weight.requires_grad


    if router_training == "full":
        routers = [m for m in model.modules() if isinstance(m, WindowPinnedRouter)]
        for router in routers:
            assert router.token_weight.grad is not None
            assert torch.isfinite(router.token_weight.grad).all()
            assert router.token_weight.grad.abs().sum() > 0
            assert all(not p.requires_grad and p.grad is None for p in router.base_gate.parameters())
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        optimizer.step()
        path = tmp_path / "full.safetensors"
        keys = export_adapter(model, str(path))
        assert any(k.endswith("selector_weight") for k in keys)
        assert any(k.endswith("token_weight") for k in keys)
        expected = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in expected:
                    p.zero_()
        load_adapter(model, str(path))
        for n, p in model.named_parameters():
            if n in expected:
                assert torch.equal(p, expected[n])


def test_full_router_rejects_missing_window():
    with pytest.raises(ValueError, match="window_size"):
        patch_moe_for_window_distill(_ToyInklingModel(), window_size=None, router_training="full")


def test_adapt_resident_mlp_wraps_shared_expert_and_dense_mlp(tmp_path):
    torch.manual_seed(23)
    # 1. Qwen3.5-style shared_expert block.
    q_model = nn.Module()
    q_model.config = SimpleNamespace(
        model_type="qwen3_5_moe",
        text_config=SimpleNamespace(model_type="qwen3_5_moe_text", hidden_size=16, num_experts=4, num_experts_per_tok=2),
    )
    q_model.embed = nn.Embedding(32, 16)
    q_layer = nn.Module()
    q_layer.self_attn = nn.Module()
    q_layer.self_attn.q_proj = nn.Linear(16, 16, bias=False)
    q_layer.mlp = _ToyQwen35SparseBlock(16, num_experts=4, top_k=2)
    q_layer.mlp.shared_expert = nn.Module()
    for p in ("gate_proj", "up_proj", "down_proj"):
        setattr(q_layer.mlp.shared_expert, p, nn.Linear(16, 16, bias=False))
    q_layer.mlp.shared_expert.forward = lambda h: q_layer.mlp.shared_expert.down_proj(
        F.silu(q_layer.mlp.shared_expert.gate_proj(h)) * q_layer.mlp.shared_expert.up_proj(h)
    )
    q_model.model = SimpleNamespace(language_model=SimpleNamespace(layers=nn.ModuleList([q_layer])))
    q_model.layers = q_model.model.language_model.layers
    q_model.head = nn.Linear(16, 32, bias=False)
    q_model.forward = lambda input_ids, attention_mask=None, use_cache=False: SimpleNamespace(
        logits=q_model.head(q_layer.mlp(q_layer.self_attn.q_proj(q_model.embed(input_ids))))
    )
    ctx_q = patch_moe_for_window_distill(
        q_model, window_size=8, lora_rank=4, router_training="full", adapt_resident_mlp=True,
    )
    q_trainable = {n for n, p in q_model.named_parameters() if p.requires_grad}
    assert any("mlp.shared_expert.up_proj.lora_b" in n for n in q_trainable)
    assert any("mlp.shared_expert.down_proj.lora_b" in n for n in q_trainable)
    loss_q, _ = distill_step(q_model, ctx_q, torch.randint(0, 32, (2, 10)), torch.tensor([3, 4]))
    loss_q.backward()
    assert q_layer.mlp.shared_expert.down_proj.lora_b.grad.abs().sum() > 0

    # 2. Gemma4-style dense parallel MLP block alongside layer.router.
    g_model = nn.Module()
    g_model.config = SimpleNamespace(
        model_type="gemma4",
        text_config=SimpleNamespace(model_type="gemma4_text", hidden_size=16, num_experts=6, top_k_experts=2, enable_moe_block=True),
    )
    g_model.embed = nn.Embedding(32, 16)
    g_layer = nn.Module()
    g_layer.self_attn = nn.Module()
    g_layer.self_attn.q_proj = nn.Linear(16, 16, bias=False)
    g_layer.mlp = nn.Module()
    for p in ("gate_proj", "up_proj", "down_proj"):
        setattr(g_layer.mlp, p, nn.Linear(16, 16, bias=False))
    g_layer.router = _ToyGemma4Router(16, num_experts=6, top_k=2)
    g_layer.experts = nn.ModuleList([nn.Linear(16, 16, bias=False) for _ in range(6)])

    def _g_forward(input_ids, attention_mask=None, use_cache=False):
        h = g_model.embed(input_ids) + g_layer.self_attn.q_proj(g_model.embed(input_ids))
        dense = g_layer.mlp.down_proj(F.gelu(g_layer.mlp.gate_proj(h)) * g_layer.mlp.up_proj(h))
        _, w, idx = g_layer.router(h.reshape(-1, 16))
        moe = torch.zeros_like(h.reshape(-1, 16))
        flat = h.reshape(-1, 16)
        for k in range(idx.shape[-1]):
            for e in range(6):
                m = idx[:, k] == e
                if m.any():
                    moe[m] = moe[m] + g_layer.experts[e](flat[m]) * w[m, k : k + 1]
        return SimpleNamespace(logits=g_model.head(h + dense + moe.reshape_as(h)))

    g_model.model = SimpleNamespace(language_model=SimpleNamespace(layers=nn.ModuleList([g_layer])))
    g_model.layers = g_model.model.language_model.layers
    g_model.head = nn.Linear(16, 32, bias=False)
    g_model.forward = _g_forward
    ctx_g = patch_moe_for_window_distill(
        g_model, window_size=8, lora_rank=4, router_training="full", adapt_resident_mlp=True,
    )
    g_trainable = {n for n, p in g_model.named_parameters() if p.requires_grad}
    assert any("layers.0.mlp.up_proj.lora_b" in n for n in g_trainable)
    assert any("layers.0.mlp.down_proj.lora_b" in n for n in g_trainable)
    assert not g_layer.mlp.up_proj.base.weight.requires_grad
    loss_g, _ = distill_step(g_model, ctx_g, torch.randint(0, 32, (2, 10)), torch.tensor([3, 4]))
    loss_g.backward()
    assert g_layer.mlp.down_proj.lora_b.grad.abs().sum() > 0
