"""Regression and parity tests for PyTorch/CUDA structural optimizations."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from telos.diffusion.ar import ar_loss_fn_pytorch
from telos.diffusion.corosred_unified import RoutingMaskCache
from telos.models.components import Attention, RMSNorm, RotaryEmbedding, apply_rope
from telos.models.transformer import TelosTransformer


def test_hoisted_rope_matches_per_layer_cast_bitwise():
    torch.manual_seed(42)
    rope = RotaryEmbedding(64, max_seq_len=128)
    q1 = torch.randn(2, 4, 64, 64, dtype=torch.bfloat16, requires_grad=True)
    k1 = torch.randn(2, 4, 64, 64, dtype=torch.bfloat16, requires_grad=True)
    q2 = q1.detach().clone().requires_grad_(True)
    k2 = k1.detach().clone().requires_grad_(True)
    cq, ck = torch.randn_like(q1), torch.randn_like(k1)

    cos_2d, sin_2d = rope(q1, 64)
    qo1, ko1 = apply_rope(q1, k1, cos_2d, sin_2d)
    (qo1 * cq + ko1 * ck).sum().backward()

    cos_4d = cos_2d.to(dtype=q2.dtype).unsqueeze(0).unsqueeze(0)
    sin_4d = sin_2d.to(dtype=q2.dtype).unsqueeze(0).unsqueeze(0)
    qo2, ko2 = apply_rope(q2, k2, cos_4d, sin_4d)
    (qo2 * cq + ko2 * ck).sum().backward()

    assert torch.equal(qo1, qo2)
    assert torch.equal(ko1, ko2)
    assert torch.equal(q1.grad, q2.grad)
    assert torch.equal(k1.grad, k2.grad)


def test_fused_rmsnorm_fp32_parity_and_bf16_accuracy():
    torch.manual_seed(42)
    x32 = torch.randn(4, 32, 128, dtype=torch.float32, requires_grad=True)
    x32_ref = x32.detach().clone().requires_grad_(True)
    w = torch.randn(128, dtype=torch.float32)
    cot = torch.randn(4, 32, 128, dtype=torch.float32)

    norm = RMSNorm(128, eps=1e-6)
    norm.weight = torch.nn.Parameter(w.clone())
    y = norm(x32)
    y.backward(cot)

    rms = torch.sqrt(torch.mean(x32_ref ** 2, dim=-1, keepdim=True) + 1e-6)
    y_ref = (x32_ref / rms) * w
    y_ref.backward(cot)

    assert (y - y_ref).abs().max().item() <= 1e-5
    assert (x32.grad - x32_ref.grad).abs().max().item() <= 1e-5

    # In BF16, fused F.rms_norm should stay close to the FP32 reference.
    norm_bf = RMSNorm(128, eps=1e-6)
    norm_bf.weight = torch.nn.Parameter(w.bfloat16())
    y_bf = norm_bf(x32.detach().bfloat16()).float()
    assert (y_bf - y_ref.detach()).abs().mean().item() < 0.005


def test_single_shot_ar_cross_entropy_matches_chunked_above_4096():
    torch.manual_seed(42)
    logits1 = torch.randn(32, 130, 64, requires_grad=True)  # 4128 tokens > 4096
    logits2 = logits1.detach().clone().requires_grad_(True)
    seqs = torch.randint(4, 64, (32, 131), dtype=torch.long)

    loss, metrics = ar_loss_fn_pytorch(logits1, seqs)
    loss.backward()

    ref_loss = F.cross_entropy(logits2.reshape(-1, 64), seqs[:, 1:].reshape(-1), reduction="mean")
    ref_loss.backward()

    assert torch.equal(loss, ref_loss)
    assert torch.equal(metrics["unweighted_ce"], loss)
    assert torch.equal(logits1.grad, logits2.grad)


def test_gqa_broadcast_expand_matches_repeat_interleave_fwd_bwd():
    torch.manual_seed(42)
    attn = Attention(d_model=128, n_heads=8, n_kv_heads=2, is_causal=True)
    rope = RotaryEmbedding(16, max_seq_len=32)
    x = torch.randn(2, 16, 128, requires_grad=True)
    cos, sin = rope(x, 16)
    out = attn(x, cos, sin)
    out.sum().backward()
    assert out.shape == (2, 16, 128)
    assert x.grad is not None and torch.isfinite(x.grad).all()

    k = torch.randn(2, 2, 16, 16, requires_grad=True)
    k_ref = k.detach().clone().requires_grad_(True)
    cot = torch.randn(2, 8, 16, 16)
    y_exp = k[:, :, None, :, :].expand(2, 2, 4, 16, 16).reshape(2, 8, 16, 16)
    y_exp.backward(cot)
    y_rep = k_ref.repeat_interleave(4, dim=1)
    y_rep.backward(cot)
    assert torch.equal(y_exp, y_rep)
    assert torch.allclose(k.grad, k_ref.grad, atol=1e-6)


def test_routing_mask_cache_topk_preserves_bos_and_count():
    torch.manual_seed(42)
    model = TelosTransformer(
        vocab_size=64, d_model=64, n_layers=1, n_heads=4,
        max_seq_len=32, is_causal=True, use_reliability_head=True,
    )
    cache = RoutingMaskCache(refresh_every_steps=10, k_targeted_ratio=0.7)
    seqs = torch.randint(4, 64, (4, 32), dtype=torch.long)
    cache.refresh(model, seqs, current_step=1, mask_prob=0.25)
    assert len(cache.cached_masks) == 1
    mask = cache.cached_masks[0]
    assert mask.shape == (4, 32)
    assert not mask[:, 0].any()
    expected_k = max(1, int(max(1, int(32 * 0.25)) * 0.7))
    assert (mask.sum(dim=-1) == expected_k).all()


def test_prefetch_int32_staging_preserves_long_tokens():
    rng = np.random.default_rng(7)
    for np_dtype in (np.uint16, np.int32):
        src = rng.integers(0, 8192, size=(8, 32), dtype=np_dtype)
        stage_dtype = torch.int32 if src.dtype in (np.uint8, np.int16, np.uint16, np.int32) else torch.long
        assert stage_dtype == torch.int32
        buf = torch.empty((8, 32), dtype=stage_dtype)
        np.copyto(buf.numpy(), src, casting="unsafe")
        assert torch.equal(buf.to(dtype=torch.long), torch.from_numpy(src.astype(np.int64)))
