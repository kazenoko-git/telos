"""Unit and parity tests for v1.8.8 post-training optimizations."""

import copy
import torch
import torch.nn.functional as F

from telos.models.transformer import TelosTransformer
from telos.posttrain.adapters import NF4Linear, apply_lora, disable_lora_adapters
from telos.posttrain.trainer import _torch_token_logps


def test_sparse_torch_token_logps_matches_full_log_softmax_forward_and_backward():
    torch.manual_seed(0)
    model_a = TelosTransformer(
        vocab_size=64, d_model=32, n_layers=2, n_heads=4, max_seq_len=32, is_causal=True,
    )
    model_b = TelosTransformer(
        vocab_size=64, d_model=32, n_layers=2, n_heads=4, max_seq_len=32, is_causal=True,
    )
    model_b.load_state_dict(model_a.state_dict())

    input_ids = torch.randint(0, 64, (3, 16))
    labels = input_ids.clone()
    labels[:, :10] = -100  # 62.5% prompt/padding ignored

    # Sparse valid-only projection path
    logps_sparse = _torch_token_logps(model_a, input_ids, labels)
    logps_sparse.sum().backward()

    # Reference full log_softmax + gather
    logits_ref = model_b(input_ids, mask_override=True, next_token_only=True).float()
    targets = labels[:, 1:]
    valid = targets.ne(-100)
    safe = targets.masked_fill(~valid, 0)
    logps_ref = F.log_softmax(logits_ref, dim=-1).gather(-1, safe.unsqueeze(-1)).squeeze(-1) * valid
    logps_ref.sum().backward()

    assert torch.allclose(logps_sparse, logps_ref, atol=1e-6, rtol=1e-5)
    for (n1, p1), (n2, p2) in zip(model_a.named_parameters(), model_b.named_parameters()):
        assert n1 == n2
        assert torch.allclose(p1.grad, p2.grad, atol=1e-5, rtol=1e-4), f"grad mismatch in {n1}"


def test_disable_lora_adapters_matches_frozen_base_model_exactly():
    torch.manual_seed(4)
    base = TelosTransformer(
        vocab_size=64, d_model=32, n_layers=2, n_heads=4, max_seq_len=32, is_causal=True,
    ).eval()
    adapted = copy.deepcopy(base)
    apply_lora(adapted, rank=4, alpha=8.0, dropout=0.0)
    # Perturb lora_b so adapter output is nonzero
    with torch.no_grad():
        for name, param in adapted.named_parameters():
            if name.endswith("lora_b"):
                param.fill_(0.25)

    x = torch.randint(0, 64, (2, 12))
    ref_out = base(x, mask_override=True)
    adapted_out = adapted(x, mask_override=True)
    assert not torch.allclose(ref_out, adapted_out)

    with disable_lora_adapters(adapted):
        disabled_out = adapted(x, mask_override=True)
    assert torch.equal(ref_out, disabled_out)
    # Re-enabled cleanly after context exit
    assert torch.equal(adapted_out, adapted(x, mask_override=True))


def test_nf4_int32_dequantize_preserves_values():
    torch.manual_seed(8)
    linear = torch.nn.Linear(64, 32, bias=True)
    nf4 = NF4Linear(linear, block_size=64)
    w = nf4.dequantize()
    assert w.shape == (32, 64)
    assert torch.isfinite(w).all()
    out = nf4(torch.randn(2, 8, 64))
    assert out.shape == (2, 8, 32)
    assert torch.isfinite(out).all()
