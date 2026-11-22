"""Unit and parity tests for v1.8.7 diffusion and interleaved decoding optimizations."""

import torch
import torch.nn.functional as F

from telos.diffusion.corosred import crsr_phase_b_loss_fn_pytorch, crsr_phase_b_self_conditioned_loss_fn_pytorch
from telos.diffusion.corosred_unified import RoutingMaskCache, corosred_unified_step_pytorch
from telos.diffusion.sampler import MDLMSampler
from telos.models.transformer import TelosTransformer


def test_telos_transformer_reliability_scores_matches_full_forward():
    torch.manual_seed(0)
    model = TelosTransformer(
        vocab_size=64, d_model=32, n_layers=2, n_heads=4, max_seq_len=32,
        is_causal=True, use_reliability_head=True,
    ).eval()
    x = torch.randint(0, 64, (3, 16))
    _, full_r = model(x, mask_override=True, return_reliability=True)
    fast_r = model.reliability_scores(x, mask_override=True)
    assert torch.equal(full_r, fast_r)


def test_corosred_unified_masked_only_projection_matches_full_forward_and_backward():
    torch.manual_seed(42)
    model_a = TelosTransformer(
        vocab_size=64, d_model=32, n_layers=2, n_heads=4, max_seq_len=32,
        is_causal=True, use_reliability_head=True,
    )
    model_b = TelosTransformer(
        vocab_size=64, d_model=32, n_layers=2, n_heads=4, max_seq_len=32,
        is_causal=True, use_reliability_head=True,
    )
    model_b.load_state_dict(model_a.state_dict())

    x = torch.randint(4, 64, (4, 16))
    weights = {"alpha": 0.6, "beta": 0.4, "gamma": 0.1, "mask_blend": 0.0}

    # Sparse masked-only projection (native model)
    torch.manual_seed(123)
    loss_sparse, m_sparse = corosred_unified_step_pytorch(
        model_a, x, vocab_size=64, schedule_weights=weights, mask_prob=0.25,
    )
    loss_sparse.backward()

    # Reference full projection via wrapper that bypasses sparse path
    class _Wrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.module = inner
        def forward(self, *args, **kwargs):
            return self.module(*args, **kwargs)

    torch.manual_seed(123)
    loss_full, m_full = corosred_unified_step_pytorch(
        _Wrapper(model_b), x, vocab_size=64, schedule_weights=weights, mask_prob=0.25,
    )
    loss_full.backward()

    assert torch.allclose(loss_sparse, loss_full, atol=1e-6, rtol=1e-5)
    assert torch.allclose(m_sparse["infill_ce"], m_full["infill_ce"], atol=1e-6, rtol=1e-5)
    for (n1, p1), (n2, p2) in zip(model_a.named_parameters(), model_b.named_parameters()):
        assert n1 == n2
        assert torch.allclose(p1.grad, p2.grad, atol=1e-5, rtol=1e-4), f"grad mismatch in {n1}"


def test_routing_mask_cache_refresh_uses_reliability_scores_without_lm_head():
    torch.manual_seed(1)
    model = TelosTransformer(
        vocab_size=64, d_model=32, n_layers=1, n_heads=4, max_seq_len=32,
        is_causal=True, use_reliability_head=True,
    )
    called_lm_head = False
    orig_proj = model.output_projection.forward

    def spy_proj(*args, **kwargs):
        nonlocal called_lm_head
        called_lm_head = True
        return orig_proj(*args, **kwargs)

    model.output_projection.forward = spy_proj
    cache = RoutingMaskCache(refresh_every_steps=10)
    cache.refresh(model, torch.randint(4, 64, (4, 16)), current_step=1, mask_prob=0.25)
    assert not called_lm_head
    assert len(cache.cached_masks) == 1
    assert not cache.cached_masks[0][:, 0].any()


def test_mdlm_sampler_vectorized_and_early_commit_margin():
    torch.manual_seed(5)
    model = TelosTransformer(
        vocab_size=64, d_model=32, n_layers=1, n_heads=4, max_seq_len=32, is_causal=False,
    ).eval()
    prompt = torch.tensor([[2, 10, 11, 12]])
    sampler = MDLMSampler(model, mask_token_id=1, num_steps=8, temperature=0.01, repetition_penalty=1.2)
    out = sampler.sample(seq_len=16, prompt_ids=prompt)
    assert out.shape == (1, 16)
    assert torch.equal(out[:, :4], prompt)
    assert not (out[:, 4:] == 1).any()

    # Early commit margin = 0.0 should commit all remaining tokens on step 1
    fast_sampler = MDLMSampler(
        model, mask_token_id=1, num_steps=8, temperature=0.01, early_commit_margin=0.0,
    )
    out_fast = fast_sampler.sample(seq_len=16, prompt_ids=prompt)
    assert out_fast.shape == (1, 16)
    assert not (out_fast[:, 4:] == 1).any()


def test_corosred_phase_b_and_c_run_cleanly_without_clear_autocast_cache():
    torch.manual_seed(9)
    model = TelosTransformer(
        vocab_size=64, d_model=32, n_layers=1, n_heads=4, max_seq_len=32,
        is_causal=True, use_reliability_head=True,
    )
    x = torch.randint(4, 64, (2, 16))
    loss_b, _ = crsr_phase_b_loss_fn_pytorch(model, x, vocab_size=64, mask_token_id=1, mask_prob=0.25)
    loss_c, _ = crsr_phase_b_self_conditioned_loss_fn_pytorch(model, x, vocab_size=64, mask_token_id=1, mask_prob=0.25)
    assert torch.isfinite(loss_b) and torch.isfinite(loss_c)
