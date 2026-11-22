"""Unit and numerical parity tests for v1.8.6 pretraining optimizations."""

import torch
import torch.nn.functional as F

from telos.models.transformer import TelosTransformer
from telos.models.vocab_loss import tiled_cross_entropy, token_chunked_cross_entropy
from telos.training.cautious_optimizer import CautiousAdamW
from telos.training.schedule import compute_vectorized_roc_auc


def test_token_chunked_cross_entropy_matches_exact_forward_and_backward():
    torch.manual_seed(0)
    hidden = torch.randn(4, 15, 32, requires_grad=True)
    weight = torch.randn(128, 32, requires_grad=True)
    targets = torch.randint(0, 128, (4, 15))

    ref_h = hidden.detach().clone().requires_grad_(True)
    ref_w = weight.detach().clone().requires_grad_(True)
    ref_loss = F.cross_entropy(F.linear(ref_h, ref_w).reshape(-1, 128), targets.reshape(-1), reduction="none").reshape(4, 15)
    ref_loss.mean().backward()

    tok_loss = token_chunked_cross_entropy(hidden, weight, targets, chunk_size=16)
    tok_loss.mean().backward()

    assert torch.allclose(tok_loss, ref_loss, atol=1e-6, rtol=1e-5)
    assert torch.allclose(hidden.grad, ref_h.grad, atol=1e-6, rtol=1e-5)
    assert torch.allclose(weight.grad, ref_w.grad, atol=1e-6, rtol=1e-5)

    via_tiled = tiled_cross_entropy(hidden.detach(), weight.detach(), targets, chunk_size=16, chunk_axis="token")
    assert torch.allclose(via_tiled, ref_loss, atol=1e-6, rtol=1e-5)


def test_single_sort_roc_auc_matches_reference():
    torch.manual_seed(7)
    scores = torch.randn(512)
    labels = (torch.rand(512) > 0.4).float()

    # Reference Wilcoxon-Mann-Whitney rank-sum
    ref_ranks = torch.argsort(torch.argsort(scores)) + 1
    n_pos = (labels == 1.0).sum().item()
    n_neg = (labels == 0.0).sum().item()
    r1 = ref_ranks[labels == 1.0].sum().float().item()
    ref_auc = (r1 - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)

    got = compute_vectorized_roc_auc(scores, labels)
    assert abs(got - ref_auc) < 1e-7

    # Edge cases: all positive or all negative
    assert compute_vectorized_roc_auc(scores, torch.ones_like(labels)) == 0.5
    assert compute_vectorized_roc_auc(scores, torch.zeros_like(labels)) == 0.5


def test_cautious_adamw_decreases_loss_and_preserves_state_dict():
    torch.manual_seed(11)
    model = TelosTransformer(
        vocab_size=64, d_model=32, n_layers=2, n_heads=4, max_seq_len=32, is_causal=True,
    )
    opt = CautiousAdamW(model.parameters(), lr=1e-2, weight_decay=0.01)
    x = torch.randint(0, 64, (4, 16))

    losses = []
    for _ in range(10):
        opt.zero_grad()
        logits = model(x, mask_override=True, next_token_only=True)
        loss = F.cross_entropy(logits.reshape(-1, 64), x[:, 1:].reshape(-1))
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0]
    state = opt.state_dict()
    opt2 = CautiousAdamW(model.parameters(), lr=1e-2, weight_decay=0.01)
    opt2.load_state_dict(state)
    assert len(opt2.state_dict()["state"]) == len(state["state"])


def test_transformer_hoisted_checkpoint_forward_and_backward():
    torch.manual_seed(3)
    model = TelosTransformer(
        vocab_size=64, d_model=32, n_layers=2, n_heads=4, max_seq_len=32,
        is_causal=True, use_grad_checkpoint=True,
    )
    x = torch.randint(0, 64, (2, 16))
    logits = model(x, mask_override=True, next_token_only=True)
    loss = F.cross_entropy(logits.reshape(-1, 64), x[:, 1:].reshape(-1))
    loss.backward()
    assert model.tok_embeddings.weight.grad is not None
    assert torch.isfinite(model.tok_embeddings.weight.grad).all()
