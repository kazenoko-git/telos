"""Unit and parity tests for post-v1.8.8 optimizations."""

import copy
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer, models, pre_tokenizers

from telos.diffusion.mdlm import mdlm_loss_from_model_pytorch, mdlm_loss_pytorch
from telos.diffusion.sampler import MDLMSampler
from telos.models.config import TelosConfig
from telos.models.transformer import TelosTransformer
from telos.posttrain import grpo
from telos.posttrain.trainer import _sft_loss, _sft_loss_from_model


def _tokenizer():
    tokens = ["[PAD]", "[MASK]", "[BOS]", "[EOS]", "[UNK]", "Q", "A", "B", "C", "D"]
    tokenizer = Tokenizer(models.WordLevel({token: i for i, token in enumerate(tokens)}, unk_token="[UNK]"))
    tokenizer.add_special_tokens(tokens[:5])
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return tokenizer


@pytest.mark.parametrize("n_kv_heads", [4, 2])
def test_pytorch_kv_cache_matches_full_causal_forward(n_kv_heads):
    torch.manual_seed(11)
    cfg = TelosConfig(
        vocab_size=64, d_model=32, n_layers=2, n_heads=4, n_kv_heads=n_kv_heads,
        max_seq_len=32, is_causal=True,
    )
    model = TelosTransformer(cfg).eval()
    tokens = torch.randint(4, 64, (3, 12))

    with torch.no_grad():
        full_hidden = model.hidden_states(tokens, mask_override=True)
        # Prefill 8 tokens, then decode 4 tokens one at a time
        h_prefill, cache = model.hidden_states_with_cache(tokens[:, :8], mask_override=True)
        assert torch.allclose(h_prefill, full_hidden[:, :8], atol=1e-5, rtol=1e-5)
        for step in range(8, 12):
            h_step, cache = model.hidden_states_with_cache(
                tokens[:, step : step + 1], kv_cache=cache, mask_override=True,
            )
            assert torch.allclose(h_step[:, 0], full_hidden[:, step], atol=1e-5, rtol=1e-5)


def test_collect_rollouts_cached_matches_uncached_greedy_including_ragged_and_eos():
    torch.manual_seed(7)
    tokenizer = _tokenizer()
    cfg = TelosConfig(
        vocab_size=tokenizer.get_vocab_size(), d_model=32, n_layers=2, n_heads=4, n_kv_heads=2,
        max_seq_len=16, is_causal=True,
    )
    model = TelosTransformer(cfg).eval()
    records = [{"prompt": "Q", "answer": "A"}, {"prompt": "Q Q", "answer": "B"}]
    prompts, eos, pad = grpo.prepare_prompts(records, tokenizer, max_length=10, max_new_tokens=4)

    @torch.no_grad()
    def sample_uncached(ids, positions):
        logits = model(
            torch.as_tensor(ids, dtype=torch.long),
            mask_override=True,
            logit_positions=torch.tensor(positions, dtype=torch.long),
        )
        logps = logits.float().log_softmax(dim=-1)
        tokens = logps.argmax(dim=-1, keepdim=True)
        return tokens.squeeze(-1).tolist(), logps.gather(-1, tokens).squeeze(-1).tolist()

    @torch.no_grad()
    def sample_prefill(prompt_ids, group_size):
        h, cache = model.hidden_states_with_cache(
            torch.as_tensor(prompt_ids, dtype=torch.long), mask_override=True,
        )
        logps = model.output_projection(h[:, -1, :]).float().log_softmax(dim=-1)
        logps = logps.repeat_interleave(group_size, dim=0)
        cache = [(k.repeat_interleave(group_size, dim=0), v.repeat_interleave(group_size, dim=0)) for k, v in cache]
        tokens = logps.argmax(dim=-1, keepdim=True)
        return tokens.squeeze(-1).tolist(), logps.gather(-1, tokens).squeeze(-1).tolist(), cache

    @torch.no_grad()
    def sample_decode(step_ids, cache):
        h, cache = model.hidden_states_with_cache(
            torch.as_tensor(step_ids, dtype=torch.long), kv_cache=cache, mask_override=True,
        )
        logps = model.output_projection(h[:, -1, :]).float().log_softmax(dim=-1)
        tokens = logps.argmax(dim=-1, keepdim=True)
        return tokens.squeeze(-1).tolist(), logps.gather(-1, tokens).squeeze(-1).tolist(), cache

    def slice_cache(cache, keep_rows):
        idx = torch.as_tensor(keep_rows, dtype=torch.long)
        return [(k.index_select(0, idx), v.index_select(0, idx)) for k, v in cache]

    reward_fn = grpo.reward_function("exact")
    rows_u, logps_u, adv_u, mean_u, unif_u = grpo.collect_rollouts(
        prompts, tokenizer, eos, pad, 3, 4, sample_uncached, reward_fn,
    )
    rows_c, logps_c, adv_c, mean_c, unif_c = grpo.collect_rollouts_cached(
        prompts, tokenizer, eos, pad, 3, 4, sample_prefill, sample_decode, slice_cache, reward_fn,
    )
    assert [r["_encoded"] for r in rows_u] == [r["_encoded"] for r in rows_c]
    for lu, lc in zip(logps_u, logps_c, strict=True):
        np.testing.assert_allclose(lu, lc, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(adv_u, adv_c, atol=1e-6)
    assert mean_u == pytest.approx(mean_c)
    assert unif_u == pytest.approx(unif_c)


def test_sparse_sft_loss_from_model_matches_dense_forward_and_backward():
    torch.manual_seed(19)
    cfg = TelosConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, max_seq_len=32, is_causal=True)
    model_a = TelosTransformer(cfg)
    model_b = copy.deepcopy(model_a)

    input_ids = torch.randint(0, 64, (3, 16))
    labels = input_ids.clone()
    labels[:, :11] = -100

    loss_sparse = _sft_loss_from_model(model_a, input_ids, labels, torch.device("cpu"))
    loss_sparse.backward()

    logits_dense = model_b(input_ids, mask_override=True, next_token_only=True).float()
    loss_dense = _sft_loss(logits_dense, labels, torch.device("cpu"))
    loss_dense.backward()

    assert torch.allclose(loss_sparse, loss_dense, atol=1e-6, rtol=1e-5)
    for (n1, p1), (n2, p2) in zip(model_a.named_parameters(), model_b.named_parameters()):
        assert n1 == n2
        assert torch.allclose(p1.grad, p2.grad, atol=1e-5, rtol=1e-4), f"grad mismatch in {n1}"

    # All-ignored labels edge case produces zero loss with valid graph
    all_ignored = torch.full_like(labels, -100)
    zero_loss = _sft_loss_from_model(model_a, input_ids, all_ignored, torch.device("cpu"))
    assert zero_loss.item() == 0.0


def test_sparse_mdlm_loss_from_model_matches_dense_forward_and_backward():
    torch.manual_seed(23)
    cfg = TelosConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, max_seq_len=32, is_causal=False)
    model_a = TelosTransformer(cfg)
    model_b = copy.deepcopy(model_a)

    targets = torch.randint(4, 64, (4, 16))
    mask_pos = torch.rand(4, 16) < 0.5
    masked_ids = torch.where(mask_pos, 1, targets)
    t_vals = torch.full((4, 1), 0.5)

    loss_sparse, metrics_sparse = mdlm_loss_from_model_pytorch(model_a, masked_ids, targets, mask_pos, t_vals)
    loss_sparse.backward()

    logits_dense = model_b(masked_ids, mask_override=False)
    loss_dense, metrics_dense = mdlm_loss_pytorch(logits_dense, targets, mask_pos, t_vals)
    loss_dense.backward()

    assert torch.allclose(loss_sparse, loss_dense, atol=1e-6, rtol=1e-5)
    assert torch.allclose(metrics_sparse["unweighted_ce"], metrics_dense["unweighted_ce"], atol=1e-6)
    for (n1, p1), (n2, p2) in zip(model_a.named_parameters(), model_b.named_parameters()):
        assert n1 == n2
        assert torch.allclose(p1.grad, p2.grad, atol=1e-5, rtol=1e-4), f"grad mismatch in {n1}"


def test_mdlm_sampler_masked_only_head_matches_full_head_exactly():
    torch.manual_seed(29)
    cfg = TelosConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, max_seq_len=32, is_causal=False)
    model = TelosTransformer(cfg).eval()

    class _FullForwardOnly(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            return self.inner(x)

    prompt = torch.randint(4, 64, (1, 6))
    sampler_sparse = MDLMSampler(model, mask_token_id=1, num_steps=8, temperature=0.0, repetition_penalty=1.0)
    sampler_full = MDLMSampler(_FullForwardOnly(model), mask_token_id=1, num_steps=8, temperature=0.0, repetition_penalty=1.0)

    out_sparse = sampler_sparse.sample(seq_len=20, prompt_ids=prompt)
    out_full = sampler_full.sample(seq_len=20, prompt_ids=prompt)
    assert torch.equal(out_sparse, out_full)
