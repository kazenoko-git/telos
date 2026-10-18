"""Contracts for the diagnostic that replaces the incomplete historical audit."""
from unittest.mock import patch

import pytest
import torch

from scripts.compare_saved_50m_gradients import gradient_geometry, task_pass
from telos.models import TelosConfig, TelosTransformer


def tiny_model():
    torch.manual_seed(42)
    return TelosTransformer(TelosConfig(vocab_size=16, d_model=16, n_layers=1,
                            n_heads=2, n_kv_heads=2, max_seq_len=8,
                            is_causal=True, use_reliability_head=True)).eval()


def test_audit_explicitly_changes_attention_for_infill():
    model = tiny_model()
    tokens = torch.tensor([[2, 4, 5, 6, 7, 8, 9, 3]])
    mask = torch.tensor([[False, False, True, True, False, False, False, False]])
    with patch.object(model, 'forward', wraps=model.forward) as calls:
        task_pass(model, [tokens], [mask], infill=False)
        assert calls.call_args.kwargs['mask_override'] is True
        task_pass(model, [tokens], [mask], infill=True)
        assert calls.call_args.kwargs['mask_override'] is False


@pytest.mark.parametrize('infill', [False, True])
def test_audit_microbatches_equal_full_fixture_gradient(infill):
    model = tiny_model()
    tokens = torch.tensor([[2, 4, 5, 6, 7, 8, 9, 3], [2, 9, 8, 7, 6, 5, 4, 3]])
    # Different mask counts catch accidental averaging of microbatch means.
    mask = torch.tensor([[False, True, False, False, False, False, False, False],
                         [False, True, True, True, True, False, False, False]])
    full_metrics, full_gradients = task_pass(model, [tokens], [mask], infill=infill)
    split_metrics, split_gradients = task_pass(
        model, list(tokens.split(1)), list(mask.split(1)), infill=infill)
    assert full_metrics['ce'] == pytest.approx(split_metrics['ce'], abs=1e-6)
    assert full_metrics['top1_pct'] == split_metrics['top1_pct']
    for key in full_gradients:
        torch.testing.assert_close(full_gradients[key], split_gradients[key], atol=1e-6, rtol=1e-5)


def test_gradient_geometry_detects_task_conflict():
    g = {'layers.0.weight': torch.tensor([3., 4.])}
    opposite = {'layers.0.weight': -g['layers.0.weight']}
    result = gradient_geometry(g, opposite, .5, .5)['all_backbone']
    assert result['causal_gradient_norm'] == 5.
    assert result['causal_infill_cosine'] == -1.
    assert result['combined_gradient_norm'] == 0.
