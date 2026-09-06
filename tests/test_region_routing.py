"""Unit test suite for Region Routing and Morphological Span Operations."""

import pytest
import numpy as np
import torch

from telos.diffusion.region import (
    dilate_flags,
    bridge_gaps,
    form_refinement_regions,
    dilate_flags_pytorch,
    bridge_gaps_pytorch,
    dilate_flags_numpy,
    bridge_gaps_numpy,
)


def test_dilation_pytorch_and_numpy():
    """Verify 1D dilation expands single flags symmetrically."""
    flags_pt = torch.tensor([[False, False, True, False, False]])
    flags_np = np.array([[False, False, True, False, False]])

    # Radius 1 expands index 2 to indices 1, 2, 3
    d_pt = dilate_flags_pytorch(flags_pt, radius=1)
    d_np = dilate_flags_numpy(flags_np, radius=1)

    expected = [[False, True, True, True, False]]
    assert d_pt.tolist() == expected
    assert d_np.tolist() == expected

    # Radius 2 expands index 2 to all indices 0..4
    d_pt_r2 = dilate_flags_pytorch(flags_pt, radius=2)
    d_np_r2 = dilate_flags_numpy(flags_np, radius=2)
    assert d_pt_r2.all()
    assert d_np_r2.all()


def test_gap_bridging_pytorch_and_numpy():
    """Verify gap bridging closes unflagged holes <= max_gap."""
    # Gap of 1 between index 0 and 2, gap of 2 between index 2 and 5, gap of 3 between index 5 and 9
    flags_pt = torch.tensor([[True, False, True, False, False, True, False, False, False, True]])
    flags_np = np.array([[True, False, True, False, False, True, False, False, False, True]])

    b_pt = bridge_gaps_pytorch(flags_pt, max_gap=2)
    b_np = bridge_gaps_numpy(flags_np, max_gap=2)

    # Gap 1 and Gap 2 closed; Gap 3 remains open
    expected = [[True, True, True, True, True, True, False, False, False, True]]
    assert b_pt.tolist() == expected
    assert b_np.tolist() == expected


def test_form_refinement_regions_expected_gain():
    """Verify form_refinement_regions with expected gain thresholding and prompt protection."""
    scores = torch.tensor([[0.05, 0.10, 0.85, 0.10, 0.15, 0.90, 0.05, 0.05]])
    # Expected gain threshold > 0.50 selects index 2 and index 5
    # Dilation r=1 expands index 2 -> {1, 2, 3}, index 5 -> {4, 5, 6}
    # Bridging max_gap=1 bridges gap between 3 and 4 -> all {1..6}
    # prompt_len = 2 protects indices 0, 1 -> only {2..6} remain True
    mask = form_refinement_regions(
        scores,
        mode="expected_gain",
        threshold=0.50,
        radius=1,
        max_gap=1,
        prompt_len=2
    )
    expected = [[False, False, True, True, True, True, True, False]]
    assert mask.tolist() == expected


def test_form_refinement_regions_top_k():
    """Verify top-k ratio correctly selects fraction of non-prompt tokens."""
    scores = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]])
    # Top 20% of 10 tokens = 2 tokens (indices 8, 9)
    # radius=0, max_gap=0 means no dilation
    mask = form_refinement_regions(
        scores,
        mode="expected_gain",
        top_k_ratio=0.20,
        radius=0,
        max_gap=0,
        prompt_len=0
    )
    assert mask[0, 8].item() is True
    assert mask[0, 9].item() is True
    assert mask[:, :8].sum().item() == 0
