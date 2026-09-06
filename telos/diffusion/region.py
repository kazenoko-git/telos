"""Region Routing & Morphological Span Operations for Diffusion Refinement.

Converts tokenwise uncertainty or expected-gain signals into coherent, contiguous
refinement regions (dilated spans and bridged gaps) to prevent isolated-token
infilling artifacts during re-diffusion.
"""

from __future__ import annotations
import math
from typing import Union

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


def dilate_flags_pytorch(flags: torch.Tensor, radius: int = 1) -> torch.Tensor:
    """Dilates a boolean flag mask by `radius` tokens in both directions.

    Args:
        flags: Boolean tensor of shape `(B, T)` or `(T,)`.
        radius: Number of neighboring tokens to expand each flag to the left and right.

    Returns:
        Boolean tensor of identical shape with dilated flags.
    """
    if radius <= 0:
        return flags

    squeeze_dim = False
    if flags.ndim == 1:
        flags = flags.unsqueeze(0)
        squeeze_dim = True

    # Use 1D max pooling across the sequence dimension for vectorized dilation
    # Kernel size = 2 * radius + 1, with symmetric padding = radius
    flags_f32 = flags.unsqueeze(1).float()
    dilated_f32 = F.max_pool1d(flags_f32, kernel_size=2 * radius + 1, stride=1, padding=radius)
    out = dilated_f32.squeeze(1) > 0.5

    if squeeze_dim:
        out = out.squeeze(0)
    return out


def bridge_gaps_pytorch(flags: torch.Tensor, max_gap: int = 2) -> torch.Tensor:
    """Bridges unflagged gaps of length <= `max_gap` between flagged spans.

    Args:
        flags: Boolean tensor of shape `(B, T)` or `(T,)`.
        max_gap: Maximum number of unflagged tokens between two flagged positions to bridge.

    Returns:
        Boolean tensor with gaps <= max_gap filled as True.
    """
    if max_gap <= 0:
        return flags

    squeeze_dim = False
    if flags.ndim == 1:
        flags = flags.unsqueeze(0)
        squeeze_dim = True

    b, t = flags.shape
    device = flags.device
    idx = torch.arange(t, device=device).unsqueeze(0).expand(b, -1)

    # Compute index of the closest True flag to the left (using cumulative max)
    left_idx = torch.where(flags, idx, torch.full_like(idx, -100000))
    left_idx = torch.cummax(left_idx, dim=1).values
    dist_left = idx - left_idx

    # Compute index of the closest True flag to the right (using reversed cumulative min)
    right_idx = torch.where(flags, idx, torch.full_like(idx, 100000))
    right_idx = torch.flip(torch.cummin(torch.flip(right_idx, dims=[1]), dim=1).values, dims=[1])
    dist_right = right_idx - idx

    # Gap length at position i is dist_left + dist_right - 1
    # If both left and right flags exist and gap <= max_gap, mark as True
    gap_size = dist_left + dist_right - 1
    in_gap = (left_idx >= 0) & (right_idx < t) & (gap_size <= max_gap)
    out = flags | in_gap

    if squeeze_dim:
        out = out.squeeze(0)
    return out


def dilate_flags_numpy(flags: np.ndarray, radius: int = 1) -> np.ndarray:
    """NumPy implementation of 1D binary dilation."""
    if radius <= 0:
        return flags
    out = np.copy(flags)
    for r in range(1, radius + 1):
        out[..., r:] |= flags[..., :-r]
        out[..., :-r] |= flags[..., r:]
    return out


def bridge_gaps_numpy(flags: np.ndarray, max_gap: int = 2) -> np.ndarray:
    """NumPy implementation of gap bridging."""
    if max_gap <= 0:
        return flags
    out = np.copy(flags)
    orig_1d = flags.ndim == 1
    if orig_1d:
        flags = flags[np.newaxis, :]

    b, t = flags.shape
    idx = np.arange(t)[np.newaxis, :].repeat(b, axis=0)

    left_idx = np.where(flags, idx, -100000)
    left_idx = np.maximum.accumulate(left_idx, axis=1)
    dist_left = idx - left_idx

    right_idx = np.where(flags, idx, 100000)
    right_idx = np.flip(np.minimum.accumulate(np.flip(right_idx, axis=1), axis=1), axis=1)
    dist_right = right_idx - idx

    gap_size = dist_left + dist_right - 1
    in_gap = (left_idx >= 0) & (right_idx < t) & (gap_size <= max_gap)
    res = flags | in_gap

    return res[0] if orig_1d else res


def dilate_flags(flags: Union['torch.Tensor', 'np.ndarray'], radius: int = 1) -> Union['torch.Tensor', 'np.ndarray']:
    """Framework-agnostic entry point for flag dilation."""
    if TORCH_AVAILABLE and isinstance(flags, torch.Tensor):
        return dilate_flags_pytorch(flags, radius=radius)
    elif NUMPY_AVAILABLE and isinstance(flags, np.ndarray):
        return dilate_flags_numpy(flags, radius=radius)
    raise TypeError(f"Unsupported array type for dilate_flags: {type(flags)}")


def bridge_gaps(flags: Union['torch.Tensor', 'np.ndarray'], max_gap: int = 2) -> Union['torch.Tensor', 'np.ndarray']:
    """Framework-agnostic entry point for gap bridging."""
    if TORCH_AVAILABLE and isinstance(flags, torch.Tensor):
        return bridge_gaps_pytorch(flags, max_gap=max_gap)
    elif NUMPY_AVAILABLE and isinstance(flags, np.ndarray):
        return bridge_gaps_numpy(flags, max_gap=max_gap)
    raise TypeError(f"Unsupported array type for bridge_gaps: {type(flags)}")


def form_refinement_regions(
    scores: Union['torch.Tensor', 'np.ndarray'],
    mode: str = "expected_gain",
    threshold: float | None = None,
    top_k_ratio: float | None = None,
    radius: int = 1,
    max_gap: int = 2,
    prompt_len: int = 0
) -> Union['torch.Tensor', 'np.ndarray']:
    """Converts continuous scoring signals into contiguous refinement mask spans.

    Args:
        scores: Tensor or ndarray of shape `(B, T)` containing:
            - `mode="expected_gain"`: Higher values indicate greater expected improvement.
            - `mode="error_prob"`: Higher values indicate higher probability of error.
            - `mode="reliability"`: Lower values indicate errors (e.g. raw Phase A logits).
        mode: One of `"expected_gain"`, `"error_prob"`, or `"reliability"`.
        threshold: Absolute cutoff value. If None, top_k_ratio is used.
        top_k_ratio: Fraction of tokens to flag (e.g. 0.20 for top 20%).
        radius: Dilation radius for expanding flagged positions into surrounding context.
        max_gap: Maximum gap between nearby flagged positions to bridge into a single span.
        prompt_len: Length of immutable prefix prompt to protect from refinement.

    Returns:
        Boolean mask of shape `(B, T)` where True indicates positions to be masked/refined.
    """
    is_torch = TORCH_AVAILABLE and isinstance(scores, torch.Tensor)

    # Step 1: Initial tokenwise thresholding / top-k selection
    if is_torch:
        b, t = scores.shape
        flags = torch.zeros_like(scores, dtype=torch.bool)
        
        # Invert scores if mode is reliability (lower = worse)
        effective_scores = -scores if mode == "reliability" else scores

        if threshold is not None:
            if mode == "reliability":
                flags = scores < threshold
            else:
                flags = scores > threshold
        elif top_k_ratio is not None:
            k = max(1, int(round((t - prompt_len) * top_k_ratio)))
            # Mask out prompt positions during ranking
            eff_for_topk = effective_scores.clone()
            if prompt_len > 0:
                eff_for_topk[:, :prompt_len] = -float("inf")
            _, topk_idx = torch.topk(eff_for_topk, k=k, dim=-1)
            flags.scatter_(1, topk_idx, True)
        else:
            raise ValueError("Either threshold or top_k_ratio must be provided.")
    else:
        b, t = scores.shape
        flags = np.zeros_like(scores, dtype=bool)
        effective_scores = -scores if mode == "reliability" else scores

        if threshold is not None:
            flags = (scores < threshold) if mode == "reliability" else (scores > threshold)
        elif top_k_ratio is not None:
            k = max(1, int(round((t - prompt_len) * top_k_ratio)))
            eff_for_topk = np.copy(effective_scores)
            if prompt_len > 0:
                eff_for_topk[:, :prompt_len] = -1e9
            topk_idx = np.argsort(eff_for_topk, axis=-1)[:, -k:]
            for row in range(b):
                flags[row, topk_idx[row]] = True
        else:
            raise ValueError("Either threshold or top_k_ratio must be provided.")

    # Step 2: Dilation (expand flagged tokens into local contiguous spans)
    dilated = dilate_flags(flags, radius=radius)

    # Step 3: Gap Bridging (close small holes between adjacent spans)
    bridged = bridge_gaps(dilated, max_gap=max_gap)

    # Step 4: Protect prompt prefix
    if prompt_len > 0:
        if is_torch:
            bridged[:, :prompt_len] = False
        else:
            bridged[:, :prompt_len] = False

    return bridged
