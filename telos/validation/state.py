"""Exact full-state comparisons that fail closed on missing or nonfinite values."""
import math
from pathlib import Path

import torch


def assert_state_equal(left, right, path='$'):
    """Check every key, container type, tensor dtype/shape and finite value.

    No key intersections, implicit casts, tolerances or ignored fields. Use the
    same output path for control and restart runs to preserve configuration identity.
    Returns the number of tensor leaves checked; raises at the first mismatch.
    """
    if type(left) is not type(right):
        raise AssertionError(f'{path}: type mismatch {type(left)} != {type(right)}')
    if isinstance(left, torch.Tensor):
        if left.shape != right.shape or left.dtype != right.dtype:
            raise AssertionError(f'{path}: tensor shape/dtype mismatch')
        a, b = left.detach().cpu(), right.detach().cpu()
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise AssertionError(f'{path}: nonfinite tensor')
        if not torch.equal(a, b):
            raise AssertionError(f'{path}: tensor values differ')
        return 1
    if isinstance(left, dict):
        if left.keys() != right.keys():
            raise AssertionError(f'{path}: key mismatch {left.keys() ^ right.keys()}')
        return sum(assert_state_equal(left[k], right[k], f'{path}[{k!r}]') for k in left)
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise AssertionError(f'{path}: length mismatch')
        return sum(assert_state_equal(a, b, f'{path}[{i}]') for i, (a, b) in enumerate(zip(left, right)))
    if isinstance(left, (float, complex)):
        finite = (math.isfinite(left.real) and math.isfinite(left.imag) and
                  math.isfinite(right.real) and math.isfinite(right.imag))
        if not finite:
            raise AssertionError(f'{path}: nonfinite scalar')
    if not isinstance(left, (str, bytes, int, float, complex, bool, type(None), Path)):
        raise AssertionError(f'{path}: unsupported state type {type(left)}')
    if left != right:
        raise AssertionError(f'{path}: values differ: {left!r} != {right!r}')
    return 0


def compare_checkpoints(first, second):
    """Load tensor/basic-container checkpoints without enabling arbitrary pickle code."""
    a, b = (torch.load(p, map_location='cpu', weights_only=True) for p in (first, second))
    checked = assert_state_equal(a, b)
    if not checked:
        raise AssertionError('No tensor leaves checked')
    return {'exact_match': True, 'tensor_leaves': checked}
