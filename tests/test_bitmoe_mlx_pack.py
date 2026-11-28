"""Unit tests verifying lossless BitNet b1.58 ternary -> MLX 2-bit affine packing parity."""

from __future__ import annotations

import numpy as np
import torch

from telos.models.bitlinear import bitlinear_weight_quant
from telos.models.bitmoe_pack import (
    pack_ternary_to_mlx_2bit,
    random_packed_ternary_mlx_2bit,
    unpack_mlx_2bit_to_float,
)


def test_pack_ternary_2d_matches_pytorch_bitlinear_losslessly():
    rng = np.random.default_rng(7)
    w_np = rng.normal(0.0, 0.05, size=(256, 512)).astype(np.float32)

    packed, scales, biases = pack_ternary_to_mlx_2bit(w_np, group_size=128)
    assert packed.dtype == np.uint32
    assert packed.shape == (256, 512 // 16)
    assert scales.shape == (256, 512 // 128)
    assert biases.shape == (256, 512 // 128)

    unpacked = unpack_mlx_2bit_to_float(packed, scales, biases, group_size=128)
    expected = bitlinear_weight_quant(torch.from_numpy(w_np)).numpy()

    # Only fp16 scale rounding (~1e-4 relative on 0.04 scale -> < 5e-5 absolute)
    max_diff = float(np.max(np.abs(unpacked - expected)))
    assert max_diff < 1e-4, f"2D ternary packing error too large: {max_diff}"


def test_pack_ternary_3d_moe_matches_pytorch_per_expert_losslessly():
    rng = np.random.default_rng(13)
    # Distinct scales per expert to verify independent per-expert absmean scaling
    w_np = rng.normal(0.0, 0.02, size=(6, 128, 256)).astype(np.float32)
    for e in range(6):
        w_np[e] *= (e + 1)

    packed, scales, biases = pack_ternary_to_mlx_2bit(w_np, group_size=128)
    assert packed.shape == (6, 128, 256 // 16)
    assert scales.shape == (6, 128, 256 // 128)

    unpacked = unpack_mlx_2bit_to_float(packed, scales, biases, group_size=128)
    expected = bitlinear_weight_quant(torch.from_numpy(w_np)).numpy()

    max_diff = float(np.max(np.abs(unpacked - expected)))
    assert max_diff < 2e-4, f"3D MoE ternary packing error too large: {max_diff}"


def test_random_packed_ternary_contains_only_valid_trit_codes():
    rng = np.random.default_rng(99)
    packed, scales, biases = random_packed_ternary_mlx_2bit((4, 64, 256), rng, group_size=128, scale_val=0.05)
    unpacked = unpack_mlx_2bit_to_float(packed, scales, biases, group_size=128)
    unique_vals = {round(float(v), 4) for v in np.unique(unpacked)}
    assert unique_vals.issubset({-0.05, 0.0, 0.05})
