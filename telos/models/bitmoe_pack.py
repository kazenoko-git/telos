"""Lossless BitNet b1.58 ternary -> MLX native 2-bit affine (uint32) packing utilities.

MLX's hardware-accelerated Apple Silicon Metal `mx.quantized_matmul` and `mx.gather_qmm`
evaluate `w = code * scale + bias` with `bits=2` (16 weights packed per uint32) and
`group_size in {32, 64, 128}`.

Setting `scale = s` (the exact BitLinear absmean scale) and `bias = -s` maps ternary
weights `{-s, 0, +s}` losslessly into 2-bit codes `{0, 1, 2}`:
    code 0 -> 0 * s - s = -s
    code 1 -> 1 * s - s =  0
    code 2 -> 2 * s - s = +s
"""

from __future__ import annotations

import numpy as np


def pack_ternary_to_mlx_2bit(
    weight: np.ndarray,
    *,
    group_size: int = 128,
    eps: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Losslessly quantize and pack a 2D [out, in] or 3D [E, out, in] weight for MLX `bits=2`.

    Returns:
        packed_u32: uint32 array of shape (..., out, in // 16)
        scales_f16: float16 array of shape (..., out, in // group_size)
        biases_f16: float16 array of shape (..., out, in // group_size)
    """
    w = np.asarray(weight, dtype=np.float32)
    if w.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D weight, got shape {w.shape}")
    in_dim = w.shape[-1]
    if in_dim % group_size != 0 or in_dim % 16 != 0:
        raise ValueError(f"Last dim ({in_dim}) must be divisible by group_size ({group_size}) and 16")

    if w.ndim == 3:
        scale = np.maximum(np.mean(np.abs(w), axis=(1, 2), keepdims=True), eps).astype(np.float32)
    else:
        scale = np.maximum(np.mean(np.abs(w)), eps).astype(np.float32)

    # Exact BitLinear absmean ternary trits in {-1, 0, +1} -> codes in {0, 1, 2}
    trits = np.clip(np.round(w / scale), -1.0, 1.0).astype(np.int8)
    codes = (trits + 1).astype(np.uint32)  # -1 -> 0, 0 -> 1, +1 -> 2

    # Pack 16 x 2-bit lanes into each uint32 (little-endian lane order matches MLX)
    grouped = codes.reshape(*w.shape[:-1], in_dim // 16, 16)
    shifts = (np.arange(16, dtype=np.uint32) * 2)
    packed = np.bitwise_or.reduce(grouped << shifts, axis=-1).astype(np.uint32)

    n_groups = in_dim // group_size
    if w.ndim == 3:
        s_bcast = np.broadcast_to(scale, (w.shape[0], w.shape[1], n_groups))
    else:
        s_bcast = np.full((w.shape[0], n_groups), float(scale), dtype=np.float32)

    scales_f16 = np.ascontiguousarray(s_bcast.astype(np.float16))
    biases_f16 = np.ascontiguousarray((-s_bcast).astype(np.float16))
    return np.ascontiguousarray(packed), scales_f16, biases_f16


def unpack_mlx_2bit_to_float(
    packed_u32: np.ndarray,
    scales_f16: np.ndarray,
    biases_f16: np.ndarray,
    *,
    group_size: int = 128,
) -> np.ndarray:
    """Reference CPU unpack matching MLX `mx.dequantize(..., group_size=group_size, bits=2)`."""
    shifts = (np.arange(16, dtype=np.uint32) * 2)
    codes = ((packed_u32[..., None] >> shifts) & np.uint32(0x3)).reshape(*packed_u32.shape[:-1], -1)
    n_groups = scales_f16.shape[-1]
    codes_g = codes.reshape(*codes.shape[:-1], n_groups, group_size).astype(np.float32)
    deq = codes_g * scales_f16.astype(np.float32)[..., None] + biases_f16.astype(np.float32)[..., None]
    return deq.reshape(*codes.shape)


def random_packed_ternary_mlx_2bit(
    shape: tuple[int, ...],
    rng: np.random.Generator,
    *,
    group_size: int = 128,
    scale_val: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Directly synthesize packed 2-bit ternary weights without allocating full float32 tensors.

    Used for fast random-weight benchmarking of the 2.25B model on Apple Silicon (~1s init).
    """
    in_dim = shape[-1]
    if in_dim % group_size != 0 or in_dim % 16 != 0:
        raise ValueError(f"Last dim ({in_dim}) must be divisible by group_size ({group_size}) and 16")
    # Uniform random 2-bit codes in {0, 1, 2} packed into uint32 via random uint8/uint32
    # Pack 16 random ternary trits per uint32 quickly using a 256-entry lookup table (4 trits/byte)
    lut = np.zeros(256, dtype=np.uint8)
    for b in range(256):
        c0 = (b & 3) % 3
        c1 = ((b >> 2) & 3) % 3
        c2 = ((b >> 4) & 3) % 3
        c3 = ((b >> 6) & 3) % 3
        lut[b] = c0 | (c1 << 2) | (c2 << 4) | (c3 << 6)
    raw_bytes = rng.integers(0, 256, size=(*shape[:-1], in_dim // 4), dtype=np.uint8)
    packed_bytes = lut[raw_bytes]
    packed_u32 = np.ascontiguousarray(packed_bytes).view(np.uint32)

    n_groups = in_dim // group_size
    scales_f16 = np.full((*shape[:-1], n_groups), scale_val, dtype=np.float16)
    biases_f16 = np.full((*shape[:-1], n_groups), -scale_val, dtype=np.float16)
    return packed_u32, scales_f16, biases_f16
