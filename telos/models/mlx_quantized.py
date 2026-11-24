"""Load an affine MLX projection or embedding from its saved tensors."""

import mlx.core as mx
import mlx.nn as nn


class AffineWeights(nn.Module):
    """A matrix stored as packed integers, with one scale and bias per group."""

    def __init__(self, weight, scales, biases, *, shape, group_size, bits):
        super().__init__()
        rows, width = shape
        if bits not in {2, 3, 4, 5, 6, 8} or group_size not in {32, 64, 128}:
            raise ValueError(f"Unsupported affine quantization: {bits} bits, group size {group_size}")
        groups, remainder = divmod(width, group_size)
        if (remainder or weight.dtype != mx.uint32 or weight.shape != (rows, width * bits // 32)
                or scales.shape != (rows, groups) or biases.shape != scales.shape
                or not mx.issubdtype(scales.dtype, mx.floating) or biases.dtype != scales.dtype):
            raise ValueError(f"Invalid packed affine tensors for a {rows} × {width} matrix")
        self.weight = weight
        self.scales = scales
        self.biases = biases
        self.group_size = group_size
        self.bits = bits

    def as_linear(self, x):
        return mx.quantized_matmul(
            x, self.weight, self.scales, self.biases,
            transpose=True, group_size=self.group_size, bits=self.bits,
        )

    def lookup(self, indices):
        # Only the requested rows are dequantized, never the embedding matrix.
        return mx.dequantize(
            self.weight[indices], self.scales[indices], self.biases[indices],
            group_size=self.group_size, bits=self.bits,
        )
