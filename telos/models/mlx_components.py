"""
Unified MLX Transformer Components.

Architecture:
- RMSNorm pre-normalization
- SwiGLU MLP activation
- Rotary Position Embeddings (RoPE) via mx.fast.rope (traditional=True)
- Grouped Query Attention (GQA) with configurable Causal Masking
"""

import math

from telos.training.core import metal_usable

if not metal_usable():
    raise ImportError("MLX is installed but Metal is unavailable in this session.")

import mlx.core as mx
import mlx.nn as nn


_PACKED_SWIGLU_FORWARD = mx.fast.metal_kernel(
    name="telos_packed_swiglu_forward",
    input_names=["packed"],
    output_names=["activated"],
    source=r"""
        uint i = thread_position_in_grid.x;
        if (i >= N * F) return;
        uint row = i / F;
        uint col = i % F;
        float gate = float(packed[row * (2 * F) + col]);
        float up = float(packed[row * (2 * F) + F + col]);
        T sigmoid = T(1.0f / (1.0f + exp(-gate)));
        T silu = T(gate * float(sigmoid));
        activated[i] = T(float(silu) * up);
    """,
)

_PACKED_SWIGLU_BACKWARD = mx.fast.metal_kernel(
    name="telos_packed_swiglu_backward",
    input_names=["packed", "cotangent"],
    output_names=["gradient"],
    source=r"""
        uint i = thread_position_in_grid.x;
        if (i >= N * F) return;
        uint row = i / F;
        uint col = i % F;
        float gate = float(packed[row * (2 * F) + col]);
        float up = float(packed[row * (2 * F) + F + col]);
        T sigmoid = T(1.0f / (1.0f + exp(-gate)));
        T silu = T(gate * float(sigmoid));
        T dy = T(cotangent[i]);

        // Keep BF16 intermediate rounding close to MLX's split SwiGLU graph.
        T grad_silu = T(float(dy) * up);
        T grad_up = T(float(dy) * float(silu));
        T grad_gate_direct = T(float(grad_silu) * float(sigmoid));
        T grad_sigmoid = T(float(grad_silu) * gate);
        T one_minus_sigmoid = T(1.0f - float(sigmoid));
        T sigmoid_grad = T(float(sigmoid) * float(one_minus_sigmoid));
        T grad_gate_sigmoid = T(float(grad_sigmoid) * float(sigmoid_grad));

        gradient[row * (2 * F) + col] =
            T(float(grad_gate_direct) + float(grad_gate_sigmoid));
        gradient[row * (2 * F) + F + col] = grad_up;
    """,
)


@mx.custom_function
def _experimental_packed_swiglu(packed):
    """Experimental packed SwiGLU activation; first-order gradients only."""
    rows, twice_hidden = packed.shape
    hidden = twice_hidden // 2
    return _PACKED_SWIGLU_FORWARD(
        inputs=[packed],
        template=[("N", rows), ("F", hidden), ("T", packed.dtype)],
        grid=(rows * hidden, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, hidden)],
        output_dtypes=[packed.dtype],
    )[0]


@_experimental_packed_swiglu.vjp
def _experimental_packed_swiglu_vjp(primals, cotangent, output):
    packed = primals
    rows, twice_hidden = packed.shape
    hidden = twice_hidden // 2
    return _PACKED_SWIGLU_BACKWARD(
        inputs=[packed, cotangent],
        template=[("N", rows), ("F", hidden), ("T", packed.dtype)],
        grid=(rows * hidden, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[packed.shape],
        output_dtypes=[packed.dtype],
    )[0]


class MLXRMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((d_model,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, self.weight, self.eps)


class MLXSwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network."""

    def __init__(self, d_model: int):
        super().__init__()
        hidden = int(d_model * 8 / 3)
        hidden = ((hidden + 63) // 64) * 64
        self.hidden_dim = hidden
        self.use_experimental_packed_kernel = False
        # One wider GEMM replaces separate gate and up projections.
        self.gate_up = nn.Linear(d_model, 2 * hidden, bias=False)
        self.w3 = nn.Linear(hidden, d_model, bias=False)

    def __call__(self, x):
        packed = self.gate_up(x)
        if self.use_experimental_packed_kernel:
            leading_shape = packed.shape[:-1]
            packed_2d = packed.reshape((-1, 2 * self.hidden_dim))
            activated = _experimental_packed_swiglu(packed_2d)
            activated = activated.reshape(leading_shape + (self.hidden_dim,))
        else:
            gate, up = mx.split(packed, 2, axis=-1)
            activated = nn.silu(gate) * up
        return self.w3(activated)


class MLXBlock(nn.Module):
    """Transformer block with configurable causal/bidirectional attention."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, is_causal: bool = False):
        super().__init__()
        self.norm1 = MLXRMSNorm(d_model)
        self.norm2 = MLXRMSNorm(d_model)
        self.head_dim = d_model // n_heads
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.kv_groups = n_heads // n_kv_heads
        self.is_causal = is_causal

        self.qkv_proj = nn.Linear(d_model, (n_heads + 2 * n_kv_heads) * self.head_dim, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.mlp = MLXSwiGLU(d_model)

        # Precompute QKV split boundaries (static across all forward passes)
        self._q_end = n_heads * self.head_dim
        self._k_end = self._q_end + n_kv_heads * self.head_dim

    def __call__(self, x, mask_override=None):
        B, T, D = x.shape
        h = self.norm1(x)

        qkv = self.qkv_proj(h)
        # Split using precomputed static index boundaries
        q, k, v = mx.split(qkv, [self._q_end, self._k_end], axis=-1)
        q = q.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Apply hardware-accelerated RoPE Metal kernel
        q = mx.fast.rope(q, self.head_dim, traditional=True, base=10000.0, scale=1.0, offset=0)
        k = mx.fast.rope(k, self.head_dim, traditional=True, base=10000.0, scale=1.0, offset=0)

        scale = 1.0 / (self.head_dim ** 0.5)
        
        # Determine mask mode: MLX fast SDPA expects "causal", None, or an mx.array mask
        if mask_override is True or mask_override == "causal":
            mask_arg = "causal"
        elif mask_override is False or (mask_override is None and not self.is_causal):
            mask_arg = None
        elif self.is_causal:
            mask_arg = "causal"
        else:
            mask_arg = mask_override

        if mask_arg is not None:
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask_arg)
        else:
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)

        out = out.transpose(0, 2, 1, 3).reshape(B, T, D)

        x = x + self.out(out)
        x = x + self.mlp(self.norm2(x))
        return x
