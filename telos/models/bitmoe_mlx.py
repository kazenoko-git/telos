"""Native Apple Silicon MLX inference engine for Telos Alpha 1 (BitMoE-2.25B-A377M).

Executes BitNet b1.58 ternary weights losslessly via Apple Silicon Metal's native
2-bit affine `mx.quantized_matmul` and `mx.gather_qmm`, with aggressive dispatch
collapsing for Apple M5 Pro:
  - Fused `qkvg_proj` + `ab_proj` in Gated DeltaNet and `qkv_proj` in SWA
  - Custom fused Metal GDN kernel (`mx.fast.metal_kernel`) collapsing 4-tap conv,
    SiLU, L2-norm, FP32 Delta-Rule recurrence, head RMSNorm, and SiLU(g) into
    1 single Metal dispatch per GDN layer (for both T=1 decode and T=4 MTP verify)
  - Shared expert fused as Expert #E inside `BitMoEMLX` (`K=3` single `gather_qmm`),
    eliminating 40 sequential quantized GEMV dispatches per token
  - Fixed-shape `(B, KV, W, D_h)` rolling SWA KV buffer for zero-retrace `mx.compile`
  - Optional 4-bit quantized LM head (4x less bandwidth for T=1 and MTP K=3)
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from .bitmoe_pack import pack_ternary_to_mlx_2bit, random_packed_ternary_mlx_2bit


# Fused single-dispatch Metal kernel for Gated DeltaNet (B=1, any T>=1, head_dim=128):
# Performs causal K-tap depthwise conv + SiLU + L2-norm + FP32 delta-rule state update
# + per-head RMSNorm + SiLU(g) gate in ONE threadgroup per head (8 x 128 threads).
_GDN_FUSED_KERNEL = mx.fast.metal_kernel(
    name="telos_gdn_fused_step",
    input_names=["qkvg", "conv_hist", "conv_w", "conv_b", "ab", "state_in", "head_norm"],
    output_names=["out", "new_conv_hist", "state_out"],
    source="""
const uint tid = thread_position_in_threadgroup.x;
const uint h = threadgroup_position_in_grid.y;
const uint D = HEAD_DIM;
const uint H = N_HEADS;
const uint INNER = H * D;
const uint CONV_DIM = 3 * INNER;
const uint HIST = CONV_K - 1;
const uint T_LEN = SEQ_LEN;

threadgroup float s_q[HEAD_DIM];
threadgroup float s_k[HEAD_DIM];
threadgroup float s_v[HEAD_DIM];
threadgroup float s_red[HEAD_DIM];

const uint c_q = h * D + tid;
const uint c_k = INNER + h * D + tid;
const uint c_v = 2 * INNER + h * D + tid;
const uint c_g = 3 * INNER + h * D + tid;

// Write updated rolling conv history for channels q, k, v
for (uint j = 0; j < HIST; ++j) {
    uint src = T_LEN + j;
    if (src < HIST) {
        new_conv_hist[j * CONV_DIM + c_q] = conv_hist[src * CONV_DIM + c_q];
        new_conv_hist[j * CONV_DIM + c_k] = conv_hist[src * CONV_DIM + c_k];
        new_conv_hist[j * CONV_DIM + c_v] = conv_hist[src * CONV_DIM + c_v];
    } else {
        uint t_src = src - HIST;
        new_conv_hist[j * CONV_DIM + c_q] = qkvg[t_src * (4 * INNER) + c_q];
        new_conv_hist[j * CONV_DIM + c_k] = qkvg[t_src * (4 * INNER) + c_k];
        new_conv_hist[j * CONV_DIM + c_v] = qkvg[t_src * (4 * INNER) + c_v];
    }
}

const uint state_base = h * D * D;
const float hn_w = float(head_norm[tid]);

for (uint step = 0; step < T_LEN; ++step) {
    // 1. Causal K-tap depthwise conv + SiLU for Q, K, V
    float acc_q = float(conv_b[c_q]);
    float acc_k = float(conv_b[c_k]);
    float acc_v = float(conv_b[c_v]);
    for (uint tap = 0; tap < CONV_K; ++tap) {
        uint src = step + tap;
        float xq, xk, xv;
        if (src < HIST) {
            xq = float(conv_hist[src * CONV_DIM + c_q]);
            xk = float(conv_hist[src * CONV_DIM + c_k]);
            xv = float(conv_hist[src * CONV_DIM + c_v]);
        } else {
            uint t_src = src - HIST;
            xq = float(qkvg[t_src * (4 * INNER) + c_q]);
            xk = float(qkvg[t_src * (4 * INNER) + c_k]);
            xv = float(qkvg[t_src * (4 * INNER) + c_v]);
        }
        acc_q += xq * float(conv_w[c_q * CONV_K + tap]);
        acc_k += xk * float(conv_w[c_k * CONV_K + tap]);
        acc_v += xv * float(conv_w[c_v * CONV_K + tap]);
    }
    float q_val = acc_q / (1.0f + metal::exp(-acc_q));
    float k_val = acc_k / (1.0f + metal::exp(-acc_k));
    float v_val = acc_v / (1.0f + metal::exp(-acc_v));

    // 2. L2-normalize Q across D threads
    s_red[tid] = q_val * q_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = D >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) s_red[tid] += s_red[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    s_q[tid] = q_val * metal::rsqrt(s_red[0] + 1e-6f);

    // 3. L2-normalize K across D threads
    s_red[tid] = k_val * k_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = D >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) s_red[tid] += s_red[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    s_k[tid] = k_val * metal::rsqrt(s_red[0] + 1e-6f);
    s_v[tid] = v_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 4. Scalar retention (alpha) and delta write (beta) gates
    float a_raw = float(ab[step * (2 * H) + h]);
    float b_raw = float(ab[step * (2 * H) + H + h]);
    float alpha = 1.0f / (1.0f + metal::exp(-a_raw));
    float beta = 1.0f / (1.0f + metal::exp(-b_raw));

    // 5. Delta-rule prediction: v_pred[tid] = sum_r k[r] * (alpha * S[h, r, tid])
    float v_pred = 0.0f;
    if (step == 0) {
        for (uint r = 0; r < D; ++r) {
            v_pred += s_k[r] * (state_in[state_base + r * D + tid] * alpha);
        }
    } else {
        for (uint r = 0; r < D; ++r) {
            v_pred += s_k[r] * (state_out[state_base + r * D + tid] * alpha);
        }
    }
    float delta_tid = beta * (s_v[tid] - v_pred);

    // 6. State update + readout: S_new = alpha*S + k^T * delta, o[tid] = sum_r q[r] * S_new[r, tid]
    float o_val = 0.0f;
    if (step == 0) {
        for (uint r = 0; r < D; ++r) {
            float s_new = state_in[state_base + r * D + tid] * alpha + s_k[r] * delta_tid;
            state_out[state_base + r * D + tid] = s_new;
            o_val += s_q[r] * s_new;
        }
    } else {
        for (uint r = 0; r < D; ++r) {
            float s_new = state_out[state_base + r * D + tid] * alpha + s_k[r] * delta_tid;
            state_out[state_base + r * D + tid] = s_new;
            o_val += s_q[r] * s_new;
        }
    }

    // 7. Per-head RMSNorm + SiLU(g) gate
    s_red[tid] = o_val * o_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = D >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) s_red[tid] += s_red[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float norm_o = o_val * metal::rsqrt(s_red[0] / float(D) + 1e-6f) * hn_w;
    float g_raw = float(qkvg[step * (4 * INNER) + c_g]);
    float g_silu = g_raw / (1.0f + metal::exp(-g_raw));
    out[step * INNER + h * D + tid] = T(norm_o * g_silu);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
""",
)


def _l2_normalize(x: mx.array, eps: float = 1e-6) -> mx.array:
    return x * mx.rsqrt(mx.sum(x * x, axis=-1, keepdims=True) + eps)


class GDNState(NamedTuple):
    conv_history: mx.array   # (B, K-1, 3*inner_dim)
    recurrent: mx.array      # (B, H, D_h, D_h) fp32


class SWACache(NamedTuple):
    keys: mx.array           # (B, KV, W, D_h) fixed static shape for mx.compile
    values: mx.array         # (B, KV, W, D_h) fixed static shape for mx.compile
    offset: mx.array         # () int32 scalar for RoPE + valid mask


class TernaryLinearMLX(nn.Module):
    """Lossless BitNet b1.58 linear layer backed by Metal 2-bit `mx.quantized_matmul`."""

    def __init__(
        self,
        weight_u32: mx.array,
        scales: mx.array,
        biases: mx.array,
        *,
        subln_weight: mx.array | None = None,
        group_size: int = 128,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.weight = weight_u32
        self.scales = scales
        self.biases = biases
        self.subln_weight = subln_weight
        self.group_size = group_size
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        if self.subln_weight is not None:
            x = mx.fast.rms_norm(x, self.subln_weight, self.eps)
        return mx.quantized_matmul(
            x,
            self.weight,
            self.scales,
            self.biases,
            transpose=True,
            group_size=self.group_size,
            bits=2,
        )

    @staticmethod
    def fuse(layers: list["TernaryLinearMLX"]) -> "TernaryLinearMLX":
        return TernaryLinearMLX(
            mx.concatenate([l.weight for l in layers], axis=0),
            mx.concatenate([l.scales for l in layers], axis=0),
            mx.concatenate([l.biases for l in layers], axis=0),
            subln_weight=layers[0].subln_weight,
            group_size=layers[0].group_size,
            eps=layers[0].eps,
        )


class DenseLinearMLX(nn.Module):
    """Small FP16/BF16 linear projection (gates & router)."""

    def __init__(self, weight: mx.array, bias: mx.array | None = None):
        super().__init__()
        self.weight = weight
        self.bias = bias

    def __call__(self, x: mx.array) -> mx.array:
        y = x @ self.weight.T
        if self.bias is not None:
            y = y + self.bias
        return y


class SquaredReLUMLPMLX(nn.Module):
    """Two-matrix ReLU^2 FFN backed by 2-bit Metal `TernaryLinearMLX`."""

    def __init__(self, w_up: TernaryLinearMLX, w_down: TernaryLinearMLX):
        super().__init__()
        self.w_up = w_up
        self.w_down = w_down

    def __call__(self, x: mx.array) -> mx.array:
        h = nn.relu(self.w_up(x))
        return self.w_down(h * h)


class GatedDeltaNetMLX(nn.Module):
    """Fused Gated DeltaNet using custom single-dispatch Metal kernel for decode & MTP."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        head_dim: int,
        conv_kernel: int,
        *,
        qkvg_proj: TernaryLinearMLX,
        o_proj: TernaryLinearMLX,
        ab_proj: DenseLinearMLX,
        conv_weight: mx.array,   # (3*inner_dim, conv_kernel)
        conv_bias: mx.array,     # (3*inner_dim,)
        head_norm: mx.array,     # (head_dim,)
        eps: float = 1e-6,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.inner_dim = n_heads * head_dim
        self.conv_kernel = conv_kernel
        self.history = conv_kernel - 1
        self.eps = eps

        self.qkvg_proj = qkvg_proj
        self.o_proj = o_proj
        self.ab_proj = ab_proj
        self.conv_weight = conv_weight
        self.conv_bias = conv_bias
        self.head_norm = head_norm

    def init_state(self, batch: int = 1, dtype=mx.float16) -> GDNState:
        return GDNState(
            conv_history=mx.zeros((batch, self.history, 3 * self.inner_dim), dtype=dtype),
            recurrent=mx.zeros((batch, self.n_heads, self.head_dim, self.head_dim), dtype=mx.float32),
        )

    def __call__(self, x: mx.array, state: GDNState) -> tuple[mx.array, GDNState]:
        b, t, _ = x.shape
        qkvg = self.qkvg_proj(x)
        ab = self.ab_proj(x)

        # Single-dispatch custom Metal kernel for B=1 decode & MTP verify (power-of-2 head_dim)
        if b == 1 and (self.head_dim & (self.head_dim - 1)) == 0 and t <= 16:
            out, new_hist, new_rec = _GDN_FUSED_KERNEL(
                inputs=[
                    qkvg[0],
                    state.conv_history[0],
                    self.conv_weight,
                    self.conv_bias,
                    ab[0],
                    state.recurrent[0],
                    self.head_norm,
                ],
                template=[
                    ("T", x.dtype),
                    ("HEAD_DIM", self.head_dim),
                    ("N_HEADS", self.n_heads),
                    ("CONV_K", self.conv_kernel),
                    ("SEQ_LEN", t),
                ],
                grid=(self.head_dim, self.n_heads, 1),
                threadgroup=(self.head_dim, 1, 1),
                output_shapes=[
                    (1, t, self.inner_dim),
                    (1, self.history, 3 * self.inner_dim),
                    (1, self.n_heads, self.head_dim, self.head_dim),
                ],
                output_dtypes=[x.dtype, x.dtype, mx.float32],
            )
            return self.o_proj(out), GDNState(new_hist, new_rec)

        # General prefill fallback for long sequences (T > 16)
        qkv = qkvg[:, :, : 3 * self.inner_dim]
        g = qkvg[:, :, 3 * self.inner_dim :]
        conv_in = mx.concatenate((state.conv_history, qkv), axis=1)
        next_hist = conv_in[:, -self.history :, :]
        acc = conv_in[:, :t, :] * self.conv_weight[:, 0]
        for tap in range(1, self.conv_kernel):
            acc = acc + conv_in[:, tap : tap + t, :] * self.conv_weight[:, tap]
        conved = nn.silu(acc + self.conv_bias)
        q, k, v = mx.split(conved, 3, axis=-1)
        q = _l2_normalize(q.reshape(b, t, self.n_heads, self.head_dim).astype(mx.float32))
        k = _l2_normalize(k.reshape(b, t, self.n_heads, self.head_dim).astype(mx.float32))
        v = v.reshape(b, t, self.n_heads, self.head_dim).astype(mx.float32)
        ab_sig = mx.sigmoid(ab.astype(mx.float32))
        alpha = ab_sig[:, :, : self.n_heads]
        beta = ab_sig[:, :, self.n_heads :]
        s = state.recurrent
        outs = []
        for idx in range(t):
            s = s * alpha[:, idx, :, None, None]
            k_t = k[:, idx]
            v_t = v[:, idx]
            v_pred = mx.sum(k_t[:, :, :, None] * s, axis=-2)
            delta = beta[:, idx, :, None] * (v_t - v_pred)
            s = s + k_t[:, :, :, None] * delta[:, :, None, :]
            outs.append(mx.sum(q[:, idx, :, :, None] * s, axis=-2))
        out = mx.stack(outs, axis=1)
        out = mx.fast.rms_norm(out.astype(x.dtype), self.head_norm, self.eps)
        out = out.reshape(b, t, self.inner_dim) * nn.silu(g)
        return self.o_proj(out), GDNState(next_hist, s)


class SlidingWindowAttentionMLX(nn.Module):
    """Fused QKV Grouped-Query Sliding-Window Attention with static-shape rolling KV cache."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        sliding_window: int,
        *,
        qkv_proj: TernaryLinearMLX,
        o_proj: TernaryLinearMLX,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.kv_dim = n_kv_heads * self.head_dim
        self.sliding_window = sliding_window
        self.scale = self.head_dim ** -0.5
        self.rope_base = rope_base

        self.qkv_proj = qkv_proj
        self.o_proj = o_proj

    def init_state(self, batch: int = 1, dtype=mx.float16) -> SWACache:
        return SWACache(
            keys=mx.zeros((batch, self.n_kv_heads, self.sliding_window, self.head_dim), dtype=dtype),
            values=mx.zeros((batch, self.n_kv_heads, self.sliding_window, self.head_dim), dtype=dtype),
            offset=mx.array(0, dtype=mx.int32),
        )

    def __call__(self, x: mx.array, cache: SWACache) -> tuple[mx.array, SWACache]:
        b, t, _ = x.shape
        qkv = self.qkv_proj(x)
        q = qkv[:, :, : self.d_model].reshape(b, t, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = qkv[:, :, self.d_model : self.d_model + self.kv_dim].reshape(b, t, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = qkv[:, :, self.d_model + self.kv_dim :].reshape(b, t, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = mx.fast.rope(q, self.head_dim, traditional=False, base=self.rope_base, scale=1.0, offset=cache.offset)
        k = mx.fast.rope(k, self.head_dim, traditional=False, base=self.rope_base, scale=1.0, offset=cache.offset)

        w = self.sliding_window
        if t >= w:
            k_buf = k[:, :, -w:, :]
            v_buf = v[:, :, -w:, :]
        else:
            k_buf = mx.concatenate((cache.keys[:, :, t:, :], k), axis=2)
            v_buf = mx.concatenate((cache.values[:, :, t:, :], v), axis=2)

        new_offset = cache.offset + t
        key_pos = mx.arange(w, dtype=mx.int32) + (new_offset - w)
        q_pos = mx.arange(t, dtype=mx.int32) + cache.offset
        valid = (key_pos[None, :] >= 0) & (key_pos[None, :] <= q_pos[:, None])
        mask = mx.where(valid, mx.array(0.0, dtype=x.dtype), mx.array(-1e4, dtype=x.dtype))[None, None, :, :]

        attn = mx.fast.scaled_dot_product_attention(q, k_buf, v_buf, scale=self.scale, mask=mask)
        attn = attn.transpose(0, 2, 1, 3).reshape(b, t, self.d_model)
        return self.o_proj(attn), SWACache(k_buf, v_buf, new_offset)


class BitMoEMLX(nn.Module):
    """Sparse Top-2 MoE with shared expert fused as Expert #E into a single K=3 `mx.gather_qmm`."""

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        *,
        router: DenseLinearMLX,
        expert_bias: mx.array,
        w_up_u32: mx.array,
        w_up_scales: mx.array,
        w_up_biases: mx.array,
        w_down_u32: mx.array,
        w_down_scales: mx.array,
        w_down_biases: mx.array,
        expert_subln: mx.array | None = None,
        shared_expert: SquaredReLUMLPMLX | None = None,
        group_size: int = 128,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = router
        self.expert_bias = expert_bias
        self.group_size = group_size
        self.eps = eps

        # Fuse shared_expert as Expert #num_experts when d_ff matches (eliminates 2 GEMV dispatches/layer)
        self.fused_shared = (
            shared_expert is not None
            and shared_expert.w_up.weight.shape == w_up_u32.shape[1:]
            and shared_expert.w_down.weight.shape == w_down_u32.shape[1:]
        )
        if self.fused_shared:
            self.w_up_u32 = mx.concatenate((w_up_u32, shared_expert.w_up.weight[None]), axis=0)
            self.w_up_scales = mx.concatenate((w_up_scales, shared_expert.w_up.scales[None]), axis=0)
            self.w_up_biases = mx.concatenate((w_up_biases, shared_expert.w_up.biases[None]), axis=0)
            self.w_down_u32 = mx.concatenate((w_down_u32, shared_expert.w_down.weight[None]), axis=0)
            self.w_down_scales = mx.concatenate((w_down_scales, shared_expert.w_down.scales[None]), axis=0)
            self.w_down_biases = mx.concatenate((w_down_biases, shared_expert.w_down.biases[None]), axis=0)
            if expert_subln is not None or shared_expert.w_down.subln_weight is not None:
                d_ff = expert_subln.shape[0] if expert_subln is not None else shared_expert.w_down.subln_weight.shape[0]
                e_sn = expert_subln if expert_subln is not None else mx.ones((d_ff,), dtype=mx.float16)
                s_sn = shared_expert.w_down.subln_weight if shared_expert.w_down.subln_weight is not None else mx.ones((d_ff,), dtype=mx.float16)
                # Shape (1, K+1, 1, d_ff): first K slots use routed expert_subln, last slot uses shared subln
                self.slot_subln = mx.concatenate([e_sn[None]] * top_k + [s_sn[None]], axis=0)[None, :, None, :]
            else:
                self.slot_subln = None
            self.shared_expert = None
            self.expert_subln = None
        else:
            self.w_up_u32 = w_up_u32
            self.w_up_scales = w_up_scales
            self.w_up_biases = w_up_biases
            self.w_down_u32 = w_down_u32
            self.w_down_scales = w_down_scales
            self.w_down_biases = w_down_biases
            self.expert_subln = expert_subln
            self.shared_expert = shared_expert
            self.slot_subln = None

    def __call__(self, x: mx.array) -> mx.array:
        b, t, d = x.shape
        n = b * t
        x_flat = x.reshape(n, d)

        logits = self.router(x_flat).astype(mx.float32)
        biased = logits + self.expert_bias[None, :]
        topk_idx = mx.argpartition(-biased, kth=self.top_k - 1, axis=-1)[:, : self.top_k]
        selected = mx.take_along_axis(logits, topk_idx, axis=-1)
        gate_w = mx.softmax(selected, axis=-1).astype(x.dtype)

        if self.fused_shared:
            shared_idx = mx.full((n, 1), self.num_experts, dtype=topk_idx.dtype)
            all_idx = mx.concatenate((topk_idx, shared_idx), axis=-1)  # (N, K+1)
            all_w = mx.concatenate((gate_w, mx.ones((n, 1), dtype=x.dtype)), axis=-1)  # (N, K+1)
        else:
            all_idx = topk_idx
            all_w = gate_w

        x_in = x_flat[:, None, None, :]
        h = mx.gather_qmm(
            x_in,
            self.w_up_u32,
            self.w_up_scales,
            self.w_up_biases,
            rhs_indices=all_idx,
            transpose=True,
            group_size=self.group_size,
            bits=2,
        )
        h = nn.relu(h)
        h = h * h
        if self.slot_subln is not None:
            h = mx.fast.rms_norm(h, None, self.eps) * self.slot_subln
        elif self.expert_subln is not None:
            h = mx.fast.rms_norm(h, self.expert_subln, self.eps)
        y = mx.gather_qmm(
            h,
            self.w_down_u32,
            self.w_down_scales,
            self.w_down_biases,
            rhs_indices=all_idx,
            transpose=True,
            group_size=self.group_size,
            bits=2,
        ).squeeze(2)
        out = mx.sum(y * all_w[:, :, None], axis=1).reshape(b, t, d)

        if self.shared_expert is not None:
            return out + self.shared_expert(x)
        return out


class MTPHeadMLX(nn.Module):
    """Single sequential Multi-Token Prediction head backed by 2-bit Metal projections."""

    def __init__(
        self,
        h_norm: mx.array,
        e_norm: mx.array,
        fuse_proj: TernaryLinearMLX,
        mlp_norm: mx.array,
        mlp: SquaredReLUMLPMLX,
        out_norm: mx.array,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.h_norm = h_norm
        self.e_norm = e_norm
        self.fuse_proj = fuse_proj
        self.mlp_norm = mlp_norm
        self.mlp = mlp
        self.out_norm = out_norm
        self.eps = eps

    def __call__(self, prev_hidden: mx.array, next_embed: mx.array) -> mx.array:
        fused = mx.concatenate(
            (mx.fast.rms_norm(prev_hidden, self.h_norm, self.eps),
             mx.fast.rms_norm(next_embed, self.e_norm, self.eps)),
            axis=-1,
        )
        h = self.fuse_proj(fused)
        h = h + self.mlp(mx.fast.rms_norm(h, self.mlp_norm, self.eps))
        return mx.fast.rms_norm(h, self.out_norm, self.eps)


class BitMoEBlockMLX(nn.Module):
    def __init__(self, attn_norm: mx.array, mixer: nn.Module, mlp_norm: mx.array, mlp: BitMoEMLX, eps: float = 1e-6):
        super().__init__()
        self.attn_norm = attn_norm
        self.mixer = mixer
        self.mlp_norm = mlp_norm
        self.mlp = mlp
        self.eps = eps

    def __call__(self, x: mx.array, cache: Any):
        mixed, next_cache = self.mixer(mx.fast.rms_norm(x, self.attn_norm, self.eps), cache)
        h = x + mixed
        return h + self.mlp(mx.fast.rms_norm(h, self.mlp_norm, self.eps)), next_cache


class BitMoETransformerMLX(nn.Module):
    """Full Telos Alpha 1 (`BitMoE-2.25B-A377M`) compiled Apple Silicon MLX inference engine."""

    def __init__(
        self,
        cfg: dict[str, Any],
        *,
        embed_weight: mx.array,
        layers: list[BitMoEBlockMLX],
        final_norm: mx.array,
        mtp_heads: list[MTPHeadMLX],
        lm_head_bits: int = 4,
        lm_head_group_size: int = 64,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = int(cfg["vocab_size"])
        self.d_model = int(cfg["d_model"])
        self.embed_weight = embed_weight
        self.layers = layers
        self.final_norm = final_norm
        self.mtp_heads = mtp_heads
        self.lm_head_bits = lm_head_bits
        self.lm_head_group_size = lm_head_group_size
        self.eps = eps

        if lm_head_bits in (4, 8):
            w_q, s_q, b_q = mx.quantize(embed_weight, group_size=lm_head_group_size, bits=lm_head_bits)
            self.lm_w_q = w_q
            self.lm_s_q = s_q
            self.lm_b_q = b_q
        else:
            self.lm_w_q = None
            self.lm_s_q = None
            self.lm_b_q = None

    def init_caches(self, batch: int = 1, dtype=mx.float16) -> list[Any]:
        return [layer.mixer.init_state(batch=batch, dtype=dtype) for layer in self.layers]

    def project_logits(self, h: mx.array) -> mx.array:
        if self.lm_w_q is not None:
            return mx.quantized_matmul(
                h, self.lm_w_q, self.lm_s_q, self.lm_b_q,
                transpose=True, group_size=self.lm_head_group_size, bits=self.lm_head_bits,
            )
        return h @ self.embed_weight.T

    def forward_with_cache(self, input_ids: mx.array, caches: list[Any]):
        x = self.embed_weight[input_ids]
        next_caches = []
        for idx, layer in enumerate(self.layers):
            x, c = layer(x, caches[idx])
            next_caches.append(c)
        h = mx.fast.rms_norm(x, self.final_norm, self.eps)
        logits = self.project_logits(h)
        return logits, h, next_caches

    def decode_step_t1(self, token_id: mx.array, caches: list[Any]):
        """Single-token decode step designed for `mx.compile`."""
        logits, _, next_caches = self.forward_with_cache(token_id, caches)
        next_tok = mx.argmax(logits, axis=-1)
        return next_tok, next_caches

    def draft_mtp(self, last_hidden: mx.array, last_token_id: mx.array) -> mx.array:
        """Greedy-draft K future tokens (B, K) from the last backbone hidden state."""
        h_curr = last_hidden
        tok_curr = last_token_id
        drafts = []
        for head in self.mtp_heads:
            e_curr = self.embed_weight[tok_curr]
            h_curr = head(h_curr, e_curr)
            logits = self.project_logits(h_curr)
            tok_curr = mx.argmax(logits, axis=-1)
            drafts.append(tok_curr)
        return mx.concatenate(drafts, axis=-1)

    def mtp_verify_and_draft_step(self, draft_window: mx.array, caches: list[Any]):
        """Parallel T=(1+K) backbone verify + K-head greedy draft, designed for `mx.compile`."""
        logits, hidden, next_caches = self.forward_with_cache(draft_window, caches)
        next_tok = mx.argmax(logits[:, -1:, :], axis=-1)
        drafts = self.draft_mtp(hidden[:, -1:, :], next_tok)
        next_window = mx.concatenate((next_tok, drafts), axis=-1)
        return next_window, next_caches

    @classmethod
    def from_config_random(
        cls,
        model_cfg: dict[str, Any],
        *,
        seed: int = 42,
        group_size: int = 128,
        lm_head_bits: int = 4,
    ) -> "BitMoETransformerMLX":
        rng = np.random.default_rng(seed)
        d = int(model_cfg["d_model"])
        v = int(model_cfg["vocab_size"])
        n_layers = int(model_cfg["n_layers"])
        n_heads = int(model_cfg["n_heads"])
        n_kv = int(model_cfg.get("n_kv_heads", n_heads))
        swa_every = int(model_cfg.get("hybrid_swa_every", 4))
        window = int(model_cfg.get("sliding_window", 512))
        gdn_h = int(model_cfg.get("gdn_n_heads", 8))
        gdn_hd = int(model_cfg.get("gdn_head_dim", d // gdn_h))
        gdn_inner = gdn_h * gdn_hd
        gdn_k = int(model_cfg.get("gdn_conv_kernel", 4))
        n_exp = int(model_cfg.get("moe_num_experts", 24))
        top_k = int(model_cfg.get("moe_top_k", 2))
        d_ff = int(model_cfg.get("moe_d_ff", 2048))
        shared_d_ff = int(model_cfg.get("moe_shared_d_ff", 2048))
        use_subln = bool(model_cfg.get("use_subln", True))
        mtp_k = int(model_cfg.get("mtp_num_heads", 3))
        mtp_d_ff = int(model_cfg.get("mtp_d_ff", d_ff))

        def _tlin(in_dim: int, out_dim: int, subln: bool = False) -> TernaryLinearMLX:
            w, s, b = random_packed_ternary_mlx_2bit((out_dim, in_dim), rng, group_size=group_size)
            sn = mx.ones((in_dim,), dtype=mx.float16) if subln else None
            return TernaryLinearMLX(mx.array(w), mx.array(s), mx.array(b), subln_weight=sn, group_size=group_size)

        embed_w = mx.array(rng.normal(0.0, 0.02, size=(v, d)).astype(np.float16))
        layers: list[BitMoEBlockMLX] = []

        for i in range(n_layers):
            is_swa = ((i + 1) % swa_every == 0)
            if is_swa:
                kv_dim = n_kv * (d // n_heads)
                mixer: nn.Module = SlidingWindowAttentionMLX(
                    d, n_heads, n_kv, window,
                    qkv_proj=_tlin(d, d + 2 * kv_dim),
                    o_proj=_tlin(d, d, subln=use_subln),
                )
            else:
                ab_bias = np.concatenate((np.full((gdn_h,), 3.0), np.full((gdn_h,), -1.0))).astype(np.float16)
                mixer = GatedDeltaNetMLX(
                    d, gdn_h, gdn_hd, gdn_k,
                    qkvg_proj=_tlin(d, 4 * gdn_inner),
                    o_proj=_tlin(gdn_inner, d, subln=use_subln),
                    ab_proj=DenseLinearMLX(
                        mx.array(rng.normal(0.0, 0.02, (2 * gdn_h, d)).astype(np.float16)),
                        mx.array(ab_bias),
                    ),
                    conv_weight=mx.array(rng.normal(0.0, 0.02, (3 * gdn_inner, gdn_k)).astype(np.float16)),
                    conv_bias=mx.zeros((3 * gdn_inner,), dtype=mx.float16),
                    head_norm=mx.ones((gdn_hd,), dtype=mx.float16),
                )

            w_up_u, w_up_s, w_up_b = random_packed_ternary_mlx_2bit((n_exp, d_ff, d), rng, group_size=group_size)
            w_dn_u, w_dn_s, w_dn_b = random_packed_ternary_mlx_2bit((n_exp, d, d_ff), rng, group_size=group_size)
            shared = SquaredReLUMLPMLX(_tlin(d, shared_d_ff), _tlin(shared_d_ff, d, subln=use_subln)) if shared_d_ff > 0 else None
            moe = BitMoEMLX(
                n_exp, top_k,
                router=DenseLinearMLX(mx.array(rng.normal(0.0, 0.02, (n_exp, d)).astype(np.float16))),
                expert_bias=mx.zeros((n_exp,), dtype=mx.float32),
                w_up_u32=mx.array(w_up_u),
                w_up_scales=mx.array(w_up_s),
                w_up_biases=mx.array(w_up_b),
                w_down_u32=mx.array(w_dn_u),
                w_down_scales=mx.array(w_dn_s),
                w_down_biases=mx.array(w_dn_b),
                expert_subln=mx.ones((d_ff,), dtype=mx.float16) if use_subln else None,
                shared_expert=shared,
                group_size=group_size,
            )
            layers.append(BitMoEBlockMLX(mx.ones((d,), dtype=mx.float16), mixer, mx.ones((d,), dtype=mx.float16), moe))

        mtp_heads = [
            MTPHeadMLX(
                h_norm=mx.ones((d,), dtype=mx.float16),
                e_norm=mx.ones((d,), dtype=mx.float16),
                fuse_proj=_tlin(2 * d, d),
                mlp_norm=mx.ones((d,), dtype=mx.float16),
                mlp=SquaredReLUMLPMLX(_tlin(d, mtp_d_ff), _tlin(mtp_d_ff, d, subln=use_subln)),
                out_norm=mx.ones((d,), dtype=mx.float16),
            )
            for _ in range(mtp_k)
        ]
        return cls(
            model_cfg,
            embed_weight=embed_w,
            layers=layers,
            final_norm=mx.ones((d,), dtype=mx.float16),
            mtp_heads=mtp_heads,
            lm_head_bits=lm_head_bits,
        )

    @classmethod
    def from_pytorch_state_dict(
        cls,
        model_cfg: dict[str, Any],
        state_dict: dict[str, Any],
        *,
        group_size: int = 128,
        lm_head_bits: int = 4,
    ) -> "BitMoETransformerMLX":
        """Losslessly convert a trained PyTorch `TelosTransformer` state_dict to fused packed 2-bit MLX."""
        def _np(key: str) -> np.ndarray:
            t = state_dict[key]
            return t.detach().cpu().float().numpy() if hasattr(t, "detach") else np.asarray(t, dtype=np.float32)

        def _f16(key: str) -> mx.array:
            return mx.array(_np(key).astype(np.float16))

        def _tlin(prefix: str) -> TernaryLinearMLX:
            w, s, b = pack_ternary_to_mlx_2bit(_np(f"{prefix}.weight"), group_size=group_size)
            sn_key = f"{prefix}.subln.weight"
            sn = _f16(sn_key) if sn_key in state_dict else None
            return TernaryLinearMLX(mx.array(w), mx.array(s), mx.array(b), subln_weight=sn, group_size=group_size)

        d = int(model_cfg["d_model"])
        n_layers = int(model_cfg["n_layers"])
        n_heads = int(model_cfg["n_heads"])
        n_kv = int(model_cfg.get("n_kv_heads", n_heads))
        swa_every = int(model_cfg.get("hybrid_swa_every", 4))
        window = int(model_cfg.get("sliding_window", 512))
        gdn_h = int(model_cfg.get("gdn_n_heads", 8))
        gdn_hd = int(model_cfg.get("gdn_head_dim", d // gdn_h))
        gdn_k = int(model_cfg.get("gdn_conv_kernel", 4))
        n_exp = int(model_cfg.get("moe_num_experts", 24))
        top_k = int(model_cfg.get("moe_top_k", 2))
        mtp_k = int(model_cfg.get("mtp_num_heads", 3))

        embed_w = _f16("tok_embeddings.weight")
        layers: list[BitMoEBlockMLX] = []

        for i in range(n_layers):
            lp = f"layers.{i}"
            is_swa = ((i + 1) % swa_every == 0)
            if is_swa:
                qkv = TernaryLinearMLX.fuse([
                    _tlin(f"{lp}.attn.q_proj"),
                    _tlin(f"{lp}.attn.k_proj"),
                    _tlin(f"{lp}.attn.v_proj"),
                ])
                mixer: nn.Module = SlidingWindowAttentionMLX(
                    d, n_heads, n_kv, window,
                    qkv_proj=qkv,
                    o_proj=_tlin(f"{lp}.attn.o_proj"),
                )
            else:
                qkvg = TernaryLinearMLX.fuse([
                    _tlin(f"{lp}.attn.q_proj"),
                    _tlin(f"{lp}.attn.k_proj"),
                    _tlin(f"{lp}.attn.v_proj"),
                    _tlin(f"{lp}.attn.g_proj"),
                ])
                ab_w = np.concatenate((_np(f"{lp}.attn.alpha_proj.weight"), _np(f"{lp}.attn.beta_proj.weight")), axis=0).astype(np.float16)
                ab_b = np.concatenate((_np(f"{lp}.attn.alpha_proj.bias"), _np(f"{lp}.attn.beta_proj.bias")), axis=0).astype(np.float16)
                # PyTorch Conv1d weight is (C, 1, K) -> (C, K)
                cw = _np(f"{lp}.attn.short_conv.weight")[:, 0, :].astype(np.float16)
                mixer = GatedDeltaNetMLX(
                    d, gdn_h, gdn_hd, gdn_k,
                    qkvg_proj=qkvg,
                    o_proj=_tlin(f"{lp}.attn.o_proj"),
                    ab_proj=DenseLinearMLX(mx.array(ab_w), mx.array(ab_b)),
                    conv_weight=mx.array(cw),
                    conv_bias=_f16(f"{lp}.attn.short_conv.bias"),
                    head_norm=_f16(f"{lp}.attn.head_norm.weight"),
                )

            w_up_u, w_up_s, w_up_b = pack_ternary_to_mlx_2bit(_np(f"{lp}.mlp.w_up"), group_size=group_size)
            w_dn_u, w_dn_s, w_dn_b = pack_ternary_to_mlx_2bit(_np(f"{lp}.mlp.w_down"), group_size=group_size)
            sh_key = f"{lp}.mlp.shared_expert.w_up.weight"
            shared = (
                SquaredReLUMLPMLX(_tlin(f"{lp}.mlp.shared_expert.w_up"), _tlin(f"{lp}.mlp.shared_expert.w_down"))
                if sh_key in state_dict
                else None
            )
            subln_key = f"{lp}.mlp.expert_subln.weight"
            moe = BitMoEMLX(
                n_exp, top_k,
                router=DenseLinearMLX(_f16(f"{lp}.mlp.router.weight")),
                expert_bias=mx.array(_np(f"{lp}.mlp.expert_bias").astype(np.float32)),
                w_up_u32=mx.array(w_up_u),
                w_up_scales=mx.array(w_up_s),
                w_up_biases=mx.array(w_up_b),
                w_down_u32=mx.array(w_dn_u),
                w_down_scales=mx.array(w_dn_s),
                w_down_biases=mx.array(w_dn_b),
                expert_subln=_f16(subln_key) if subln_key in state_dict else None,
                shared_expert=shared,
                group_size=group_size,
            )
            layers.append(BitMoEBlockMLX(_f16(f"{lp}.attn_norm.weight"), mixer, _f16(f"{lp}.mlp_norm.weight"), moe))

        mtp_heads = [
            MTPHeadMLX(
                h_norm=_f16(f"mtp.heads.{k}.h_norm.weight"),
                e_norm=_f16(f"mtp.heads.{k}.e_norm.weight"),
                fuse_proj=_tlin(f"mtp.heads.{k}.fuse_proj"),
                mlp_norm=_f16(f"mtp.heads.{k}.mlp_norm.weight"),
                mlp=SquaredReLUMLPMLX(_tlin(f"mtp.heads.{k}.mlp.w_up"), _tlin(f"mtp.heads.{k}.mlp.w_down")),
                out_norm=_f16(f"mtp.heads.{k}.out_norm.weight"),
            )
            for k in range(mtp_k)
        ]
        return cls(
            model_cfg,
            embed_weight=embed_w,
            layers=layers,
            final_norm=_f16("final_norm.weight"),
            mtp_heads=mtp_heads,
            lm_head_bits=lm_head_bits,
        )
