"""Metal projection/CE with threadgroup-local logits and a recomputed VJP."""

from functools import lru_cache

import mlx.core as mx


_SIMD_BLOCK = (32, 64)
_TENSOR_BLOCK = (64, 128)


# Four SIMD groups compute a 32 x 64 projection tile. The same projection
# code is used in both passes so low-precision logits round identically.
_HEADER = r"""
#include <metal_simdgroup_matrix>

template <uint D, uint V, uint R, typename X, typename W>
inline void project_tile(
    X x, W w, threadgroup float* a,
    threadgroup float* b, threadgroup float* scores,
    uint row, uint col, uint tid, uint simd
) {
    simdgroup_float8x8 accumulator[4][2];
    for (uint r = 0; r < 4; ++r)
        for (uint v = 0; v < 2; ++v)
            accumulator[r][v] = simdgroup_float8x8(0);
    for (uint start = 0; start < D; start += 8) {
        for (uint i = tid; i < 32 * 8; i += 128) {
            uint r = row + i / 8;
            uint k = start + i % 8;
            a[i] = r < R && k < D ? float(x[r * D + k]) : 0.0f;
        }
        for (uint i = tid; i < 64 * 8; i += 128) {
            uint v = col + i / 8;
            uint k = start + i % 8;
            b[i] = v < V && k < D ? float(w[v * D + k]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 rhs[2];
        for (uint v = 0; v < 2; ++v)
            simdgroup_load(rhs[v], b + (simd * 2 + v) * 64, 8, ulong2(0), true);
        for (uint r = 0; r < 4; ++r) {
            simdgroup_float8x8 lhs;
            simdgroup_load(lhs, a + r * 64, 8);
            for (uint v = 0; v < 2; ++v)
                simdgroup_multiply_accumulate(accumulator[r][v], lhs, rhs[v], accumulator[r][v]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint r = 0; r < 4; ++r)
        for (uint v = 0; v < 2; ++v)
            simdgroup_store(accumulator[r][v], scores + r * 8 * 64 + simd * 16 + v * 8, 64);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
"""

_PROJECT = r"""
    uint row = threadgroup_position_in_grid.x * 32;
    uint col = threadgroup_position_in_grid.y * 64;
    uint tid = thread_position_in_threadgroup.x;
    uint simd = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    threadgroup float a[256], b[512], scores[2048];
    project_tile<D, V, R>(hidden, weight, a, b, scores, row, col, tid, simd);
"""

_STATS = r"""
    for (uint r = simd; r < BM && row + r < R; r += 4) {
        uint target = targets[row + r];
        float maximum = -INFINITY;
        float non_target[BN / 32];
        for (uint i = 0; i < BN / 32; ++i) {
            uint v = col + lane + i * 32;
            float score = v < V ? float(T(scores[r * BN + lane + i * 32])) : 0.0f;
            if (v == target) selected[row + r] = score;
            non_target[i] = v < V && v != target ? score : -INFINITY;
            maximum = max(maximum, non_target[i]);
        }
        maximum = simd_max(maximum);
        float total = 0.0f;
        if (isfinite(maximum))
            for (uint i = 0; i < BN / 32; ++i)
                total += exp(non_target[i] - maximum);
        total = simd_sum(total);
        if (lane == 0) {
            uint i = (row + r) * ((V + BN - 1) / BN) + col / BN;
            maxima[i] = maximum;
            totals[i] = total;
        }
    }
"""

_forward = mx.fast.metal_kernel(
    name="telos_fused_projection_stats",
    input_names=["hidden", "weight", "targets"],
    output_names=["maxima", "totals", "selected"],
    header=_HEADER,
    source=_PROJECT + _STATS,
)

_backward = mx.fast.metal_kernel(
    name="telos_fused_projection_vjp",
    input_names=["hidden", "weight", "targets", "target_scores", "loss", "cotangent", "target_gradient"],
    output_names=["hidden_grad", "weight_grad"],
    atomic_outputs=True,
    header=_HEADER,
    source=_PROJECT + r"""
        for (uint i = tid; i < 2048; i += 128) {
            uint r = row + i / 64;
            uint v = col + i % 64;
            float gradient = 0.0f;
            if (r < R && v < V) {
                float logp = (float(T(scores[i])) - target_scores[r]) - loss[r];
                float value = v == uint(targets[r]) ? target_gradient[r] : exp(logp) * cotangent[r];
                gradient = float(T(value));
            }
            scores[i] = gradient;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Each SIMD group owns 16 vocabulary rows for the weight gradient.
        // Sum over all 32 token rows before writing to global memory.
        threadgroup float inputs[4 * 64], output[4 * 64];
        threadgroup float* tile = inputs + simd * 64;
        threadgroup float* result = output + simd * 64;
        uint v0 = col + simd * 16;
        for (uint k = 0; k < D; k += 8) {
            simdgroup_float8x8 product[2];
            product[0] = simdgroup_float8x8(0);
            product[1] = simdgroup_float8x8(0);
            for (uint j = 0; j < 32; j += 8) {
                for (uint i = lane; i < 64; i += 32) {
                    uint r = row + j + i / 8;
                    uint d = k + i % 8;
                    tile[i] = r < R && d < D ? float(hidden[r * D + d]) : 0.0f;
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
                simdgroup_float8x8 x;
                simdgroup_load(x, tile, 8);
                for (uint v = 0; v < 2; ++v) {
                    simdgroup_float8x8 grad;
                    simdgroup_load(grad, scores + j * 64 + simd * 16 + v * 8, 64, ulong2(0), true);
                    simdgroup_multiply_accumulate(product[v], grad, x, product[v]);
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
            for (uint block = 0; block < 2; ++block) {
                simdgroup_store(product[block], result, 8);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                for (uint i = lane; i < 64; i += 32) {
                    uint v = v0 + block * 8 + i / 8;
                    uint d = k + i % 8;
                    if (v < V && d < D)
                        atomic_fetch_add_explicit(&weight_grad[v * D + d], result[i], memory_order_relaxed);
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        // Each SIMD group owns a feature slice for the hidden gradient.
        // Sum over all 64 vocabulary rows before writing to global memory.
        for (uint k = simd * 8; k < D; k += 32) {
            simdgroup_float8x8 product[4];
            for (uint r = 0; r < 4; ++r)
                product[r] = simdgroup_float8x8(0);
            for (uint j = 0; j < 64; j += 8) {
                for (uint i = lane; i < 64; i += 32) {
                    uint v = col + j + i / 8;
                    uint d = k + i % 8;
                    tile[i] = v < V && d < D ? float(weight[v * D + d]) : 0.0f;
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
                simdgroup_float8x8 w;
                simdgroup_load(w, tile, 8);
                for (uint r = 0; r < 4; ++r) {
                    simdgroup_float8x8 grad;
                    simdgroup_load(grad, scores + r * 8 * 64 + j, 64);
                    simdgroup_multiply_accumulate(product[r], grad, w, product[r]);
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
            for (uint block = 0; block < 4; ++block) {
                simdgroup_store(product[block], result, 8);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                for (uint i = lane; i < 64; i += 32) {
                    uint r = row + block * 8 + i / 8;
                    uint d = k + i % 8;
                    if (r < R && d < D)
                        atomic_fetch_add_explicit(&hidden_grad[r * D + d], result[i], memory_order_relaxed);
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    """,
)

# Tensor operations on M5 read projection inputs from device memory. The
# rounded output stays in threadgroup memory and becomes the logits gradient
# in place during backward. Both gradient products reduce whole tiles before
# writing FP32 atomics.
_TENSOR_HEADER = r"""
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
"""

_TENSOR_PROJECT = r"""
    uint row = threadgroup_position_in_grid.x * BM;
    uint col = threadgroup_position_in_grid.y * BN;
    uint tid = thread_position_in_threadgroup.x;
    uint simd = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    threadgroup T scores[BM * BN];

    constexpr auto descriptor = matmul2d_descriptor(BM / 2, BN / 2, D, false, true);
    matmul2d<descriptor, execution_simdgroup> operation;
    uint x0 = (simd % 2) * (BN / 2);
    uint y0 = (simd / 2) * (BM / 2);
    if (row + y0 < R && col + x0 < V) {
        // MPP's cooperative tensor API expects an unqualified element type.
        // These input tensor views are only used as multiplication operands.
        auto x = tensor(const_cast<device T*>(hidden) + (row + y0) * D,
                        dextents<int, 2>{D, min(int(R - row - y0), BM / 2)}, array<int, 2>{1, D});
        auto w = tensor(const_cast<device T*>(weight) + (col + x0) * D,
                        dextents<int, 2>{D, min(int(V - col - x0), BN / 2)}, array<int, 2>{1, D});
        auto z = tensor(scores + y0 * BN + x0, extents<int, BN / 2, BM / 2>{}, array<int, 2>{1, BN});
        auto result = operation.template get_destination_cooperative_tensor<decltype(x), decltype(w), T>();
        operation.run(x, w, result);
        result.store(z);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
"""

_TENSOR_ACCUMULATE = r"""
template <int M, int N, int K, bool TransposeA, typename A, typename B>
inline void accumulate_gradient(
    A a, B b, device atomic<float>* out, uint row, uint col, uint rows, uint width
) {
    constexpr auto shape = matmul2d_descriptor(M, N, K, TransposeA, false);
    matmul2d<shape, execution_simdgroup> operation;
    auto result = operation.template get_destination_cooperative_tensor<A, B, float>();
    operation.run(a, b, result);
    for (uint i = 0; i < result.get_capacity(); ++i) {
        if (!result.is_valid_element(i)) continue;
        auto location = result.get_multidimensional_index(i);
        uint r = row + location[1];
        uint d = col + location[0];
        if (r < rows && d < width)
            atomic_fetch_add_explicit(out + r * width + d, result[i], memory_order_relaxed);
    }
}
"""

_tensor_forward = mx.fast.metal_kernel(
    name="telos_fused_wide_projection_stats",
    input_names=["hidden", "weight", "targets"],
    output_names=["maxima", "totals", "selected"],
    header=_TENSOR_HEADER,
    source=_TENSOR_PROJECT + _STATS,
)

_tensor_backward = mx.fast.metal_kernel(
    name="telos_fused_wide_projection_vjp",
    input_names=["hidden", "weight", "targets", "target_scores", "loss", "cotangent", "target_gradient"],
    output_names=["hidden_grad", "weight_grad"],
    atomic_outputs=True,
    header=_TENSOR_HEADER + _TENSOR_ACCUMULATE,
    source=_TENSOR_PROJECT + r"""
        threadgroup float row_tscore[BM], row_loss[BM], row_cot[BM], row_tgrad[BM];
        threadgroup uint row_target[BM];
        if (tid < BM && row + tid < R) {
            uint r_abs = row + tid;
            row_target[tid] = uint(targets[r_abs]);
            row_tscore[tid] = target_scores[r_abs];
            row_loss[tid] = loss[r_abs];
            row_cot[tid] = cotangent[r_abs];
            row_tgrad[tid] = target_gradient[r_abs];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < BM * BN; i += 128) {
            uint lr = i / BN;
            uint r = row + lr;
            uint v = col + i % BN;
            float value = 0.0f;
            if (r < R && v < V) {
                float logp = (float(scores[i]) - row_tscore[lr]) - row_loss[lr];
                value = v == row_target[lr] ? row_tgrad[lr] : exp(logp) * row_cot[lr];
            }
            scores[i] = T(value);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto dz = tensor(scores, extents<int, BN, BM>{}, array<int, 2>{1, BN});
        for (uint k = simd * 32; k < D; k += 128) {
            auto w = tensor(const_cast<device T*>(weight) + col * D + k,
                            dextents<int, 2>{min(int(D - k), 32), min(int(V - col), BN)},
                            array<int, 2>{1, D});
            accumulate_gradient<BM, 32, BN, false>(dz, w, hidden_grad, row, k, R, D);
        }
        uint v0 = col + simd * (BN / 4);
        if (v0 < V) {
            auto slice = tensor(scores + simd * (BN / 4), extents<int, BN / 4, BM>{}, array<int, 2>{1, BN});
            for (uint k = 0; k < D; k += 32) {
                auto h = tensor(const_cast<device T*>(hidden) + row * D + k,
                                dextents<int, 2>{min(int(D - k), 32), min(int(R - row), BM)},
                                array<int, 2>{1, D});
                accumulate_gradient<BN / 4, 32, BM, true>(slice, h, weight_grad, v0, k, V, D);
            }
        }
    """,
)

_merge = mx.fast.metal_kernel(
    name="telos_fused_projection_merge_statistics",
    input_names=["maxima", "totals", "target_scores"],
    output_names=["loss"],
    source=r"""
        uint row = threadgroup_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x;
        uint simd = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        threadgroup float partial[4];
        threadgroup float row_max;
        float maximum = -INFINITY;
        for (uint tile = tid; tile < TILES; tile += 128)
            maximum = max(maximum, maxima[row * TILES + tile]);
        maximum = simd_max(maximum);
        if (lane == 0) partial[simd] = maximum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float largest = max(max(partial[0], partial[1]), max(partial[2], partial[3]));
            row_max = isfinite(largest) ? largest : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total = 0.0f;
        for (uint tile = tid; tile < TILES; tile += 128) {
            uint i = row * TILES + tile;
            total += totals[i] * exp(maxima[i] - row_max);
        }
        total = simd_sum(total);
        if (lane == 0) partial[simd] = total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            // Exclude the target before reducing so tiny positive CE values
            // survive rounding, including the one-class case (sum == 0).
            float sum = (partial[0] + partial[1]) + (partial[2] + partial[3]);
            float delta = log(sum) + (row_max - target_scores[row]);
            float z = exp(-abs(delta));
            float residual = z < 1e-3f ? z * (1.0f - 0.5f * z) : log(1.0f + z);
            loss[row] = max(0.0f, delta) + residual;
        }
    """,
)


def _launch(hidden, weight, block):
    rows, width = hidden.shape
    vocab = weight.shape[0]
    m, n = block
    row_groups = (rows + m - 1) // m
    vocab_groups = (vocab + n - 1) // n
    return dict(
        template=[("T", hidden.dtype), ("D", width), ("V", vocab), ("R", rows), ("BM", m), ("BN", n)],
        grid=(row_groups * 128, vocab_groups, 1),
        threadgroup=(128, 1, 1),
    )


def _project_loss(hidden, weight, targets, kernel, block):
    rows = hidden.shape[0]
    columns = block[1]
    shape = (rows, (weight.shape[0] + columns - 1) // columns)
    maxima, totals, target_score = kernel(
        inputs=[hidden, weight, targets], output_shapes=[shape, shape, (rows,)],
        output_dtypes=[mx.float32] * 3, **_launch(hidden, weight, block),
    )
    loss, = _merge(
        inputs=[maxima, totals, target_score], template=[("TILES", shape[1])],
        grid=(rows * 128, 1, 1), threadgroup=(128, 1, 1),
        output_shapes=[(rows,)], output_dtypes=[mx.float32],
    )
    return loss, mx.stop_gradient(target_score)


def _projection_vjp(primals, cotangents, outputs, kernel, block):
    hidden, weight, targets = primals
    loss, target_score = outputs
    cotangent = cotangents[0]
    # The target contribution is exp(-loss) - 1. Its series near zero
    # preserves tiny gradients on MLX runtimes with approximate expm1.
    target_gradient = mx.where(loss < 1e-3, -loss * (1.0 - 0.5 * loss), mx.expm1(-loss)) * cotangent
    dh, dw = kernel(
        inputs=[hidden, weight, targets, target_score, loss, cotangent, target_gradient],
        output_shapes=[hidden.shape, weight.shape], output_dtypes=[mx.float32] * 2,
        init_value=0, **_launch(hidden, weight, block),
    )
    return dh.astype(hidden.dtype), dw.astype(weight.dtype), mx.zeros_like(targets)


@mx.custom_function
def _simd_loss(hidden, weight, targets):
    return _project_loss(hidden, weight, targets, _forward, _SIMD_BLOCK)


@_simd_loss.vjp
def _simd_vjp(primals, cotangents, outputs):
    return _projection_vjp(primals, cotangents, outputs, _backward, _SIMD_BLOCK)


@mx.custom_function
def _tensor_loss(hidden, weight, targets):
    return _project_loss(hidden, weight, targets, _tensor_forward, _TENSOR_BLOCK)


@_tensor_loss.vjp
def _tensor_vjp(primals, cotangents, outputs):
    return _projection_vjp(primals, cotangents, outputs, _tensor_backward, _TENSOR_BLOCK)


@lru_cache(maxsize=1)
def _has_tensor_accelerator():
    device_info = getattr(mx, "device_info", None)
    if device_info is None:
        return False
    architecture = device_info(mx.gpu).get("architecture", "")
    return architecture.startswith("applegpu_g17")


def fused_projection_cross_entropy(hidden, weight, targets):
    """Return FP32 per-token CE for valid class IDs on an MLX Metal device.

    Supports first-order FP32/BF16 gradients. FP32 atomic sums in backward
    may differ slightly between runs. Gradients and tile statistics are the
    global outputs, while logits and logits gradients live in threadgroups.
    """
    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        raise ValueError("Fused projection cross entropy requires an MLX Metal GPU")
    if hidden.ndim < 2 or weight.ndim != 2 or hidden.shape[-1] != weight.shape[-1]:
        raise ValueError("hidden and weight projection dimensions must match")
    if targets.shape != hidden.shape[:-1]:
        raise ValueError("targets must match hidden token dimensions")
    if hidden.dtype not in (mx.float32, mx.bfloat16) or weight.dtype != hidden.dtype:
        raise ValueError("fused projection requires matching FP32 or BF16 hidden and weights")
    if not mx.issubdtype(targets.dtype, mx.integer):
        raise ValueError("targets must contain integer class indices")
    if not weight.size or not targets.size:
        raise ValueError("fused projection requires nonempty weights and targets")
    width = weight.shape[1]
    use_tensor = width % 32 == 0 and _has_tensor_accelerator()
    function = _tensor_loss if use_tensor else _simd_loss
    losses, _ = function(
        hidden.reshape(-1, weight.shape[1]), weight, targets.reshape(-1).astype(mx.int32)
    )
    return losses.reshape(targets.shape)
