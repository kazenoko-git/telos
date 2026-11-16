"""Experimental residual-add/RMSNorm forward fusion; no production wiring.

Keep the native RMSNorm VJP so this experiment isolates forward dispatch and
memory traffic. First-order autodiff only. The residual is rounded to its input
dtype before normalization, matching the reference graph's arithmetic boundary.
"""
from functools import lru_cache
import mlx.core as mx

_kernel = mx.fast.metal_kernel(
    name="telos_probe_residual_rms_norm",
    input_names=["x", "branch", "weight", "epsilon"],
    output_names=["residual", "normalized"],
    source=r"""
        uint row = threadgroup_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x;
        uint lane = thread_index_in_simdgroup;
        uint simd = simdgroup_index_in_threadgroup;
        threadgroup float partial[32];
        float values[ITEMS];
        float squares = 0.0f;
        for (uint i = 0; i < ITEMS; ++i) {
            uint col = tid * ITEMS + i;
            float value = 0.0f;
            if (col < D) {
                T rounded = T(float(x[row * D + col]) + float(branch[row * D + col]));
                residual[row * D + col] = rounded;
                value = float(rounded);
            }
            values[i] = value;
            squares += value * value;
        }
        squares = simd_sum(squares);
        if (lane == 0) partial[simd] = squares;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        squares = simd_sum(tid < (THREADS / 32) ? partial[tid] : 0.0f);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) partial[0] = metal::precise::rsqrt(squares / float(D) + epsilon[0]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float inverse_rms = partial[0];
        for (uint i = 0; i < ITEMS; ++i) {
            uint col = tid * ITEMS + i;
            if (col < D)
                normalized[row * D + col] = T(float(T(values[i] * inverse_rms)) * float(weight[col]));
        }
    """,
)


@lru_cache(maxsize=8)
def _operation(eps):
    @mx.custom_function
    def operation(x, branch, weight):
        threads = ((x.shape[-1] + 127) // 128) * 32
        outputs = _kernel(
            inputs=[x, branch, weight, mx.array([eps], dtype=mx.float32)],
            template=[("D", x.shape[-1]), ("ITEMS", 4), ("THREADS", threads), ("T", x.dtype)],
            grid=(x.size // x.shape[-1] * threads, 1, 1), threadgroup=(threads, 1, 1),
            output_shapes=[x.shape, x.shape], output_dtypes=[x.dtype, x.dtype],
        )
        return tuple(outputs)

    @operation.vjp
    def backward(primals, cotangents, outputs):
        _, _, weight = primals
        residual, _ = outputs
        residual_cotangent, normalized_cotangent = cotangents
        _, (grad_residual, grad_weight) = mx.vjp(
            lambda value, scale: mx.fast.rms_norm(value, scale, eps),
            (residual, weight), (normalized_cotangent,),
        )
        combined = residual_cotangent + grad_residual
        return combined, combined, grad_weight

    return operation


def residual_norm(x, branch, weight, eps=1e-5):
    if x.shape != branch.shape or weight.shape != (x.shape[-1],):
        raise ValueError("Residual operands and normalization weight must match")
    if (mx.default_device() != mx.gpu or x.dtype not in (mx.float32, mx.bfloat16)
            or branch.dtype != x.dtype or weight.dtype != x.dtype or not x.size
            or x.shape[-1] > 4096):
        residual = x + branch
        return residual, mx.fast.rms_norm(residual, weight, eps)
    return _operation(eps)(x, branch, weight)
