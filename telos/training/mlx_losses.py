"""Memory-efficient Metal cross entropy with FP32 reductions and a fused VJP.

Unlike projection-and-loss fusion this still accepts a materialized logits
matrix. It avoids the vocabulary-sized log-softmax intermediates in MLX 0.32.2
on Metal. Higher order autodiff and vmap are not supported by this kernel.
"""

import mlx.core as mx
import mlx.nn as nn


_forward = mx.fast.metal_kernel(
    name="telos_cross_entropy",
    input_names=["logits", "targets"],
    output_names=["loss"],
    source=r"""
        uint row = threadgroup_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x;
        uint lane = thread_index_in_simdgroup;
        uint simd = simdgroup_index_in_threadgroup;
        threadgroup float shared[8];
        float maximum = -INFINITY;
        for (uint col = tid; col < V; col += 256)
            maximum = max(maximum, float(logits[row * V + col]));
        maximum = simd_max(maximum);
        if (lane == 0) shared[simd] = maximum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        maximum = simd_max(tid < 8 ? shared[tid] : -INFINITY);
        if (tid == 0) shared[0] = maximum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        maximum = shared[0];
        // Ensure every thread has read maximum before reusing shared storage.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total = 0.0f;
        for (uint col = tid; col < V; col += 256)
            total += exp(float(logits[row * V + col]) - maximum);
        total = simd_sum(total);
        if (lane == 0) shared[simd] = total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        total = simd_sum(tid < 8 ? shared[tid] : 0.0f);
        if (tid == 0)
            loss[row] = log(total) + (maximum - float(logits[row * V + targets[row]]));
    """,
)

_backward = mx.fast.metal_kernel(
    name="telos_cross_entropy_grad",
    input_names=["logits", "targets", "loss", "cotangent"],
    output_names=["gradient"],
    source=r"""
        uint idx = thread_position_in_grid.x;
        if (idx >= N) return;
        uint row = idx / V;
        uint col = idx % V;
        // Subtract the target logit first to avoid loss of precision when
        // every logit has a large common offset.
        float shifted = float(logits[idx]) - float(logits[row * V + targets[row]]);
        float probability = exp(shifted - loss[row]);
        gradient[idx] = T((probability - float(col == targets[row])) * cotangent[row]);
    """,
)


@mx.custom_function
def _metal_cross_entropy(logits, targets):
    return _forward(
        inputs=[logits, targets], template=[("V", logits.shape[-1])],
        grid=(targets.size * 256, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[targets.shape], output_dtypes=[mx.float32],
    )[0]


@_metal_cross_entropy.vjp
def _cross_entropy_vjp(primals, cotangent, output):
    logits, targets = primals
    gradient = _backward(
        inputs=[logits, targets, output, cotangent],
        template=[("V", logits.shape[-1]), ("N", logits.size), ("T", logits.dtype)],
        grid=(logits.size, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[logits.shape], output_dtypes=[logits.dtype],
    )[0]
    return gradient, mx.zeros_like(targets)


def cross_entropy(logits, targets, *, reduction="none"):
    """Class-index CE over the last axis; targets must be valid vocabulary IDs.

    Preserves the standard scalar/per-token reduction contract. FP32 loss
    accumulation is intentional even when the model uses BF16 or FP16.
    """
    if targets.shape != logits.shape[:-1]:
        raise ValueError("targets must have the shape of logits without the last axis")
    if not mx.issubdtype(targets.dtype, mx.integer):
        raise ValueError("targets must contain integer class indices")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError(f"Unknown reduction: {reduction}")
    if logits.shape[-1] == 0:
        raise ValueError("logits must have at least one class")
    if mx.metal.is_available() and mx.default_device() == mx.gpu and targets.size:
        losses = _metal_cross_entropy(logits, targets)
    else:
        losses = nn.losses.cross_entropy(logits.astype(mx.float32), targets, reduction="none")
    if reduction == "mean":
        return mx.mean(losses)
    if reduction == "sum":
        return mx.sum(losses)
    return losses
