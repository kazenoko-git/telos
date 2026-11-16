"""Experimental tiled tied-head projection/CE with a recomputed first-order VJP.

Vocabulary tiles recompute projection work during backward; actual allocation
depends on graph scheduling and must be measured. BF16
partition and target scores come from the same rounded matmul. The logits VJP
subtracts the target in FP32 before rounding, preserving small confident-target
gradients. Tiled hidden-gradient sums can differ from a full GEMM's reductions.
"""
from functools import lru_cache
import mlx.core as mx

_logits_gradient = mx.fast.metal_kernel(
    name="telos_tiled_vocab_gradient",
    input_names=["logits", "targets", "target_score", "loss", "cotangent", "tile_start"],
    output_names=["gradient"],
    source=r"""
        uint idx = thread_position_in_grid.x;
        if (idx >= N) return;
        uint row = idx / V;
        uint col = idx % V;
        float probability = exp((float(logits[idx]) - target_score[row]) - loss[row]);
        float selected = float(targets[row] == uint(tile_start[0]) + col);
        gradient[idx] = T((probability - selected) * cotangent[row]);
    """,
)


_tile_statistics = mx.fast.metal_kernel(
    name="telos_vocab_tile_statistics",
    input_names=["logits", "targets", "tile_start"],
    output_names=["maximum", "total", "selected"],
    source=r"""
        uint row = threadgroup_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x;
        uint lane = thread_index_in_simdgroup;
        uint simd = simdgroup_index_in_threadgroup;
        threadgroup float shared[8];
        float value = -INFINITY;
        for (uint col = tid; col < V; col += 256)
            value = max(value, float(logits[row * V + col]));
        value = simd_max(value);
        if (lane == 0) shared[simd] = value;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        value = simd_max(tid < 8 ? shared[tid] : -INFINITY);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) shared[0] = value;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float largest = shared[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float sum = 0.0f;
        for (uint col = tid; col < V; col += 256)
            sum += exp(float(logits[row * V + col]) - largest);
        sum = simd_sum(sum);
        if (lane == 0) shared[simd] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sum = simd_sum(tid < 8 ? shared[tid] : 0.0f);
        if (tid == 0) {
            maximum[row] = largest;
            total[row] = sum;
            int index = int(targets[row]) - int(tile_start[0]);
            selected[row] = index >= 0 && index < int(V) ? float(logits[row * V + index]) : 0.0f;
        }
    """,
)


@lru_cache(maxsize=16)
def _operation(chunk_size):
    @mx.custom_function
    def loss_and_score(hidden, weight, targets):
        rows = hidden.shape[0]
        maximum = mx.full((rows,), -float("inf"), dtype=mx.float32)
        total = mx.zeros((rows,), dtype=mx.float32)
        selected = mx.zeros((rows,), dtype=mx.float32)
        for start in range(0, weight.shape[0], chunk_size):
            tile = weight[start:start + chunk_size]
            logits = mx.matmul(hidden, tile.T)
            if mx.default_device() == mx.gpu:
                tile_max, tile_sum, score = _tile_statistics(
                    inputs=[logits, targets, mx.array([start], dtype=mx.int32)],
                    template=[("V", tile.shape[0])],
                    grid=(rows * 256, 1, 1), threadgroup=(256, 1, 1),
                    output_shapes=[(rows,), (rows,), (rows,)], output_dtypes=[mx.float32] * 3,
                )
            else:
                logits = logits.astype(mx.float32)
                tile_max = mx.max(logits, axis=-1)
                tile_sum = mx.sum(mx.exp(logits - tile_max[:, None]), axis=-1)
                local = mx.clip(targets - start, 0, tile.shape[0] - 1)
                score = mx.take_along_axis(logits, local[:, None], axis=-1).squeeze(-1)
                score = mx.where((targets >= start) & (targets < start + tile.shape[0]), score, 0.)
            next_max = mx.maximum(maximum, tile_max)
            total = total * mx.exp(maximum - next_max) + tile_sum * mx.exp(tile_max - next_max)
            maximum = next_max
            selected = selected + score
        losses = mx.log(total) + (maximum - selected)
        # Auxiliary target scores are deliberately nondifferentiable; callers
        # expose only losses. The VJP implements CE directly rather than
        # differentiating separate low-precision partition/target branches.
        return losses, mx.stop_gradient(selected)

    @loss_and_score.vjp
    def backward(primals, cotangents, outputs):
        hidden, weight, targets = primals
        losses, selected = outputs
        cotangent = cotangents[0]
        hidden_gradient = mx.zeros(hidden.shape, dtype=mx.float32)
        weight_gradients = []
        for start in range(0, weight.shape[0], chunk_size):
            tile = weight[start:start + chunk_size]
            logits = mx.matmul(hidden, tile.T)
            if mx.default_device() == mx.gpu:
                gradient = _logits_gradient(
                    inputs=[logits, targets, selected, losses, cotangent, mx.array([start], dtype=mx.int32)],
                    template=[("N", logits.size), ("V", tile.shape[0]), ("T", logits.dtype)],
                    grid=(logits.size, 1, 1), threadgroup=(256, 1, 1),
                    output_shapes=[logits.shape], output_dtypes=[logits.dtype],
                )[0]
            else:
                probability = mx.exp((logits.astype(mx.float32) - selected[:, None]) - losses[:, None])
                indicator = targets[:, None] == mx.arange(start, start + tile.shape[0])[None, :]
                gradient = ((probability - indicator) * cotangent[:, None]).astype(logits.dtype)
            # Preserve the logits-gradient dtype used by the full projection.
            # Summing rounded tile GEMMs in FP32 bounds accumulation drift, but
            # is not a bitwise-equivalence promise for BF16 hidden gradients.
            hidden_gradient = hidden_gradient + mx.matmul(gradient, tile).astype(mx.float32)
            weight_gradients.append(mx.matmul(gradient.T, hidden))
        return hidden_gradient.astype(hidden.dtype), mx.concatenate(weight_gradients, axis=0), mx.zeros_like(targets)

    return loss_and_score


def tiled_projection_cross_entropy(hidden, weight, targets, *, chunk_size=256):
    """Per-token CE; supports first-order FP32/BF16 tied-head training only."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if hidden.ndim < 2 or weight.ndim != 2 or hidden.shape[-1] != weight.shape[-1]:
        raise ValueError("hidden and weight projection dimensions must match")
    if targets.shape != hidden.shape[:-1]:
        raise ValueError("targets must match hidden token dimensions")
    if hidden.dtype not in (mx.float32, mx.bfloat16) or weight.dtype != hidden.dtype:
        raise ValueError("tiled projection requires matching FP32 or BF16 hidden and weights")
    if not mx.issubdtype(targets.dtype, mx.integer):
        raise ValueError("targets must contain integer class indices")
    if weight.shape[0] < 1 or not targets.size:
        raise ValueError("tiled projection requires nonempty vocabulary and targets")
    losses, _ = _operation(chunk_size)(hidden.reshape(-1, hidden.shape[-1]), weight, targets.reshape(-1))
    return losses.reshape(targets.shape)


@lru_cache(maxsize=16)
def _token_operation(chunk_size):
    @mx.custom_function
    def loss(hidden, weight, targets):
        from .mlx_losses import cross_entropy
        values = []
        for start in range(0, hidden.shape[0], chunk_size):
            logits = mx.matmul(hidden[start:start + chunk_size], weight.T)
            values.append(cross_entropy(logits, targets[start:start + chunk_size], reduction="none"))
        return mx.concatenate(values, axis=0)

    @loss.vjp
    def backward(primals, cotangent, output):
        from .mlx_losses import cross_entropy, _metal_cross_entropy_gradient
        hidden, weight, targets = primals
        hidden_gradients = []
        weight_gradient = mx.zeros(weight.shape, dtype=mx.float32)
        for start in range(0, hidden.shape[0], chunk_size):
            inputs = hidden[start:start + chunk_size]
            local_targets = targets[start:start + chunk_size]
            logits = mx.matmul(inputs, weight.T)
            local_cotangent = cotangent[start:start + chunk_size]
            if mx.default_device() == mx.gpu:
                gradient = _metal_cross_entropy_gradient(
                    logits, local_targets, output[start:start + chunk_size], local_cotangent
                )
            else:
                _, (gradient,) = mx.vjp(
                    lambda z: cross_entropy(z, local_targets, reduction="none"),
                    (logits,), (local_cotangent,),
                )
            hidden_gradients.append(mx.matmul(gradient, weight))
            weight_gradient = weight_gradient + mx.matmul(gradient.T, inputs).astype(mx.float32)
        return mx.concatenate(hidden_gradients, axis=0), weight_gradient.astype(weight.dtype), mx.zeros_like(targets)

    return loss


def token_chunked_projection_cross_entropy(hidden, weight, targets, *, chunk_size=256):
    """Per-token CE with whole-vocabulary token tiles and recomputed logits.

    Targets must be valid class IDs. First-order FP32/BF16 training only. Weight
    gradient accumulation differs from a full GEMM's reduction order; it is not
    a bitwise-equivalent training update, particularly in BF16.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if hidden.ndim < 2 or weight.ndim != 2 or hidden.shape[-1] != weight.shape[-1]:
        raise ValueError("hidden and weight projection dimensions must match")
    if targets.shape != hidden.shape[:-1]:
        raise ValueError("targets must match hidden token dimensions")
    if hidden.dtype not in (mx.float32, mx.bfloat16) or weight.dtype != hidden.dtype:
        raise ValueError("token projection requires matching FP32 or BF16 hidden and weights")
    if not mx.issubdtype(targets.dtype, mx.integer):
        raise ValueError("targets must contain integer class indices")
    if weight.shape[0] < 1 or not targets.size:
        raise ValueError("token projection requires nonempty vocabulary and targets")
    return _token_operation(chunk_size)(
        hidden.reshape(-1, hidden.shape[-1]), weight, targets.reshape(-1)
    ).reshape(targets.shape)
