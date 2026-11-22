#!/usr/bin/env python3
"""M5 Pro ABBA screening harness for v1.8.6 MLX/Metal pretraining candidates.

Run locally on Apple Silicon outside any Metal-blocking sandbox:
    .venv/bin/python scratch/exp_v186_mlx_candidates.py
"""
import time
import json
import mlx.core as mx
import mlx.nn as nn

from telos.training.mlx_losses import cross_entropy


def _bench_ce(rows=2040, vocab=8192, dtype=mx.bfloat16, warmup=5, iters=30):
    mx.random.seed(42)
    logits = mx.random.normal((rows, vocab)).astype(dtype)
    targets = mx.random.randint(0, vocab, (rows,)).astype(mx.int32)
    mx.eval(logits, targets)

    def fn(z):
        return cross_entropy(z, targets, reduction="mean")

    vg = mx.value_and_grad(fn)
    for _ in range(warmup):
        l, g = vg(logits)
        mx.eval(l, g)
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        l, g = vg(logits)
        mx.eval(l, g)
    mx.synchronize()
    dt_ms = (time.perf_counter() - t0) / iters * 1000.0

    # Exact FP32 reference check
    ref = nn.losses.cross_entropy(logits.astype(mx.float32), targets, reduction="mean")
    mx.eval(ref)
    return {
        "rows": rows,
        "vocab": vocab,
        "dtype": str(dtype),
        "fwd_bwd_ms": round(dt_ms, 3),
        "loss_diff_vs_fp32_ref": float(mx.abs(l - ref).item()),
    }


def main():
    report = {
        "mlx_version": mx.__version__,
        "device": mx.device_info(),
        "metal_ce_v186_bf16": _bench_ce(dtype=mx.bfloat16),
        "metal_ce_v186_fp32": _bench_ce(dtype=mx.float32),
    }
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
