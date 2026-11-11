# Opt-in ANE linear trainer probe — 2026-11-11

`scripts/train_mlx_ane_linear.py` adds an isolated trainer for one 512x512 linear layer. Choose `--mode gpu`, `--mode ane`, or `--mode pipeline`; no paradigm trainer or model architecture is changed. The ANE modes compile one dynamic-weight Core ML projection and inspect its compute plan. This exact shape's plan preferred `MLNeuralEngineComputeDevice` for the casts and matrix multiply. Placement is still a plan estimate, not direct hardware occupancy.

All modes used the same seeded synthetic regression batches, FP32 master matrix, FP16 forward precision, four-microbatch accumulation, and AdamW update after each full accumulation window. In pipeline mode a single worker-prefetched ANE forward for the next microbatch while MLX computed the current microbatch's analytic input/weight gradients. The weights stayed frozen within the accumulation window. No optimizer update crossed a microbatch boundary.

A short six-update check excluded the first two updates:

| Mode | Median update time | Updates/s | Peak MLX GiB | Final loss |
|---|---:|---:|---:|---:|
| GPU forward + backward | 3.61 ms | 277.1 | 0.0125 | 0.00138497 |
| Serial ANE forward + GPU backward | 3.52 ms | 284.1 | 0.0127 | 0.00138504 |
| One-item ANE/GPU pipeline | 3.85 ms | 259.8 | 0.0117 | 0.00138504 |

The serial ANE mode was 2.5% faster than GPU-only in this tiny probe; the pipeline was 6.7% slower. Only four warmed updates were timed, so the serial result is not a robust win. Final weights in both ANE modes differed from GPU-only by at most 0.000123. Peak MLX allocation stayed near 13 MiB; this excludes Core ML allocations.

This is an opt-in single-layer training probe, not a hook into the AR/MDLM/COROSred trainers. It does not establish full-model benefit, exact optimizer-state parity, or measured simultaneous ANE/GPU utilization. Run with `PYTHONPATH=/tmp/telos-ane-deps:. .venv/bin/python scripts/train_mlx_ane_linear.py --mode pipeline --rows 256 --accumulation 4 --updates 6 --output <path>`. Core ML Tools must be installed separately. Results and final weight files are in `logs/mlx_ane_linear_trainer_20261111/`.
