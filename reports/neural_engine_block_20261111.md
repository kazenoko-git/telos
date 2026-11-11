# Neural Engine full-block probe — 2026-11-11

A complete 512-wide, 8-head, 256-token transformer block was compared on M5 Pro at batches 1, 4, and 8. It includes RMSNorm, RoPE, causal attention and SwiGLU. Core ML was set to CPU plus Neural Engine; MLX ran on the GPU. FP16 projections/MLP/RoPE use FP32 RMS and attention accumulation.

Each mode had 10 warmups and 30 measured passes, repeated in alternating order. “Refresh” supplied current FP32 master-weight arrays on every Core ML call; the block weights were held numerically constant for latency measurement. The change-projection check confirms changed weights affect the output without recompiling. This is a forward benchmark, not training.

| Batch | MLX resident | MLX host round trip | Core ML cached weights | Core ML refreshed weights |
|---:|---:|---:|---:|---:|
| 1 | 0.529 ms | 0.322 ms | 0.955 ms | 2.238 ms |
| 4 | 0.596 ms | 0.635 ms | 4.990 ms | 6.216 ms |
| 8 | 1.033 ms | 1.130 ms | 8.843 ms | 10.060 ms |

The refreshed Core ML path was roughly 4–9× slower than the resident MLX block across these batches. At batch 8, refreshing weights added about 0.66 ms over cached weights; the complete refreshed block took 10.06 ms versus 1.03 ms for MLX resident. So handoff contributes, but does not explain the full gap. The compute plan preferred the Neural Engine for most projections and SwiGLU operations; it placed RMSNorm and FP32 attention operations on the CPU. These are estimates, not measured device occupancy.

Memory stayed bounded: largest child RSS was 496 MiB, largest MLX peak was 81 MiB, and swap did not grow.

## Numerical limits

The dynamic-weight path had about 0.83% relative L2 error versus MLX, while its CPU-only control had about 0.32% error at batch 1 and about 13.7% at batches 4 and 8. The fixed-weight CPU control also had high error. That instability means this export has not established reliable numerical parity, so it is not ready for training integration. Placement and timing results are exploratory.

The earlier tiny linear probe was much faster than this full block. Larger batches do not amortize the Core ML path enough to make this export faster. That supports the hypothesis that the tiny-test result overstated ANE benefit, while this probe cannot rule out a more suitable Core ML graph or direct ANE implementation.

Reproduce with coremltools 9.0 available on `PYTHONPATH` and run `scripts/probe_neural_engine_block.py --batch 1|4|8 --output <path>` as three separate processes. The full operation-placement and timing records are in `logs/neural_engine_block_20261111/summary.json`.
