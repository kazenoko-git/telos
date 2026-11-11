# MLX training optimization results — 2026-11-11

Reference workload: 50M autoregressive model, real corpus, MBS 8, accumulation 8, sequence 256, M5 Pro. Sustained warmed throughput was about **37.9k tokens/s**. Individual short-run baselines varied, so each result below is compared only with its matched experiment; do not compare absolute rates across different run lengths or instrumentation modes.

| Optimization or test | Measured throughput change | RAM / status |
|---|---:|---|
| Defer materializing the final accumulated gradient | **+0.29%** (37,613 → 37,721 tok/s) | Peak MLX +0.049 GiB (1.725 → 1.774). Accepted and enabled; parity tests passed. |
| One-microbatch lookahead during accumulation | **−13.60%** (29,021 → 25,074 tok/s) | No peak change (1.925 GiB). Baseline drift was high; no win established. |
| Compile gradient additions alone | **−2.36%** (37,598 → 36,710 tok/s) | No peak change (1.724 GiB). Not adopted. |
| Tiled/fused vocabulary projection plus loss, chunks 256/1,024/8,192 | **−12% to −17%** vs standard projection + Metal CE (38.1k baseline; candidates 31.7k–33.6k) | Peak 1.78–1.86 GiB vs 1.72 GiB. Remains experimental/off. |
| Increase microbatch 4 → 8 at fixed 16,384 tokens/update, sequence 256 | **+7–10%** (34.8k → 37.1–38.2k tok/s) | Peak +0.52 GiB (1.20 → 1.72). MBS16 at 2.75 GiB cap reached 37.4k tok/s, no win, at 2.63 GiB MLX peak (trainer displayed 2.82 GB). |
| Shorten sequence 256 → 128 at fixed tokens/update and 2,048 tokens/microbatch | **+5–9%** (37.1–38.2k → 40.2–40.3k tok/s) | Peak 1.66 GiB. Changes model context/training examples, so not behavior-preserving. |
| Lengthen sequence 256 → 512 at same token shapes | **−12–15%** (to 32.6k tok/s) | Peak 1.88 GiB. Preserves token count per update but changes context. |
| Keep the per-microbatch loss/gradient function compiled | **about +9.3%** vs bypassing compilation (compiled 36.9–37.2k; uncompiled 33.9k tok/s) | Peak was about 80–87 MiB lower with compilation. Keep compiled. `fused_step` itself does not change the accumulated MBS8 path. |
| Compile backward together with gradient accumulation | Short retry: **+0.8%** (38.82k → 39.13k), within run variation. A longer candidate run stopped at update 22 when swap fell below its guard. | Retry peak 1.69–1.70 GiB, effectively equal. Six parity tests passed (FP32/BF16, GA2/8, optimizer/master state, LR changes, RNG). Opt-in only; no confirmed speedup. |
| Force MLX fused SDPA attention | Full-model pairs disagreed: **+12%** in one, **−1%** in the other. Four-run means: 36.29k vs 34.58k tok/s, inconclusive. | Full-model peak 1.64 vs 1.67 GiB. Block benchmark at batch 8 was ~24% faster, with 0.314 → 0.248 GiB allocation and tiny output/gradient differences. No default change. |
| Skip unused final hidden position in causal AR | **−5.8%** in one short run (37.59k → 35.40k tok/s). | Peak −0.023 GiB. No performance win; remains diagnostic-only. |
| Use Core ML / Neural Engine for a transformer block | **4–9× slower** than resident MLX forward at batches 1/4/8; refreshed-weight batch 8 was 10.06 ms vs MLX 1.03 ms. | Not integrated. Placement is estimated; numerical parity was unstable. |

## What this says

The only adopted speed change with a measured end-to-end gain is deferring the final accumulation wait, at **+0.29%**. Keeping microbatch compilation is clearly beneficial compared with bypassing it, but it was already the normal path. Larger microbatches help until RAM becomes the limiting constraint. The other experimental code paths either lost, varied too much to establish a win, or changed training context/numerics enough to remain optional.

The fused-attention block result did not carry consistently through full training. The new compiled-backward accumulation path passed small-model behavior checks but has no convincing full-model throughput improvement yet. Neither is enabled by default.

Profiling also found about **236 encoded command buffers per optimizer update**. Existing command-buffer timeline intervals covered 99.7% of a 4.46-second trace window, with a median gap of about 4 microseconds. This suggests little dispatch-timeline downtime, but it is not a direct GPU busy-time measure and does not itself produce a speedup.

## Evidence and limitations

The detailed measurements, full run logs, stage timings, counters, memory limits, and caveats are in [the GPU profile](mlx_gpu_profile_20261111.md), [accumulation report](mlx_accumulation_20261111.md), [shape data](mlx_shapes_20261111.json), [compiled accumulation retry data](mlx_compiled_accumulation_retry_20261111.json), and [forced-attention data](mlx_forced_fused_attention_20261111.json). Most end-to-end candidates used short runs; they do not establish convergence equivalence or sustained thermal performance. Small floating-point differences can alter later training trajectories even when loss and gradient checks are close.

## Training options exposed from the report

Final-gradient materialization deferral remains active in the normal accumulated training path. The compiled-backward accumulation option is available as `--compiled-accumulation`; it remains opt-in because its measured 0.8% retry difference is within run variation. Forced fused SDPA is available as `--force-fused-attention`, also off by default: it was faster in a block benchmark, but full-model pairs were inconsistent, so its throughput benefit is unconfirmed. Neither option changes checkpoint parameter shapes. Forced SDPA and packed SwiGLU can produce small floating-point differences.
