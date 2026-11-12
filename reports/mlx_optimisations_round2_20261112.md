# MLX training optimisation, round 2 — 2026-11-12

The useful new result is **19.3% lower peak MLX allocation at similar
throughput** by reducing command-buffer limits to 16 operations / 16 MiB on the
existing fast path. Mean peak fell from 1.581 to
1.276 GiB. Both reversed-order runs showed the
memory saving. Pooled throughput was 43,852
versus 42,676 input tokens/s (+2.8%).
The run-to-run spread prevents calling that small difference a confirmed speed win.

The existing experimental fast path was 11.9% faster than defaults in
this phase. That is a remeasurement of prior work, not a new round-2 optimisation.
Larger command buffers lost throughput and consumed more memory. Doubling the
microbatch gained only about 1.6% while adding roughly 0.76 GiB. Alternate Metal
synchronization showed no clear speed gain. Keep those changes out of defaults.

## Measurements

Phase 1 (32 updates; 24 warmed):

| Configuration | Warmed tok/s, runs 1 / 2 | Pooled tok/s | Change vs fast path | Mean peak MLX GiB |
|---|---:|---:|---:|---:|
| Existing fast path, MBS8 / GA8 | 44,850 / 44,016 | 44,429 | +0.0% | 1.570 |
| 128 ops / 128 MiB buffers | 42,417 / 41,309 | 41,856 | -5.8% | 1.684 |
| 1,000 ops / 400 MiB buffers | 37,184 / 38,090 | 37,632 | -15.3% | 2.201 |
| MBS16 / GA4 | 45,088 / 45,178 | 45,133 | +1.6% | 2.329 |

Phase 2 (64 updates; 52 warmed):

| Configuration | Warmed tok/s, runs 1 / 2 | Pooled tok/s | Change vs fast path | Mean peak MLX GiB |
|---|---:|---:|---:|---:|
| Training defaults | 38,982 / 37,304 | 38,124 | -10.7% | 1.723 |
| Existing fast path, MBS8 / GA8 | 43,473 / 41,908 | 42,676 | +0.0% | 1.581 |
| 16 ops / 16 MiB buffers | 43,394 / 44,319 | 43,852 | +2.8% | 1.276 |
| Alternate Metal synchronization | 43,562 / 43,361 | 43,461 | +1.8% | 1.581 |

Pooled rates are total warmed tokens divided by total completed training-step
time across both repetitions. Mean peaks are arithmetic means of the two
per-run peak allocations. Do not compare absolute rates across phases as if
these were additional optimisation gains.

All 16 runs completed: 12,582,912 input tokens in total. No memory guard fired.
Free swap settled at 1,575 MiB after the first run and remained stable through
the remaining trials. System memory and utilization are whole-system readings.

## Applying the memory setting

Set these before importing MLX or launching the existing training command:

```sh
export MLX_MAX_OPS_PER_BUFFER=16
export MLX_MAX_MB_PER_BUFFER=16
```

The measured result includes `--compiled-accumulation`,
`--force-fused-attention`, and `--experimental-packed-swiglu`. The smaller
buffer limits were not isolated on the default configuration, other model
sizes, or longer contexts. They remain an optional tuning setting; no global
shell configuration or package default was changed.

## Numerical and duration limits

Saved BF16 model tensors were compared after the two phases. Repeated reference
runs themselves were not bitwise identical, so bitwise checkpoint equality is
not an appropriate acceptance criterion here. The second phase's scheduling
variants differed from the first fast reference by relative L2 values of
0.001710–0.003124;
the repeated reference differed by 0.002842. Maximum
absolute weight differences were 0.00781250.
These are short-run final-weight checks, not optimizer-state parity, gradient
parity, convergence equivalence, or an evaluation of trained-model quality.
Per-trial comparison data are in `weight_comparison.json` in each run directory.

Each longer trial ran for approximately 24–28 seconds. These results do not
establish multi-hour thermal throughput or a universal speed ceiling. The new
runner passed Python compilation and CLI help checks. Production model,
optimizer, loss and trainer code were not modified.

## Method

Real Metal GPU execution outside the restricted sandbox on the Apple M5 Pro,
24 GiB unified memory, MLX 0.32.2, AC power with `lowpowermode 2` reported by
`pmset`. All trials used the normal trainer on `data/english_corpus_5b.bin`,
49,166,848 parameters, causal AR, BF16 model weights with FP32 master weights,
sequence length 256, seed 42, and 16,384 input tokens per optimizer update.
The LR schedule and corpus positions match within each comparison phase.

The reference enables the existing compiled accumulation, forced fused SDPA,
and experimental packed SwiGLU options. `defaults` leaves those options off.
All new variants change one scheduling setting or microbatch plan relative
to that reference; sequence length and effective batch remain fixed.

Each variant runs in a fresh process, once in forward order and once in reverse
order. First phase: 32 updates per trial, excluding the first 8 from throughput.
Second phase: 64 updates, excluding the first 12. Rates measure completed
training-step calls, excluding setup, final checkpoint writes and corpus
preparation outside those calls. These are input-token rates; each 256-token
sequence supplies 255 next-token targets. Peaks include warmup. Peak MLX
allocation is not whole-system RAM or process RSS.

The profiler uses a 3.2 GiB advisory MLX limit plus update-boundary checks;
system monitoring stops training after three low-headroom samples, at 30%
free system memory or 768 MiB free swap. These guards are advisory and sampled,
not hard OS allocation limits.

## Reproduction and evidence

```sh
.venv/bin/python scripts/benchmark_mlx_round2.py --output logs/my_round2
.venv/bin/python scripts/benchmark_mlx_round2.py --output logs/my_round2_sync \
  --updates 64 --warmup 12 --variants defaults reference buffers16 fastsync
```

Actual run data: `logs/mlx_round2_20261112/` and
`logs/mlx_round2_20261112_sync/`. Each contains `results.json`, per-trial
metadata, update timings, system telemetry, console logs, final model weights,
and a power-settings snapshot. Final model weights remain local and are not
tracked in Git. No production training defaults were changed.

The scheduling settings are present in the source for the installed MLX
version: [MLX 0.32.2 environment readers](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/utils.h)
and [Metal command-buffer limits](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/device.cpp).
These are version-specific tuning experiments, not universal hardware settings.
