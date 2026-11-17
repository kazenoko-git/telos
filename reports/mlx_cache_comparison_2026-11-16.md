# MLX cache-size comparison — 2026-11-16

Follow-up: the comparison completed after memory was freed on November 17;
see [completed results](mlx_cache_comparison_2026-11-17.md).

Status: no valid cache-size comparison yet. Both 128 MiB baseline attempts
stopped on the unchanged observer swap-growth guard before measurement warmup.
256 and 512 MiB trials have not run. No throughput or gap-reduction claim is supported.

## Preserved attempts

| Attempt | Completed updates | Warmed updates | Outcome |
|---|---:|---:|---|
| `logs/mlx_cache_comparison_20261116_01/untraced_00_128` | 1 | 0 | Observer memory guard; system swap grew about 1 GiB |
| `logs/mlx_cache_comparison_20261116_02/untraced_00_128` | 6 | 0 | Observer memory guard; system swap grew about 650 MiB |

Both attempts used the same initial-weight hash
`b95399e433484c6b0b87d846f5a89cd136d68900592debadac9dd0123a466be3`.
`cache_override.json` verifies the requested and applied cache limit.
The worker interruption and partial data are retained, rather than presented as
completed timing trials. No competing benchmark was active when testing began.

The idle system also showed substantial GPU activity from other applications,
often around 40%. This is additional measurement interference; it does not
establish which app caused memory or GPU activity at every point.

## Prepared experiment

`scripts/compare_mlx_cache.py` uses fresh processes and reversed order
128, 256, 512, 512, 256, 128 MiB for each of two separate groups:

- Untraced: 64 updates each, excluding the first 16 updates from throughput.
- Traced: 96 updates each, attaching the successful lightweight Metal/CPU
  instrument combination after 20 updates for a requested two-second capture.
  These timings are diagnostic, not throughput comparisons.

The same frozen source, input corpus prefix, model shape, precision, optimizer,
seed and experimental options are used throughout. MBS8 / accumulation8 /
sequence256, 16,384 input tokens per update. The cache-size override is confined
to the diagnostic worker. No trainer, model, package defaults or shell settings
are changed. A manifest records source/data/harness hashes; a final assertion
checks equal initial weights across completed trials. The existing 3.2 GiB MLX
memory limit and memory/swap guards remain in force.

Each completed trace is exported and processed by the existing gap and CPU
correlation scripts. Compare gap duration and counts per captured second,
preparation/submission timing, CPU stack categories and retained cache memory,
with reversed-order repetitions. A larger cache is a hypothesis, not a presumed win.

When system headroom is adequate, use a fresh output directory:

```sh
.venv/bin/python scripts/compare_mlx_cache.py \
  --source /private/tmp/telos-fusion-frozen-f558146 \
  --data /Users/ivansamuel/Documents/Projects/telos-mlx-fusion/reports/mlx_fusion_2026-11-16/corpus_prefix.bin \
  --output logs/mlx_cache_comparison_fresh
```

The frozen source is temporary; archive it before depending on this path later.
The runner stops after an invalid trial and preserves partial results. This
report must be updated with actual completed results before any cache setting
is recommended or changed.
