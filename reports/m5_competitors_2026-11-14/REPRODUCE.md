# Reproducing the first M5 comparison

The experiment uses production Télos v1.5.5 model code, with no local uncommitted
optimization experiments. Scripts and evidence commits advance during the run;
the summarizer checks that benchmark script hashes and all numeric configuration
fields stayed identical across the four paired trials.

Create an isolated Python 3.13 environment and install the versions listed in
`requirements.lock.txt`. Run from the repository root with that environment's
Python. The local real-corpus file is required and is not redistributed here.
Metadata records a hash of the exact int32 token prefix used, including the
disjoint 2,048-token validation slice. On-disk corpus IDs are uint16.

```sh
PYTHONPATH=. python scripts/benchmark_m5_matched.py \
  --parity-only --output /tmp/my-m5-parity

python scripts/run_m5_matched_pairs.py \
  --data /absolute/path/to/english_corpus_5b.bin \
  --output reports/my-m5-pairs

python scripts/summarize_m5_matched.py reports/my-m5-pairs
```

Output directories must not already exist. The pair runner removes inherited
experimental Metal command-buffer and synchronization settings. Use
`--commit-results` only when each trial should also be added and committed to
the current Git branch; the default does not modify Git history.

Each trial runs a small FP32 mapping/gradient check, then initializes the full
49,166,848-parameter BF16 model from identical weights, measures its held-out
loss, performs 64 optimizer updates, and measures the held-out loss again.
The first 12 updates are excluded from training throughput. Four trials process
4,194,304 training input tokens in total. Smoke runs are reported separately.

Step timing covers data-array creation, forward/backward, accumulation, global
clipping, optimizer update, and evaluation of all pending model/master/optimizer
arrays. It excludes parity checks, setup, validation, telemetry, and serialization.
Whole-process wall time includes those operations. Warmed input-token rates use
16,384 input tokens/update; target-token rates use 16,320 supervised positions.

Held-out loss is measured on eight adjacent 256-token sequences, not a public
capability benchmark or a statistically representative convergence evaluation.
System memory counters describe the whole machine; macOS process RSS does not
include all Metal allocations. Do not add RSS and MLX peak allocation together.

The initial failed smoke directory is intentionally retained. It is a benchmark
script compilation-state bug, fixed before the successful smoke and pair runs.
