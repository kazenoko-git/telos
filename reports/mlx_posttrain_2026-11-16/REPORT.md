# MLX SFT/DPO optimization — 2026-11-16

The final paired synthetic benchmark measured **5.55% faster SFT** and **25.46%
faster DPO**, including setup and checkpoint export. Both DPO paths include the
same correctness fix. Peak MLX allocation was effectively unchanged. These are
short single-machine results with repeated examples, not a universal throughput
or trained-model quality claim.

Implementation is on `codex/mlx-posttrain-optimization`, based on remote HEAD
`1d22c138a239a5f2755d16545f0833b08aa367d1` (v1.7.2), verified at the start of
this task. The original `telos` working tree and its unfinished edits were left
untouched. The implementation is included in v1.7.3. No runtime upgrade was applied.

## Changes

- MLX collates already-tokenized rows directly into NumPy int32 staging arrays,
  then MLX arrays. This removes per-example Torch tensor construction and the
  int64-to-int32 conversion. Prompt masking, answer truncation, EOS supervision,
  padding and sample order remain the same.
- MLX DPO caches individual frozen-reference response scores, so reshuffling
  previously seen examples into different batches can still hit the cache.
  Identity is the immutable prepared-record index, response kind and padded
  width, within one frozen-reference invocation. The capacity remains equivalent
  to the configured batch budget: `2 * batch_size * reference_cache_size`
  response scores. Values are host FP32 scalars, not retained device graphs.
- A miss evaluates the original complete paired batch. It deliberately preserves
  reference forward shapes rather than evaluating a compacted subset of misses.
  All rows must hit to skip that batch's forward. Disabling the cache or setting
  its size to zero evaluates every reference batch. The cache is rebuilt on
  resume; saved checkpoint format and resume identity are unchanged.

## DPO zero-margin gradient fix

The released MLX DPO loss expanded softplus as
`maximum(-margin, 0) + log1p(exp(-abs(margin)))`. Direct MLX differentiation at
zero margin returns **0**, whereas the derivative of `softplus(-margin)` is
**-0.5**. Policy and reference normally agree initially, so this can prevent
learning. The first large diagnostic exposed all-zero optimizer moments in DPO.

The implementation now uses `logaddexp(0, -margin)`. The zero-margin derivative
and finite losses/gradients at extreme margins have regression coverage, and
integration coverage asserts that DPO produces nonzero optimizer moments.
This intentionally fixes the old training trajectory; equality with the broken
loss is not an acceptance criterion.

For a fair optimization comparison, the final baseline is v1.7.2 with **only**
this same loss fix applied. `baseline_loss_fix.patch` preserves the exact change.
Its source was `/private/tmp/telos-mlx-dpo-baseline`; it retains the original
Torch collation and whole-batch reference cache. The candidate includes both the
loss fix and the optimizations described above.

## Final measurements

Apple M5 Pro, Python 3.13.14, MLX 0.32.2, Torch 2.13.0. Each fresh process uses
47,069,696 parameters, width 512, 14 layers, two heads, vocabulary 4096, BF16
weights, FP32 masters, AdamW, sequence length 256, batch size 4, 32 distinct
synthetic records, seed 42 and 256 completed updates. Compilation and default
reference caching are enabled. Paired DPO processes eight sequences per update.
The same source weights and sampled-record sequence are used in each comparison.

| Workload | Baseline seconds, two runs | Candidate seconds, two runs | Pooled update-rate gain | Peak MLX allocation |
|---|---:|---:|---:|---:|
| SFT | 8.629 / 8.192 | 8.013 / 7.924 | 5.55% | Both about 1901.8 MiB |
| DPO | 17.167 / 17.338 | 13.869 / 13.634 | 25.46% | Both about 2330.2 MiB |

Order is baseline/candidate/candidate/baseline separately for each workload.
Gain is the ratio of summed baseline elapsed to summed candidate elapsed minus
one, with equal update counts. Both pair orders improve for each workload.
Timers fence device completion before and after the full `posttrain` call.
The timing includes loading/conversion, tokenization, compile warmup, training,
logging, final model export and full resume-state export. It excludes Python
imports and creation of the synthetic source checkpoint/data. These are completed
updates per end-to-end second, not warm-only or useful-token throughput.

DPO benefits from example reuse. Larger datasets exceeding the cache capacity,
changing padded widths, or cold single-pass workloads may see much smaller gains.
Two repetitions, an interactive desktop, synthetic repetitive answers, and
8–17 second final trials do not establish sustained thermal performance,
convergence, real-dataset gains or results on other Apple chips. No peak-memory
improvement is claimed; MLX allocation is not total process/system memory.

## Numerical checks and tests

`final_state_comparison.json` compares both candidate/baseline pairs and repeated
baselines. For SFT and corrected DPO, all final model weights, FP32 master weights,
optimizer tensors, reference tensors and MLX RNG arrays are **bitwise equal**;
all maximum differences and relative L2 differences are zero. Sampler state
also matches. This is evidence for these specific runs, not a general bitwise
reproducibility guarantee.

The focused suite passed **59 tests in 3.21 seconds** on CPU and Metal:

```sh
/Users/ivansamuel/Documents/Projects/telos/.venv/bin/python -m pytest -q \
  tests/test_mlx_dpo_loss.py tests/test_mlx_posttrain_batches.py tests/test_posttrain_resume.py \
  tests/test_lora_posttrain.py tests/test_posttrain.py tests/test_mlx_losses.py
```

New checks cover exact collation, reordered reference-score reuse, capacity and
zero-size behavior, padding-width identity, FP32/BF16 DPO losses and gradients,
and cache-on/off training-state parity. Integration covers full training,
LoRA, QLoRA, adapter dropout, BF16 masters, fixed/batch padding, compiled/eager
updates and interrupted/resumed execution with an empty rebuilt cache. It checks
weights, masters, moments, reference, host/MLX RNGs and sampling state. Python
compilation and `git diff --check` passed. No full repository suite or model
quality evaluation was run.

## Reproduction and artifacts

From the candidate worktree, choose fresh output directories and run each source
in baseline/candidate/candidate/baseline order, for each method:

```sh
/Users/ivansamuel/Documents/Projects/telos/.venv/bin/python \
  scripts/benchmark_posttrain.py --source SOURCE --output FRESH_OUTPUT \
  --method dpo --steps 256 --batch-size 4 --length 256 --width 512 \
  --layers 14 --records 32 --precision bf16 --compile
# Repeat with --method sft. For baseline use v1.7.2 plus baseline_loss_fix.patch.
```

`final_results.json` retains all final timing/memory measurements and per-source
SHA-256 manifests. `summary.json` contains computed gains and parameter count;
`runtime.json` records the runtime. Per-trial logs, configs, synthetic datasets
and measurement JSON remain local; the repository preserves aggregate measurements,
source manifests, comparison results and console logs. Generated checkpoint binaries were removed
after saving comparison results to avoid retaining many GiB of disposable data;
checkpoint paths in the measurement records therefore describe the original
outputs rather than retained files.

Earlier diagnostics are retained separately in `results.json`,
`large_results.json`, `weight_comparison.json` and stage-profile summaries. Per-trial
fixture/config directories remain local and are regenerated by the benchmark runner.
The first 512-step sweep showed a cache regression against the original
zero-gradient DPO implementation; it is not used for the final working-DPO speed
claim. A separate instrumented 128-update diagnostic with 32 records reduced
reference forwards from 128 to 17 and reference-evaluation wait time from about
2.37 to 0.29 seconds. Instrumentation changes timing, so those stage numbers are
not mixed into the final end-to-end comparison. The corrected final baseline and
paired reruns above supersede the earlier DPO throughput interpretation.
