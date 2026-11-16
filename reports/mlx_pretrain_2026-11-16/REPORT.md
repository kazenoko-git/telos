# MLX pre-training optimization — 2026-11-16

A repeatable improvement was found for **49.17M autoregressive pre-training at
1,024-token context** on Apple M5 Pro: isolated **MLX 0.32.4** delivered **18.58%
higher warmed completed-update throughput** than MLX 0.32.2. Peak MLX allocation
increased from about **1.25 to 1.85 GiB**. The 256-token comparison did not show a
repeatable gain, so this does not justify a blanket runtime upgrade.

Earlier throughput around 42–49k tokens/s was measured at 256-token context.
The 1,024-token runs perform more attention work per input token; their matched
comparison is 25.9k versus 30.7k tokens/s. The percentage gain applies within that
longer-context comparison.

This is an upstream-runtime/configuration result. The new kernel and dispatch
prototypes did not establish improvements and remain outside production code.
The shared Python environment, production defaults and original dirty checkout
were preserved. Research is recorded on `codex/mlx-pretrain-optimization`, based on
v1.7.3 (`9ea78b71280332632519bbc1374cfcabb50580e7`).

## Matched production-loop measurements

All runtime comparisons used the real local English corpus, BF16 model weights,
FP32 masters, AdamW, seed 42, width 512, 14 layers, eight equal heads, vocabulary
8192 and 49,166,848 parameters. Both sides enabled the existing compiled
accumulation, packed SwiGLU, fused QKV/RoPE and forced fused attention options,
with 16-operation/16-MiB command buffers. Model geometry, data cursor, LR schedule,
precision and effective batch were held fixed within each comparison.

Every trial was a fresh process; order was 0.32.2/0.32.4/0.32.4/0.32.2. Input-token
rates divide warmed tokens by summed completed training-step call time. The
accumulated step materializes optimizer/model state before returning; there is no
unfenced queued-work throughput claim. Python imports and setup/export are
excluded from the warmed rates.

| Context / comparison | 0.32.2 tok/s, repetitions | 0.32.4 tok/s, repetitions | Pooled rate gain |
|---|---:|---:|---:|
| 1024, initial 32 updates; first 8 excluded | 24,063 / 26,329 | 30,655 / 29,143 | +18.83% |
| 1024, confirmation 64 updates; first 12 excluded | 26,082 / 25,771 | 30,346 / 31,147 | **+18.58%** |
| 256, 32 updates; first 8 excluded | 44,526 / 36,583 | 34,947 / 43,364 | -3.64%; opposing pair results |

At context 1024, MBS2/accumulation8 processes 16 sequences and 16,384 input tokens
per update. At context 256, MBS8/accumulation8 processes 64 sequences and the same
input-token count. **Changing context length is not counted as an optimization**;
only matched runtime pairs are compared. Input tokens include the final input
position and special tokens, not exclusively supervised targets.

The confirmation's pooled rates are **25,925 versus 30,741 input tokens/s**.
Profile wall times, which include training setup and checkpoint export but exclude
Python imports/profiler setup, were 54.31/52.75 seconds versus 48.37/46.74 seconds:
about 12.6% higher setup/export-inclusive update rate. Peak allocations were
1.247/1.247 GiB versus 1.851/1.849 GiB. Allocation includes warmup and is not total
process RSS or system RAM.

Fresh public-launcher captures verified all 86 initial parameter tensors have
identical names, dtypes, shapes and raw-data SHA-256 digests across runtimes.
All confirmation checkpoints reached global step 64, optimizer step 64, data row
1024 and nonzero final LR approximately 4e-5. Commands, imported extension paths,
source manifests, timings and system telemetry are retained.

The longer trials lasted under a minute each. These are not multi-hour thermal,
convergence, trained-model quality, other-chip, other-model-size or all-context
results. The interactive desktop had background GPU activity and changing swap
usage. Positive results in both pair orders and a second comparison strengthen
this particular measurement; they do not establish exclusive GPU execution.

## Correctness and restart checks

- The focused attention/QKV/step/accumulation/checkpoint/loss suite passed **43
  tests** on isolated 0.32.4. The full suite passed **344 tests, 7 skipped** in
  32.57 seconds with the parent process using 0.32.4. Some existing subprocess
  tests replace PYTHONPATH, so that full-suite total alone does not prove every
  child used the isolated runtime.
- Additional shape-matched causal attention checks at `[2,8,1024,64]` compared
  outputs and Q/K/V gradients against independent FP32 matmul/softmax math.
  On 0.32.4, FP32 relative L2 errors were below 4.7e-7. BF16 output error was
  0.163%, and gradient errors 0.425–0.486%. The corresponding 0.32.2 BF16 gradient
  errors were 0.394–0.512%. Both passed the predeclared component gates; this is
  not whole-model quality equivalence.
- Explicit fresh-process **AR and MDLM** controls, actual SIGTERM after an atomic
  step-2 checkpoint, and restarts to step 12 passed strict full-checkpoint
  comparisons. Every child asserted its 0.32.4 version and imported extension
  path. Checks included every key/container, dtype/shape, finite tensor/scalar
  value, model/master/optimizer state, RNGs, data cursor and configuration.
  All 35 AR and 36 MDLM tensor leaves matched exactly. These used a small
  width-128, one-layer model at context1024/effective batch2, not a 50M
  interrupted process.
- Across the runtime versions, final 50M weights are **not bitwise equal**:
  relative L2 difference was 0.1744%, FP32 masters 0.1147%. Repeated 0.32.2
  controls matched exactly. Final logged losses were 6.8610 versus 6.8607.
  These short loss/state comparisons do not establish equivalent convergence.

Detailed evidence: `attention_long_0322.json`, `attention_long_0324.json`,
`initial_weights_comparison.json`, `initial_*/results.json`,
`runtime_long_update_progress.json`, `runtime_long_final_state_differences.json`,
`runtime_resume_ar/results.json`, `runtime_resume_mdlm/results.json` and
`validation/` console logs.

## Prototypes that were not promoted

| Candidate | Observation | Decision |
|---|---|---|
| Residual-add/RMSNorm forward fusion | Small FP32/BF16 outputs and VJPs matched exactly. Synchronous component forward/backward was about 20% slower; median paired one-block change was about +0.08%. | No useful block improvement. |
| One-microbatch-ahead graph preparation | Eight small state/RNG cases and selected 50M model/master/optimizer/RNG/cursor groups matched exactly. First throughput pair was tied; the final baseline slowed substantially amid desktop activity, confounding the pooled result. | No repeatable speed win established. |
| Final accumulated backward + clipping + Adam compilation | Eight small state/RNG cases and final 50M model/master/optimizer tensors matched exactly. Interleaved 64-update test: pooled -1.39%, median paired +0.86%. | Mixed timing, no reliable gain. |
| MBS32/accumulation2 versus MBS8/accumulation8 | Same 64-sequence effective batch and context256; pooled -4.13%, median paired -1.37%, much larger allocation. BF16 accumulation ordering also changed final tensors. | No measured benefit. |

The exploratory code is in `scratch/mlx_pretrain_20261116/`. It is not imported by
production training. The only existing-tool change exposes the already-released
QKV/RoPE flag in the bounded profiler and records the imported MLX module path.
The initial stage diagnostic overlapped another local validation worker and is
retained as triage only, not an uncontended throughput baseline.

## Reproduce

From `/Users/ivansamuel/Documents/Projects/telos-mlx-pretrain`, use fresh output
paths. The isolated runtime already exists at `/private/tmp/telos-mlx-0.32.4`;
it can be recreated with `uv pip install --target NEW_DIRECTORY mlx==0.32.4`
and substituted into PYTHONPATH. This does not upgrade the shared environment.

```sh
# The matched runtime comparison creates four fresh production training runs.
PYTHONPATH=. /Users/ivansamuel/Documents/Projects/telos/.venv/bin/python \
  scratch/mlx_pretrain_20261116/run_runtime_abba.py \
  --updates 64 --warmup 12 --output FRESH_COMPARISON_DIRECTORY

# A single tested long-context run on the isolated newer runtime.
PYTHONPATH=/private/tmp/telos-mlx-0.32.4:. \
MLX_MAX_OPS_PER_BUFFER=16 MLX_MAX_MB_PER_BUFFER=16 \
/Users/ivansamuel/Documents/Projects/telos/.venv/bin/python \
  scripts/profile_mlx_training.py --output FRESH_TRIAL_DIRECTORY \
  --tokens 1048576 --seq-len 1024 --microbatch 2 --grad-accum 8 \
  --data /Users/ivansamuel/Documents/Projects/telos/data/english_corpus_5b.bin \
  --experimental-qkv-rope --experimental-packed-swiglu \
  --force-fused-attention --compiled-accumulation \
  --max-mlx-gib 3.2 --critical-memory-free-percent 25 \
  --critical-swap-free-mib 768
```

Memory limits are advisory; headroom guards operate at update/sample boundaries.
The comparison refuses other Python workers, except the verified remote Colab
control CLI, which does not execute local MLX workloads. An early confirmation
attempt stopped before its second trial when that CLI appeared; the remaining
fresh trials resumed only after checking identical source manifests and settings.

`runtime_summary.json` contains the computed rates. Per-trial metadata, configs,
step timings, console logs, telemetry and source hashes are retained. Generated
checkpoint binaries were removed after saving comparison results to avoid
retaining many GiB of disposable data; measurement paths describe original
outputs. Scripts can regenerate the fixtures and checkpoints.

Primary implementation references: [MLX 0.32.4 Metal attention/VJP](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/scaled_dot_product_attention.cpp),
[MLX 0.32.2 RMSNorm rounding/reduction](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/kernels/rms_norm.metal).
The dedicated backward implementation is a plausible explanation for the
long-context improvement, not a kernel-level causal attribution: changing runtime
also changes other backend code.
