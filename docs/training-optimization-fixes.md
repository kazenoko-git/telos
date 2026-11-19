# Training audit fixes — 2026-11-19

Six audit priorities were addressed. Hardware-dependent changes remain optional
where the available evidence does not justify a default change.

| Priority | Implementation | Effect on training numerics |
| --- | --- | --- |
| Disabled gradient clipping | CUDA scaled and ordinary optimizer paths skip clipping when `grad_clip <= 0`, matching TPU/MLX behavior. | Intentionally restores nonzero gradients in configurations that previously zeroed or incorrectly scaled them. Positive clipping is unchanged. |
| Native CUDA BF16 | The trainer uses `cuda_has_native_bf16(device)` and normalizes precision aliases. | Unsupported BF16 configurations fall back to FP16 plus GradScaler. Precision and gradients can change. No claim is made that previous T4 validation used emulated BF16. |
| MLX autotuning | Training and tuning share compiled step construction, compiled accumulation, asynchronous accumulation and the selected evaluation policy. Trials also honor `async_eval` and include the final fence in timing. | Trials remain isolated from training. Selecting a different batch plan can change accumulation rounding or stochastic execution order; equal effective batch alone does not promise bitwise identity across plans. |
| MLX optimizer overlap | Accumulated updates support `training.async_eval: true`, with array-handle snapshots, bounded dispatch and explicit checkpoint/logging fences. Eager evaluation and stage profiling keep their synchronization boundaries. | Exact losses, model, optimizer, master-weight and MLX RNG arrays matched in the local ABBA comparison. Default remains false. |
| COROSred diagnostics | Removed overwritten metric work; explicit `compute_metrics=False` skips diagnostic reductions without removing BCE normalization. Trainer requests metrics on tracking and logging steps. MLX probe cadence is configurable. | Skipping PyTorch diagnostics preserved loss, gradients and RNG exactly in the regression test. Probe cadence and feedback cadence changes can change future gradients and remain optional. |
| XLA LR graph specialization | Added optional `XLAAdamW`, transferring changing decay and bias-correction controls as device data rather than using changing scalar constants in optimizer expressions. The host LR scheduler is unchanged. | Preserves the exact host LR sequence and AdamW mathematics, but arithmetic ordering can change FP32 rounding. Physical small FP32 checks passed; native AdamW also remained compilation-stable on Torch-XLA 2.9.0. The representative BF16 full-trainer check used native AdamW. No sustained speed benefit is established; default remains native AdamW. |

Physical TPU and CUDA follow-up: see [v1.8.2 validation](releases/v1.8.2-validation.md)
for the tested AR configurations, exact replay summaries, and evidence limits.

## Configuration

These are configuration dictionary/YAML fields, not new CLI switches:

```yaml
training:
  async_eval: false              # true opts into MLX optimizer overlap
  xla_dynamic_optimizer: false   # true opts into the TPU candidate
corosred:
  reliability_cadence: 1         # MLX: try 25 to reduce probe frequency
  cadence: 1                    # CUDA: try 25 to reduce feedback readbacks
```

Do not copy this CUDA cadence value into a TPU configuration unintentionally:
the existing TPU feedback default remains 25. Existing `crsr` aliases are also
accepted. Changing cadence can alter schedule gates and adaptive rebalancing.
MLX keeps its previous per-update probe behavior unless explicitly configured.
Checkpoint/replay tests cover the default cadence and a cadence of three.

## Local evidence

The short synchronized MLX ABBA experiment used d_model=512, 14 layers,
vocabulary=8192, T=256, microbatch=8 and accumulation=8. Each trial had five
warmup updates and twelve measured updates. Measured blocks include their final
GPU fence; saved arrays include losses, model, Adam state, master weights and RNG.

- Synchronous pooled throughput: **44,239 tokens/s**.
- Asynchronous pooled throughput: **44,049 tokens/s**, or **-0.43%**.
- All four saved numerical-state hashes matched exactly.
- One asynchronous trial reached 2.52 GB peak allocation versus about 2.13 GB
  for the other trials. This is not evidence of a memory improvement.
- Clocks and thermals were not controlled. The experiment establishes no speed
  advantage; overlap remains optional.

Reproduce with:

```sh
.venv/bin/python scripts/benchmark_mlx_update_overlap.py \
  --output reports/training_fixes_2026-11-19/mlx_update_overlap.json
```

The XLA optimizer CPU check ran sixteen warmup updates, preserving all LR
values exactly, with maximum parameter difference **1.1920929e-7** against native
AdamW. Unit coverage also checks AMSGrad, decay, zero LR, and checkpoint replay.
CPU results cannot establish XLA graph stability.

On physical TPU, run:

```sh
python scripts/validate_xla_dynamic_optimizer.py --device xla \
  --output reports/xla_dynamic_optimizer.json
```

The validator records actual `CompileTime` counts and fails if compilation
continues after four updates. The subsequent physical TPU check passed, but
native AdamW was also compilation-stable in the small FP32 comparison. These
results do not establish a need for the candidate optimizer.

Subsequent physical checks passed for the native AR trainer on a single TPU
v5e-1 runtime (BF16) and Kaggle two-T4 DDP (requested BF16 falling back to FP16
with GradScaler), including accumulation and exact checkpoint/resume summaries.
See the linked release validation for scope; sustained performance remains
unmeasured.

## Regression result

The initial local fix suite passed: **449 passed, 7 skipped** (`.venv/bin/python -m
pytest -q`). Skips are not evidence of physical CUDA or TPU acceptance. No
existing test tolerances were relaxed. Unrelated working-tree changes were
preserved.

The integrated v1.8.2 release checkout subsequently passed **464 tests, with
7 skipped**. Hardware results are documented separately in the release validation.
