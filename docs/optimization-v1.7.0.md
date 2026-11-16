# v1.7.0 backend optimization and adapter training

Implemented from v1.6.2 (`08e56fe`), in an isolated worktree. The original working
checkout and its unrelated edits remain untouched. The MLX QKV/RoPE experiment
was copied deliberately, with its tests, and remains opt-in.

## Benchmark contract

Pretraining benchmark mode now fences device work at update boundaries, excludes
five completed warmup updates and reports synchronized update latency. Aggregate
elapsed includes host input preparation and between-update reporting. Distributed
latency/elapsed use the maximum across workers, so duration-based stopping follows
the same branch on all workers. Training outside benchmark mode keeps its normal
asynchronous execution policy. This deliberately measures synchronized updates;
it is not an unconstrained asynchronous throughput experiment.

JSON retains existing fields and adds `timing`, `warmup_updates`, `token_unit`
and `valid_measurement`. Runs ending during warmup have no valid measurement;
they report zero measured updates rather than counting startup as steady state.
Rates count input tokens, including the final input position and any padding,
not necessarily supervised next-token targets.

## Implemented

- CUDA: mixed-precision post-training with FP32 parameter/update state, BF16 or
  scaled FP16 autocast, scaler resume, fused AdamW, optional model compilation,
  torchrun DDP and per-rank state. The existing DataParallel fallback remains
  available when multiple GPUs are requested without torchrun. SFT normalization
  uses global supervised-token counts with averaged worker gradients.
- CUDA input: two reusable pinned staging buffers, a dedicated copy stream,
  readiness events, consumer stream lifetime tracking and cancellation/teardown.
  Native CUDA SDPA GQA avoids explicitly repeating K/V heads.
- PyTorch pretraining: project only next-token positions for AR; optional packed
  QKV/gate-up projections with canonical split checkpoint keys; optional tiled,
  checkpointed projection/CE. Tiling retains FP32 reductions and recomputes logits
  during backward. It trades compute for lower activation memory.
- MLX: compiled SFT/DPO update with explicit model/optimizer/RNG/master state;
  class-index CE replaces full FP32 log-softmax; paired DPO inputs share a forward.
  The existing QKV unpack/RoPE/head-permutation Metal implementation is now wired
  into the current release as an opt-in with shape/dtype/device guards.
- XLA: fixed post-training sequence shapes, bulk CPU collation/transfer, BF16
  autocast with FP32 model/update weights, loss reporting collectives only on
  shared logging updates, and identical initial adapter factors across ranks.
  Forced pretraining GC/allocator trimming is disabled by default; YAML
  `training.xla_cleanup_every` can restore an explicit cadence.
- Shared post-training: one-time tokenization, unchanged sampled record order,
  bounded frozen-reference score cache and concatenated DPO pairs. Cache identity
  uses token and label sequences, is local to one frozen reference/run, and is
  recomputed after resume. Maximum entries default to 128 batches.
- Explicit `auto|on|off` activation-checkpointing policy in pretraining.
- Native checkpoint loading honors configured head counts, context and tied
  embeddings, including merged adapter exports.

The audit also proposed further experiments. This release does not introduce
selected-position MDLM compaction, alter UNDLM supervision, replace XLA LR
schedules, or add an unvalidated TPU input loader. MLX accumulation retains its
bounded activation-memory materialization policy; compiled accumulation remains
an existing opt-in. These changes require their own hardware/convergence evidence.

## LoRA and QLoRA

`telos posttrain --finetune lora|qlora` supports SFT and DPO. Default targets are
transformer attention/MLP linear projections; embeddings, output heads and base
weights stay frozen. LoRA uses FP32 factors, zero-initialized B, scaling
`alpha/rank` and optional input dropout. The inactive reliability head is frozen.

PyTorch QLoRA uses packed 4-bit NF4 with block size 64, FP32 scales and portable
per-projection dequantization/matmul. It supports the shared CPU/CUDA/XLA code
path but has no dedicated fused CUDA quantized matmul. MLX uses native affine
4-bit group-64 quantization and quantized matmul; input widths must be multiples
of 64. Embeddings/output heads are unquantized. Double quantization and paged
optimizers are not implemented. This release does not claim the full QLoRA
paper's memory/performance results.

Checkpoint outputs:

| Backend | Merged model | Adapter only | Complete resume |
|---|---|---|---|
| PyTorch | `checkpoint_final.pt` | `checkpoint_final.adapter.pt` | `checkpoint_final.resume.pt` (+ rank files) |
| MLX | `model.safetensors` | `model.adapter.safetensors` + `.json` | `model.resume.npz` |

Merged exports reload through normal native evaluation. Adapter-only loading is
available through `telos.posttrain.adapters.load_adapter` (PyTorch) and
`telos.posttrain.mlx_adapters.load_adapter` (MLX), with the original base
checkpoint supplied for SHA256 verification. Load the original base model first;
for MLX, match the precision used during adapter training. Move a PyTorch model
to its target device after loading the adapter. Example:

```python
from telos.eval.runner import load_model_from_checkpoint
from telos.posttrain.adapters import load_adapter

model, _, _ = load_model_from_checkpoint("base.pt")
load_adapter(model, "checkpoint_final.adapter.pt", base_checkpoint="base.pt")
model.eval()
```

Resume restores frozen references, full quantized/base state, optimizer,
FP32 masters where applicable, sampler, RNG and FP16 scaler. Resume identity
includes the base hash/path and all new options. Old post-training resume
identities are rejected instead of silently changing execution semantics.
A merged export starts a new run; it is not an adapter-training resume artifact.

## Local validation

Runtime: Python 3.13.14, PyTorch 2.13.0, MLX 0.32.2 on Apple M5 Pro/Metal.
CUDA and TPU hardware are unavailable here. CPU control-flow simulations and
hardware-guarded tests are explicitly separate from accelerator verification.

Final full-suite result: **277 passed, 7 skipped, 6 failures reproduced on the
untouched baseline** (five macOS sandbox execution failures and one MLX FP32
checkpoint-gradient tolerance failure). No tolerances were loosened.

The release report under
`reports/backend_optimization_v1.7.0_2026-11-15/` records exact suite results,
baseline failure identifiers, source hashes and benchmark metadata.
Focused checks cover LoRA/QLoRA frozen weights, trainable factors, nonzero-dropout
resume, BF16 master resume, merged reload, adapter reload, packed projection
outputs/gradients/resume, tiled CE gradients, benchmark fencing, CUDA pipeline
control flow and XLA fixed-shape/reporting control flow. Normal benchmark mode
completed AR/MDLM/UNDLM/COROSred on both CPU and Metal with five measured updates
after five warmups.

Physical CUDA (single/multi-GPU, FP16 scaling, copy-stream overlap, GQA, NF4),
TPU launcher/compilation/collectives and sustained convergence still require
hardware validation. The previous v1.5.5 Colab launcher failure was not retested
here. No local result establishes distributed speedup or model quality.

## Short Metal performance diagnostic

Fresh-process A/B/B/A comparisons used v1.6.2 versus the v1.7 working source,
2,048 updates per trial, a two-layer D128 model, V4096, batch 2, length 64,
FP32 source weights and synthetic repeated text. Timing fences the entire call
and includes model conversion, compilation, tokenization and final checkpoint
export; it is not a warmed per-update measurement. Source hashes accompany each
trial, since the working-tree implementation was not yet committed at run time.

| Method | v1.6.2 seconds (A1 / A2) | v1.7 seconds (B1 / B2) | Ratio of summed elapsed |
|---|---:|---:|---:|
| SFT | 3.642 / 4.699 | 2.594 / 2.084 | 1.78x |
| DPO | 8.943 / 8.869 | 2.492 / 2.621 | 3.48x |

These are small, short synthetic diagnostics. DPO reuses identical truncated
response pairs, giving the reference cache a best-case hit rate; the result is
not representative of a unique-record dataset. SFT shows substantial run-to-run
variation. No general or sustained speedup is inferred. First paired final
weights differed by relative L2 `1.93e-5` (SFT) and `1.11e-5` (DPO); this is a
short numerical comparison, not convergence equivalence. Source manifests
identify exactly what was measured; subsequent defensive fixes and documentation
are covered by final regression checks, not by a new performance claim.

Reproduce each trial in a fresh output directory using
`scripts/benchmark_posttrain.py --source CHECKOUT --output FRESH_DIR --method sft
--steps 2048` (add `--compile` for v1.7; choose `dpo` for the other diagnostic).
Model checkpoints remain local; only measurements/source hashes are tracked.

## References

- [LoRA](https://arxiv.org/abs/2106.09685)
- [QLoRA](https://arxiv.org/abs/2305.14314)
- [MLX quantized linear](https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.QuantizedLinear.html)
- [CUDA timing and streams](https://docs.pytorch.org/docs/stable/notes/cuda)
- [MLX compilation](https://ml-explore.github.io/mlx/build/html/usage/compile.html)


## v1.7.1 follow-up

The six failures above are fixed in v1.7.1 without removing tests or loosening
numerical tolerances. macOS sandbox launch now resolves interpreter/scratch
symlinks and allows dyld to open the root directory itself; access to unrelated
files, networking and Python process forking remains denied. MLX defaults to
full-precision FP32 GEMMs, and low-precision embedding gradients accumulate in
FP32 before casting back once. See the v1.7.1 validation report for evidence.

`MLX_ENABLE_TF32=1` remains an explicit speed/precision tradeoff. Set it before
starting Python. MLX caches the setting at its first matmul; importing Télos
cannot retroactively change a backend already initialized by another library.
BF16/FP16 matmuls retain native acceleration. These precision fixes can change
training trajectories relative to v1.7.0, and FP32 full-precision matmuls can be
slower than TF32. This patch does not claim a throughput improvement.
