# COROSred TPU/CUDA parity audit — 17 October 2026

Update, 18 October: Ivan confirms batch/data accounting was already extensively
investigated. Do not treat the comparability checklist below as a new priority.
The recovered investigation history and revised next step are in
`tpu_cuda_history_recovery_2026-10-18.md`.

The saved 50M trajectories establish a training-quality gap. They do not yet
establish a backend defect or its cause. This audit inspected the actual final
checkpoint payloads, their embedded configs and scheduler states, and current
trainer/objective/launcher code. Historical agent diagnoses remain hypotheses.

## Verified checkpoint facts

| Field | CUDA `50m_lightning` | TPU `unified/50m_python` | TPU `unified/50m_tpu_schedstep1` |
|---|---|---|---|
| Global step | 12,716 | 12,716 | 12,716 |
| Saved model tensor dtype | FP32, all 133 tensors | FP32, all 133 tensors | FP32, all 133 tensors |
| Saved embedding/output values equal | Yes | Yes | Yes |
| Optimizer state present | Yes | No | No |
| Scheduler `update_cadence` | 1 | 10 | 10 |
| Config microbatch | 192 | 48 | 48 |
| Config accumulation | 2 | 1 | 1 |
| Config device count | 1 | 8 | 8 |
| Config corpus path | `python_corpus_2.5b.bin` | `python_corpus_5b.bin` | `python_corpus_5b.bin` |

Model architecture and nominal LR, warmup, decay, clipping, causal ratio and
objective-weight settings agree across these embedded configs. The alternate
TPU config explicitly sets objective `sched_cadence=1`, but its saved LR
scheduler still has cadence 10. That experiment did not isolate LR cadence.

Saved equal embedding values are consistent with tying; they do not prove the
in-memory parameter aliasing or optimizer grouping throughout original training.
FP32 saved tensors likewise do not establish every operation's compute dtype.

## Comparability still to establish

1. **Actual TPU topology.** In current multi-process execution, 48 samples per
   replica times eight replicas equals CUDA's 192 times two accumulation steps:
   384 sequences per update. In SPMD, the current trainer treats 48 as a global
   batch. `_device_count=8` in a config alone does not distinguish these cases.
   The launch mode and observed runtime world size of the original run are needed.
2. **Token stream identity.** Different corpus filenames do not prove different
   contents. The local 5B file exists; the named CUDA 2.5B file is absent. Matching
   consumed token hashes or original corpus provenance is needed before claiming
   identical examples. Equal steps alone do not prove equal consumed tokens.
3. **Objective partitioning.** Current code rounds the causal/infill split per
   microbatch. At ratio 0.65, CUDA's two 192-row microbatches use 250 causal and
   134 infill rows; eight 48-row TPU replicas would use 248 and 136. Assignment
   of individual rows to the two tasks also differs. Masked-token loss is averaged
   per microbatch/replica, so different partitions are not an exact gradient match.
   These are confounders, not a demonstrated explanation for the observed gap.
4. **Original training code.** Checkpoints do not identify a training Git commit
   or source hash. Present-day reduction code and checkpoint upload dates cannot
   prove which implementation produced the original weights.
5. **Exact resume is unavailable for these TPU payloads.** Current TPU saving
   omits AdamW state, while `load_checkpoint` expects `optimizer_state_dict`.
   Weight-only initialization is possible; it is not equivalent to resuming
   Adam moments. This does not explain a run trained uninterrupted from scratch.

## Controlled probe prepared

`scripts/diagnose_backend_parity.py` freezes checkpoint weights, token IDs,
infill masks, causal partition, diagnostic loss weights, and fresh AdamW settings
in one portable fixture. Captures save forward outputs (including reliability
scores), gradients before clipping, parameter deltas after clipping and one
update, component losses and the preclip norm. Comparison requires matching
fixture and source hashes and reports absolute error, relative L2 and cosine.

The prepared fixture uses CUDA's final checkpoint, eight 64-token rows from the
local 5B corpus, and a fixed mask. It exercises the current production unified
objective with alpha 0.7, beta 0.3 and gamma 0.1. These intentionally fixed probe
weights are not a reconstruction of the original run's schedule or EMA gates.
The fresh optimizer is identical on each backend; original CUDA moments are
deliberately not used because corresponding TPU moments are unavailable.

Commands from the repository root:

```bash
python scripts/diagnose_backend_parity.py prepare \
  --checkpoint checkpoints/corosred/50m_lightning/checkpoint_final.pt \
  --data data/python_corpus_5b.bin \
  --output scratch/backend_parity/fixture.pt

python scripts/diagnose_backend_parity.py capture \
  --fixture scratch/backend_parity/fixture.pt --device cuda --precision fp32 \
  --output scratch/backend_parity/cuda_fp32.pt

python scripts/diagnose_backend_parity.py capture \
  --fixture scratch/backend_parity/fixture.pt --device xla --precision fp32 \
  --output scratch/backend_parity/xla_fp32.pt

python scripts/diagnose_backend_parity.py compare \
  --left scratch/backend_parity/cuda_fp32.pt \
  --right scratch/backend_parity/xla_fp32.pt \
  --output scratch/backend_parity/cuda_vs_xla_fp32.json
```

Run the same fixture and source files on each machine; do not regenerate the
fixture independently. Begin with one replica and no SPMD. Repeat with
`--precision bf16` after the FP32 comparison. This is an inference/backward/AdamW
operator probe, not the real distributed trainer. It does not test original
optimizer state, data iteration, accumulation, distributed all-reduce, fused
CUDA AdamW, dropout/checkpointing, dynamic schedules, or long-run convergence.
If the operator probe agrees, add those factors individually in subsequent
trainer tests; agreement here alone cannot exonerate distributed training.

## Validation

Tiny-model end-to-end checks: independent CPU FP32 captures matched exactly;
CPU BF16 produced detectable differences; comparison rejected a different
fixture hash. Two actual 50M CPU FP32 captures also matched exactly across all
saved forward tensors, gradients and parameter updates (maximum absolute
difference zero). Their diagnostic loss was 1.6790300608 and preclip gradient
norm 7.4013557434. These are small fixed-probe values, not a reevaluation of the
published validation trajectory. Results are in
`scratch/backend_parity/cpu_repeatability.json`. No TPU or CUDA execution was
available for this audit. No production training behavior has been changed.
