# TPU/CUDA investigation history recovered — 18 October 2026

Scope: historical training-code investigation, following Ivan's instruction to
leave previously completed batch/data accounting aside. No training code was
changed and no accelerator time was requested during this recovery.

## Sources reviewed

The live Codex chat `01a149ef-6395-7763-a021-f3dfcdbf98fe` confirms that the
Antigravity archive was already exported. The relevant original transcripts
are under `.agent-history/antigravity-ide/chats/`:

- `0c855a7e-dd7a-4139-bd27-5b23f1165730`: **Investigating XLA Versus CUDA
  Discrepancies**, 19 September. Read its opening user statement, final answer,
  relevant tool results and gradient-analysis launch/cancellation records.
- `d1aa3880-8338-4612-b659-7a752b762d65`: **TPU Divergence Analysis Report**,
  13 September. Read its remediation plan and closing implementation proposal.
- `c498704a-5904-4bfd-93eb-e8089c486ea4`: **Model Evaluation Framework
  Overhaul**. Read selected user corrections, the 22 September fixes and
  evaluation summaries, and the 6 October handoff.
- `99809d60-d4ff-4e3e-a457-624189bfa764`: prior resume/master-weight audit.
- `84c05c0f-f22e-42ce-9600-57e1ed7ec419`: selected original training logs and
  user reports, including an explicit 11 September launch at `57f867a`.

This was targeted recovery, not a complete reading of all 19 chats. Original
chat step IDs below are retained in the JSONL records; they do not always equal
the physical line number.

## Prior exclusions that should not be rediscovered

The 19 September opening user message (step 0) explicitly records earlier
investigations of causal-mask precision, cross-core loss-coefficient variance,
loss-pooling algebra as a backend differentiator, permanent task-slot effects,
lexical skew, effective batch/token/LR accounting, and raw-generation checks.
The 16 September user correction in the evaluation-overhaul chat (step 3502)
also explicitly rejects the BF16-parameter-underflow explanation.

These records establish what was already investigated and Ivan's direction
for continued work. They should not be replaced with generic new suggestions
to check batch size, count devices, or disable the intended BF16 compute path.

## Claimed resolutions versus supporting evidence

### 1. "Distributed AdamW variance damping solved the gap"

The 19 September closing answer (step 225) makes this claim, but does not
demonstrate it. Both accumulation and replica averaging ordinarily aggregate
gradients before a single AdamW moment update. If their resulting gradient and
optimizer state agree, their moments do not differ merely because aggregation
was distributed. Different nonlinear loss reductions or arithmetic are separate
questions; the claimed mechanism is not a proof of the observed gap.

A small local CPU float64 sanity check used identical initial parameters,
20 prescribed updates, eight gradient contributions, and identical AdamW
hyperparameters. Mean aggregation versus scaled sequential accumulation gave
maximum parameter and second-moment differences of zero. Re-averaging identical
copies changed the aggregate by at most 2.22e-16. This is an algebra check, not a
CUDA/XLA experiment or validation of real BF16 kernels.

### 2. "Double reduction attenuates gradients by world_size again"

The comment in `b286a90` says a second reduction would introduce a further
factor of 1/world_size. That is not the mathematical behavior of a repeated
mean over already identical gradients. It can add communication and numerical
rounding; a universal extra attenuation factor does not follow.

Official PyTorch/XLA v2.8.0 source confirms `reduce_gradients` sums with scale
1/world_size. `optimizer_step` calls `reduce_gradients`, then `optimizer.step`,
and synchronizes only when its optional barrier is enabled. Therefore the
13 September plan's converse claim, that this wrapper necessarily supplies
missing special optimizer graph tracking, is also unsupported by that source.

Source:
https://github.com/pytorch/xla/blob/v2.8.0/torch_xla/core/xla_model.py

The exact historical installed XLA build was not established here. These source
checks reject broad API explanations; they do not prove the historical run had
no runtime-specific problem.

### 3. "BF16 cross-entropy underflow was identified repo-wide"

The 13 September plan states this as settled, but was still a proposed patch
in its closing answer (step 138). PyTorch/XLA v2.8.0 registers cross-entropy,
log-softmax, and BCE-with-logits with FP32 autocast policy. BF16 forward logits
alone do not establish BF16 loss computation inside autocast, much less actual
underflow. This is another reason not to relaunch generic precision debugging.

Source:
https://github.com/pytorch/xla/blob/v2.8.0/torch_xla/csrc/autocast_mode.cpp

### 4. An exact CE threshold for formula copying

Step 225 claims formula copying unlocks below approximately 1.40 nats.
The retrieved observations compare a few checkpoint outcomes; they do not
establish a universal threshold. Similarly, fitted convergence extrapolations
are not measurements of an eventual loss floor or extra training required.

## Last productive training-code changes recovered

Verified against Git, not just the chat's proposed commit commands:

| Date | Commit | Change |
|---|---|---|
| 20 September | `def9953` | Global seed initialization and XLA RNG synchronization |
| 22 September | `a8a2386` | Configurable shared default for LR, objective, and metric cadences |
| 22 September | `ebc7cae` | Per-task per-active-token mean loss normalization |
| 22 September | `3c1331b` | Cadence-dependent EMA decay |
| 22 September | `44ccf63` | Pass actual metric cadence to the tracker |

The loss-normalization commit removes both the sub-batch fractions and the
full-sequence denominator on masked infill. This is a genuine objective change,
not a demonstration that the earlier backend gap was caused by that objective.
The EMA change addresses a genuine cadence asymmetry, but its linear decay
adjustment is approximate: matching a heuristic time constant is not identical
to matching every fine-grained EMA trajectory.

The October 6 recap (evaluation-overhaul step 11235) says the next empirical
step is new training with the corrected code. The completed HumanEval and
Private Unseen figures cited there evaluate existing `50m_seed` and
`50m_lightning` weights. They are not a completed paired backend retraining
after all these changes. No such completed comparison was found in the reviewed
segments; that is a coverage statement, not proof none exists elsewhere.

## A concrete unfinished diagnostic

In the 19 September investigation:

- Step 203 launches `scratch/compute_gradient_norms_and_delta.py`.
- Step 204 starts background task `task-204`.
- Status records show progress through the 50M, 75M and 100M models.
- Steps 223–224 cancel it. The retained task log has progress messages but no
  final result tables. Step 225 nevertheless immediately declares a resolution.

The surviving script also needs correction before its results are usable:

1. Both forward calls omit `mask_override`. Loaded COROSred checkpoints are
   causal by default, so the purported infill gradient is computed with causal
   attention rather than explicitly bidirectional attention.
2. It compares 50M CUDA to 75M/100M TPU checkpoints, omitting the available
   same-size 50M TPU controls.
3. Gradient magnitudes at trained endpoints are descriptive. They cannot by
   themselves establish which backend operation caused the training gap.

## Recommended next work

Finish that diagnostic locally using the saved 50M CUDA and 50M TPU checkpoints,
explicit `mask_override=True` for causal and `False` for infill, and fixed input
and corruption tensors. Record separate task-gradient norms, their cosine,
and weighted combined gradients by layer under a declared objective version.
Separate descriptions of learned endpoints from causal claims about training.

This resumes an interrupted investigation without repeating the closed
batch/data checks or requiring a fresh TPU allocation. It can identify which
objectives and parameter blocks merit instrumentation when TPU time is next
available. It cannot substitute for eventual runtime evidence from that device.
