# Independent OpenCode review assessment — 18 October 2026

Both explicitly selected free models returned read-only reviews. Raw reports are saved unchanged as `deepseek_review.md` and `glm_review.md`; they are external-model claims, not validated findings. OpenCode step-finish events reported total model cost0. No Codex reviewer was resumed, no production code was edited, and no new training/evaluation experiments were launched.

## Result

Neither established the root cause of the saved TPU/CUDA quality gap. Both prioritized TPU optimizer-state reset on resume. The local source check confirms `telos/train/cli.py:359–368` skips loading optimizer state on TPU while attempting it for CUDA/CPU when state is present. This matters only if the affected historical run actually resumed; those exact-run events remain unproven. A save sequence alone does not prove resumption. The prior unified archive report already documents this behavior and limits its causal attribution, so it is not a new discovery.

DeepSeek correctly rejects the code comment claiming two full averaging all-reduces shrink gradients by1/64. A second average of already equal replica gradients is mathematically idempotent. This was also already identified in `research/antigravity_archive_review_2026-10-18/unified/findings.md`, so it strengthens an existing correction rather than solving the gap. Its claim that all reset effects relax over10steps ignores the second-moment time constant. Its categorical claim that remaining differences must be kernels is stronger than its own unresolved resume/recipe confounds permit. File sizes support, but do not independently prove, exact checkpoint contents; the report also has an inconsistent50MB/214MB number.

GLM identifies historical objective changes, but its confident assignment of revisions to the CUDA run is not established. Its own chronology places the upload about5.25hours after a commit; a presumed run lasting at most3.5hours could fit entirely after that commit. The argument therefore does not prove the run used the pre-commit implementation. Upload time and assumed throughput cannot establish training source. It also confuses the candidate endpoint hash with the baseline endpoint and infers resumption from successive saved checkpoints. These claims are rejected. Neither local file mtimes after copying nor file upload timestamps reliably identify original training sessions.

Both note metric cadence may not have been set to1 when LR and objective cadence were set to1. This was already bounded in prior cadence work; the detached head, shared routing defect and clipping effects limit causal interpretation. Historical EMA code must be distinguished from the later compensated version; DeepSeek mixes those eras in parts of its reasoning.

## Paper-safe takeaway

Two independent external-model reviews did not isolate a cause. They found or repeated code/recipe hazards, notably conditional optimizer-state reset, but supplied no new evidence that those mechanisms caused the evaluated gap. Do not promote their ranked candidates into a diagnosis. The user's supplied identical-starting-weights condition was included in both briefs.
