# Kaggle version-log findings — 20 October 2026

**Subsequent downloads:** The user supplied three additional alternate-account logs, now preserved in `kaggle_logs/` with `alternate_account_manifest.json`. They contain three completed50M COROSred runs and one completed100M AR run, all logging eight rank-zero identities per run. The AR log records commit `b286a90`. Pending-download/coverage statements below describe the earlier inspection stage; see the [primary consolidated report](../TPU_RANK_COLLAPSE_BUG_REPORT.md).

Read-only inspection in the existing signed-in Arc/Kaggle session (`kazenoko`). Full logs were downloaded through Kaggle's visible Download logs control, copied locally, and hashed. Notebook titles were not used to infer model size.

| Version | Script version ID | Actual model/run | Completion | Worker rank messages |
|---|---|---|---|---|
| 8 | 351499919 | Unified50M,50m_seed,12,716 steps | Completed; uploaded checkpoints | Eight rank0 messages |
| 7 | 351383019 | Unified50M,50m_seed,25,432 planned steps; save every200 | Kaggle says canceled after11,661.1s; log ends during step10,200 save at4,152.7s | Only master/startup rank0 flushed; other ranks unknown |
| 4 | 350392839 | Unified75M,75m_python,19,074 steps | Completed; uploaded checkpoints | Eight rank0 messages |
| New10 | 359630346 | Unified100M,100m_r50_tpu_v5e8_seed42_cad1 | Completed; checkpoint independently evaluated | Each rank0–7 exactly once in saved train.log |

## What this establishes

The repeated-rank observation is now present in directly downloaded Kaggle logs from completed later unified runs, not only an early archived conversation. Version4 explicitly reports74,647,680 target parameters and saves/uploads75m_python. Version8 reports49,166,848 target parameters and saves/uploads50m_seed. Both print eight rank0 messages. This is incorrect replica identity for the inspected eight-worker replicated trainer, whose self.rank also determines data offsets. The new run has correct distinct rank identities.

With batch48 and8 workers, all-rank0 indexing presents48 distinct sequence rows per optimizer step across384 presentations, instead of384 distinct rows. It advances by384 rows per step. Random masks can still differ; duplicate raw input does not imply identical gradients. The exact batch streams were not captured from these historical runs.

The invalid xm.get_ordinal() call plus silent exception fallback0 is the verified compatible code mechanism. The new xr.global_ordinal() patch reproduces the successful trainer's exact recorded hash. Old logs do not independently print exact Torch/XLA version or source SHA, so API-failure attribution remains tied to the inspected source rather than an exception trace in these logs.

## Version7 cancellation

Confirmed: free disk after dataset download was displayed as10.1GB (the notebook formats GiB), checkpoints214.5MB each, save every200 steps. Exactly50 checkpoint saves are verified through step10,000. Log ends at the51st save (step10,200), without confirmation, traceback, training-complete footer, final checkpoint or upload.

Storage arithmetic strongly fits exhaustion:50×214.5MB≈10.725GB≈9.99GiB;51×214.5MB≈10.940GB≈10.19GiB. The displayed10.1GiB free-space rounding interval is below10.19GiB. This supports a checkpoint-write stall/failure followed by later cancellation. No explicit ENOSPC error is present; exact cancellation actor/reason is unknown. The difference between last-log time69.2min and cancellation194.4min is consistent with a stall but does not prove it.

Version7 therefore is not a completed quality endpoint and must not be treated as one.

## Boundaries and remaining coverage

Version2 was identified by the user as another75M run; its log download was pending at this report revision. The existing kazenoko session was inspected; the alternate account was not yet inspected. Specific older100m_schedstep1 and50m_tpu_lrcadence_schedstep1 files are still not cryptographically linked to these downloaded logs. The new findings strengthen a shared historical rank/sharding diagnosis without claiming every historical checkpoint has been individually attributed.

Raw logs and a SHA256 manifest are in kaggle_logs/. No notebook was run, edited, saved, deleted or published.
