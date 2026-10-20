# Successful TPU run versus historical unified TPU checkpoints — 20 October 2026

**Evidence update:** Subsequent direct Kaggle-log recovery confirms repeated rank-zero identities in completed later unified runs and a completed AR run whose log records source commit `b286a90`. The original comparison below predates those downloads. The consolidated current account is the [primary rank-collapse bug report](../TPU_RANK_COLLAPSE_BUG_REPORT.md).

## Finding

The new100M TPU endpoint eliminates the large historical endpoint-quality deficit. Historical schedule-only and LR+schedule cadence-one checkpoints really do exist and remain worse; cadence alone is not established as the explanation. A targeted archive search now finds direct historical user-pasted logs with repeated rank0 worker identities. This increases support for the rank-API/data-sharding defect as a contributor, but does not link that defect conclusively to each later saved50M/100M checkpoint.

## Eligible checkpoint inventory

Only explicitly unified COROSred finals were included. Legacy separate PhaseA/B/C models were excluded. Configurations below were read from checkpoint contents, not inferred from folder names. Missing fields mean unspecified, not a measured default.

| Checkpoint | Steps | Alpha floor | Causal ratio | Saved LR cadence | Explicit schedule cadence | Explicit metric cadence | Saved LRH AUC EMA |
|---|---:|---:|---:|---:|---:|---:|---:|
| 100m_r50_tpu_v5e8_seed42_cad1 | 25432 | 0.6 | 0.65 | 1 | 1 | 1 | 0.7690 |
| 100m_5b | 25432 | 0.5 | 0.65 | 10 | unspecified | unspecified | 0.4929 |
| 100m_tpu_2 | 25432 | 0.5 | 0.65 | 10 | unspecified | unspecified | 0.4817 |
| 100m_5b_alpha075 | 25432 | 0.75 | 0.65 | 10 | unspecified | unspecified | 0.5099 |
| 100m_schedstep1 | 25432 | 0.65 | 0.7 | 10 | 1 | unspecified | 0.4770 |
| 50m_python | 12716 | 0.5 | 0.65 | 10 | unspecified | unspecified | 0.4968 |
| 50m_tpu_schedstep1 | 12716 | 0.5 | 0.65 | 10 | 1 | unspecified | 0.5022 |
| 50m_tpu_lrcadence_schedstep1 | 12716 | 0.5 | 0.65 | 1 | 1 | unspecified | 0.5102 |

All8 inventoried finals have FP32 saved model tensors and equal saved embedding/output tensors. This does not prove historical computation dtypes or runtime tying. All8 omit optimizer state. Older runs have no runtime/source-hash sidecars in these local folders; the new run records Torch2.8.0+cpu, Torch/XLA2.8.0, libtpu0.0.17 and the patched trainer/CLI hashes.

## Comparable100M endpoint measurements

Existing common-fixture CPU FP32 results:128 sequences, length512, identical random masks and explicit causal/bidirectional attention. No retraining or regenerated fixture.

| Endpoint | Causal CE | Infill CE |
|---|---:|---:|
| 100m_5b | 1.726202 | 1.195184 |
| 100m_tpu_2 | 1.737497 | 1.210185 |
| 100m_5b_alpha075 | 1.741622 | 1.295387 |
| 100m_schedstep1 | 1.730354 | 1.305765 |
| 100m_r50_tpu_v5e8_seed42_cad1 | 1.356237 | 0.593795 |
| 100m_r50_modal_h100_seed42_cad1 | 1.360218 | 0.589983 |

## Historical50M cadence experiments

Previously saved same-fixture16-sequence FP32 measurements, separate from the128-sequence100M fixture above. These are smaller models and are not a direct scale-matched comparison with the new100M model.

| Endpoint | Causal CE | Infill CE |
|---|---:|---:|
| cuda_lightning | 1.475287 | 1.073203 |
| tpu_baseline | 1.986974 | 1.860621 |
| tpu_schedstep1 | 1.963892 | 1.766006 |
| TPU LR+schedule cadence1 | 1.994002 | 1.778208 |

The LR+schedule-one candidate saves actual scheduler update_cadence1. Its config does not explicitly set metric cadence1. Historical source a2f09a0 defaults metric synchronization to25 on TPU; exact run/source linkage remains unproven. Therefore it is not a verified all-cadence-one ablation.

## Newly recovered rank evidence

Archive: `.agent-history/antigravity-ide/chats/84c05c0f-f22e-42ce-9600-57e1ed7ec419/transcript.jsonl` (Unified Trainer Development Plan). Targeted search covered all19 archived transcript.jsonl files for literal topology messages; this is not a new full semantic review of the archive.

- Line11646, source USER_EXPLICIT, timestamp2026-09-10T18:29:57Z: user-pasted completed unified training log,3815 steps, checkpoint path `checkpoints/corosred/unified/`; seven literal worker topology messages all say `(8 Cores, Rank 0)`. The earlier user startup paste at line10976 also says rank0. This appears to be an early smaller unified run, not a directly identified50M/100M cadence endpoint.
- Line6674, USER_EXPLICIT,2026-09-08: eight rank0 topology messages in an older separate-phase run that then errors. It is corroborating runtime evidence only and is excluded from unified checkpoint-quality comparison.
- The successful new100M train.log contains exactly one topology message for each of ranks0–7.

## Verified mechanism and scope

1. Git b19e133 (7September) introduced rank-partitioned data indexing, but obtained rank through xm.get_ordinal(), catching every exception and assigning rank0. Representative historical source a2f09a0 retains that code.
2. Official XLA v2.8.0 xla_model.py does not define get_ordinal; runtime.py defines global_ordinal. Tagged sources and hashes were saved locally. Sources: https://raw.githubusercontent.com/pytorch/xla/v2.8.0/torch_xla/core/xla_model.py and https://raw.githubusercontent.com/pytorch/xla/v2.8.0/torch_xla/runtime.py .
3. The fresh notebook patch uses xr.global_ordinal() without silently substituting0. Applying only that replacement to the current local trainer reproduces the exact new-run recorded trainer SHA256: 888d21b53e734192ed7cb560b352de8fc36125855c5bd81ddd9bc7263312381b.
4. Source indexing uses start=rank*48 and advances by384 for8 replicas. A pure index-arithmetic replay confirms all-rank0 yields48 distinct sequence rows per step, whereas ranks0–7 yield384. Both still execute384 sequence presentations across the8 workers. Random masks/dropout can differ, so this does not imply identical gradients.
5. Thus the failure changes actual input diversity and corpus traversal, not merely display or nominal batch accounting. At512 tokens, those cases expose24,576 versus196,608 distinct-position input tokens per step. This arithmetic is conditional on the inspected replicated path; no new TPU execution was performed.
6. It can affect AR and COROSred because both share this trainer. It survives LR and objective cadence fixes, which do not modify rank acquisition.

## Other simultaneous changes and what remains unresolved

- New run explicitly sets metric, LR and objective cadences1. Old LR+schedule-one does not establish every-step metric synchronization. The September22 tracker-cadence compensation fix is also newer than the representative older scheduling implementation.
- Alpha floor changed from.50/.65/.75 to.60, and one older100M schedule-only run used causal ratio.70 instead of.65. These are recipe differences, not isolated hardware tests.
- New notebook removes XLA_USE_BF16 and XLA_DOWNCAST_BF16. Old environment values are unavailable; FP32 saved tensors alone cannot reconstruct their values.
- Source revisions, ranks and exact Torch/XLA runtime versions are not independently recorded for the later original/schedule-only/LR+schedule-one endpoints. No user-exact later-checkpoint startup rank log was found in the targeted archive search.
- New-run metric feedback and gamma activation differ from older runs. Historical cadence replay already established limitations on a direct backbone-gradient effect due to detached head features and inactive routing-cache refresh; global clipping remains an indirect path.
- The known50M path/content mismatch remains: local50m_python final is not the same file as the LR+schedule-one final that matches pinned HF50m_python final. The new inventory preserves hashes to prevent silently merging lineages.

## Assessment

The strongest newly supported lead is incorrect replica rank acquisition, rather than a generic TPU arithmetic penalty or a claim that LR cadence1 was never applied. Historical all-rank0 logs, the invalid API call, its data-indexing consequences, and the new patched successful run form a concrete consistent mechanism. They do not yet prove this was the cause of every later named checkpoint deficit. Exact older startup logs/runtime/source evidence would close that remaining attribution gap; this report does not claim a controlled fix ablation.

Production source was not modified, no training/cloud job was launched, and no commit or push was made. New artifacts: checkpoint_inventory.json, archive_rank_evidence.json, archive_rank_context.json, archive_targeted_runs.json, rank_api_and_index_verification.json and version-tagged XLA sources.
