# 100M unified COROSred checkpoint comparison — 18 October 2026

Matched CPU FP32, explicit causal/bidirectional attention,128 sequences of512 tokens across4 nonadjacent corpus slices; identical20% normal-token corruption and strict checkpoint loading. These are endpoint measurements, not a controlled training-backend experiment.

| Checkpoint | Causal CE | Causal top1 | Infill CE | Infill top1 | LRH AUROC |
|---|---:|---:|---:|---:|---:|
| 100m_r50_modal_h100_seed42_cad1 | 1.360218 | 70.62% | 0.589983 | 86.89% | 0.4957 |
| 100m_5b | 1.726202 | 65.93% | 1.195184 | 76.88% | 0.4918 |
| 100m_5b_alpha075 | 1.741622 | 65.62% | 1.295387 | 75.32% | 0.4060 |
| 100m_schedstep1 | 1.730354 | 65.78% | 1.305765 | 74.94% | 0.4086 |
| 100m_tpu_2 | 1.737497 | 65.57% | 1.210185 | 76.56% | 0.4868 |

All five completed checkpoints pass strict loading and declare the same geometry; actual unique parameters105,995,521 including the reliability head. The legacy top-level100m checkpoint is excluded because it is a one-step synthetic PhaseA artifact.

The new H100 checkpoint has the lowest aggregate CE on both tasks among the evaluated endpoints. Earlier runs differ in alpha floor, causal split and/or cadence; their declared device is XLA, but exact training-source/runtime provenance is incomplete. The pending new v5e-8 run is not included. Corpus slices are outside the first5B positions of the larger15B file; document-disjoint held-out status is not established.

## Per-slice comparison against new H100 endpoint

- 100m_5b: old minus new causal CE per slice[0.3542, 0.4401, 0.3824, 0.2873]; infill CE[0.6729, 0.6291, 0.5806, 0.5378].
- 100m_5b_alpha075: old minus new causal CE per slice[0.3705, 0.4454, 0.4063, 0.3035]; infill CE[0.8045, 0.7291, 0.665, 0.6223].
- 100m_schedstep1: old minus new causal CE per slice[0.3544, 0.4399, 0.3865, 0.2997]; infill CE[0.7752, 0.7487, 0.6759, 0.6632].
- 100m_tpu_2: old minus new causal CE per slice[0.3576, 0.4449, 0.4055, 0.301]; infill CE[0.6963, 0.6529, 0.5847, 0.5465].

Raw per-sequence metrics, checkpoint hashes, full embedded configurations, source hashes and fixture fingerprints are in results.json; the exact tokens and masks are in fixture.pt. HumanEval is being evaluated separately with the8192-token tokenizer and the complete164-task suite.
