# Cadence replay and causal-path audit — 18 October 2026

The cause of the saved 50M TPU/CUDA quality gap remains unidentified. This audit reduces support for cadence as the sole large-gap explanation; it does not prove cadence harmless or assign another cause.

## What was executed

`scripts/audit_cadence_paths.py` replays current scheduling code and historical scheduling/loss code at Git revision `a2f09a06eefa836c91efe8f937646127f754ece5` (17 September), with CPU execution and source hashes. This is a representative historical implementation, not a proven source revision for either saved training run. No TPU/CUDA kernel comparison or historical training reconstruction was performed.

All synthetic metric replays receive the same per-step signal: 0.50 before step 1,000 and 0.60 thereafter, with identical accuracy and ranking inputs. Scheduler weights read the previous step's metric state, as in the trainer. Metric sampling differs by cadence. These are invented probe observations, not recovered run telemetry.

| Replay | First step with gamma active |
|---|---:|
| Historical metric cadence 1 | 1,069 |
| Historical metric cadence 25 | 2,701 |
| Current metric cadence 1 | 1,069 |
| Current metric cadence 25 | 1,051 |

The historical EMA is demonstrably slower in physical training steps when updated once every 25 steps. Current linear cadence compensation largely removes this specific synthetic lag but is not an exact reconstruction of every-step observations. Nominal alpha/beta are identical across all four metric-only replays, because they depend on step progress rather than those metrics. The saved configurations disable adaptive loss rebalancing.

## LR and objective step quantization

Using the saved 50M schedule parameters across 12,716 updates:

| LR cadence | Change in sum of applied LR versus cadence 1 |
|---|---:|
| 10 | +0.0580% |
| 25 | +0.1547% |

Maximum absolute nominal alpha changes from objective-step quantization are 0.000773 at cadence 10 and 0.002059 at cadence 25; maximum beta changes are 0.000442 and 0.001175. These measure schedule values, not resulting optimization trajectories. Small schedule differences do not mathematically rule out amplified training effects. The eligible saved cadence-1 experiment already retained a substantial quality deficit.

## Does metric cadence reach the backbone?

Source inspection and controlled backward passes establish the following in the inspected current and historical implementations:

1. Metric feedback controls gamma and the intended masking blend, not nominal alpha/beta.
2. The reliability head consumes `h.detach()`. Its BCE therefore does not directly backpropagate into the transformer backbone.
3. The trainer's routing refresh guard calls `schedule.get_weights(step)` without LRH metrics. That call always returns `mask_blend=0`, even when `get_weights(step, .8, .8)` would return 1. Consequently the trainer does not populate a newly empty routing cache through this guard. Both inspected versions share this problem.
4. An empty routing cache generates uniform random masks even if the training loss receives a positive masking blend.
5. With identical tiny-model weights, inputs and random seed, toggling gamma from 0 to 0.1 and blend from 0 to 0.8 gives an exact maximum backbone-gradient difference of **0 before clipping**, in both historical and current loss implementations. Head gradients change as expected.
6. Global gradient clipping includes the head and backbone together. This leaves an indirect path: changing head gradients can rescale backbone gradients when clipping is active. The configured threshold 1 was inactive in this tiny probe. Forcing threshold 0.25 produced a nonzero backbone-gradient difference (current loss: max absolute 0.00009583). This is a reachability demonstration, not a measurement of clipping effects during the original 50M training.

The routing refresh omission is a confirmed implementation defect in the inspected sources. It affects both paths and is not established as the TPU-specific cause. It was not changed here. Enabling routing now would change the training experiment and must not silently be folded into a parity replay.

The CUDA/CPU validation monitor additionally writes probe ROC-AUC into the tracker at checkpoints, whereas multi-core TPU skips that probe. Training-time accelerator feedback uses balanced accuracy as its ranking proxy. This is a real source-level feedback asymmetry, but the detached-head/empty-cache constraints and the clipping caveat still apply. No claim is made that this asymmetry explains the observed loss gap.

## Conclusion and remaining evidence gap

- Historical metric cadence is a real mismatch, already addressed by later fixes.
- The tested direct path from metric cadence to backbone gradients is absent when the routing cache stays empty and rebalancing is disabled; global clipping remains an indirect path.
- LR/objective cadence-1 saved weights still underperform, so a generic cadence diagnosis is insufficient.
- The first eligible TPU intermediate already underperforms by step 4,000, but sparse weights do not record the gradients, clipping coefficients, Adam moments or backend execution that produced it.
- A decisive backend diagnosis needs the same starting state and inputs with captured pre/post-reduction gradients, clipping norm/coefficient and actual parameter updates under the intended FP32-parameter/BF16-compute execution. Pure schedule replay and checkpoint evaluation cannot substitute for that evidence. A short trace can be captured when TPU access exists and compared later; simultaneous allocations or full retraining are not required.

Artifacts: `logs/cadence_path_audit_20261018.json`, `scripts/audit_cadence_paths.py`, and `tests/test_cadence_path_audit.py`. Validation: 20 tests passed across the new causal-path checks and existing schedule/unified-loss tests. Production training code was unchanged; no commits or pushes were made for this audit.
