# Saved 50M optimizer replay — 18 October 2026

## Result

Ordinary and fused AdamW on this Mac's CPU matched bit for bit across eight prescribed updates: all 49,430,017 parameter values, first moments and second moments. There were zero unequal elements at every measured update. This is a CPU optimizer-arithmetic result, not a physical CUDA/XLA comparison and not a diagnosis of the historical training gap.

## Execution contract

- PyTorch 2.13.0, CPU FP32, two CPU threads; no MPS use or interference with the ongoing Mac model evaluation.
- Start from `checkpoints/corosred/50m_lightning/checkpoint_final.pt`, including its saved optimizer moments at step 12,716 and saved group hyperparameters/LR.
- Both variants receive independently cloned weights and optimizer states. Pointer checks reject state aliasing. Dispatch spies confirmed 16 calls each to the single-tensor and fused implementations (eight updates, two groups each).
- Use the existing fixed-fixture captured FP32 gradients from that same checkpoint. SHA-256 verifies the checkpoint/fixture/capture chain.
- Prescribed gradient scale stream: 1, 0.5, 1.5, -0.25, 1, 0.75, -0.5, 0.01. Global clipping threshold 1 is applied identically before each update. The first seven updates clip; the final update does not. This is an operator replay with prescribed gradients, not a continuation of the training trajectory.
- Group sizes and every moment-tensor shape match current parameter order. The historical checkpoint does not contain a parameter-name manifest, so exact historical name-to-state ordering is not independently proven. This caveat does not invalidate the comparison of two implementations given identical numerical inputs.
- Tests verify replay states remain independent and that deliberately different gradients produce detectable update differences. Nine tests passed across optimizer replay, cadence causal paths and the saved-checkpoint gradient evaluator.

## What it establishes

No discrepancy was found between the two CPU AdamW paths for this saved state and gradient stream. The experiment does not exercise CUDA fused kernels, XLA lowering, collective reduction, BF16 forward/backward execution, or historical TPU optimizer moments. It cannot exclude an accelerator-specific optimizer issue.

## Hardware boundary

This session has MPS available, CUDA unavailable, and no installed torch_xla runtime. Kaggle credentials are not configured. An actual TPU/CUDA capture could not be executed from this environment. The existing fixed-fixture backend probe can capture per-device forward outputs, gradients and updates, but no new physical TPU/CUDA result was produced here. Its single-replica design also does not validate distributed reduction.

To distinguish a hardware/runtime issue from an application issue, the remaining comparison must run the same FP32-parameter/BF16-compute fixture on those devices, capture pre/post-reduction and pre/post-clipping gradients, and compare actual updates. Calling the unexplained deficit a "TPU quirk" would leave the cause unknown.

Artifacts: `scripts/replay_saved_optimizer.py`, `tests/test_saved_optimizer_replay.py`, `logs/saved_optimizer_replay_20261018.json`. Production training code was unchanged. No commits, pushes, uploads or cloud jobs were made for this replay.
