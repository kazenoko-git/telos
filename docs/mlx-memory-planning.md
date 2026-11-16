# MLX memory planning and variable accumulation

MLX pretraining chooses its microbatch from the model geometry and the machine's
unified memory. The requested **effective batch** stays fixed:

```
effective sequences per update = microbatch × gradient accumulation
training ceiling = min(60% of system RAM, Metal recommended working set,
                       optional autotune_memory_gb cap)
training allocation budget = training ceiling − 1 GiB
```

The 1 GiB reserve is inside the ceiling, not added above 60%. For example, a
24 GiB machine has a ceiling of at most 14.4 GiB and an allocation budget of at
most 13.4 GiB. GiB means 1,073,741,824 bytes, so the reserve exceeds decimal 1 GB.
A smaller device recommendation or configured cap lowers those values.

The planner considers every exact divisor of the effective batch and chooses
the largest estimated fitting microbatch. If the full effective batch fits,
accumulation becomes 1. Otherwise, accumulation increases to preserve the
requested effective batch. For example, an effective batch of 64 with a fitting
microbatch of 8 uses accumulation 8. Batch planning runs even when measured
autotuning is disabled or batch sizes were supplied explicitly.

Existing explicit gradient checkpointing stays enabled. Otherwise the planner
first tries microbatches without checkpointing. If even microbatch 1 cannot fit,
it retries with checkpointing unless checkpointing was explicitly disabled. If that still fails, it raises an error before
training: accumulation cannot make the model and optimizer state themselves fit.
Reduce the model or context length in that case.

## Estimates and measured tuning

The estimate includes parameter and FP32 optimizer/master/update storage,
layer activations, vocabulary logits, and fallback attention matrices, with a
1.8 safety factor. It is a conservative planning estimate, **not a calculated
exact maximum or an operating-system memory guarantee**. Actual allocation
changes with MLX kernels, objective, checkpointing and temporary workspaces.
The fixed headroom remains separate from that safety factor.

Measured autotuning remains available through the existing `autotune` setting.
It uses the same allocation budget, screens candidates by the estimate, and
selects the largest microbatch among successful trials whose observed peaks
fit. At equal microbatch sizes it prefers no checkpointing, then lower memory.
Throughput is still recorded but does not override the memory policy. Warmup
peaks count as well as measured-update peaks. The optional `autotune_memory_gb`
cap can make the ceiling smaller; it cannot exceed 60% of system RAM.

MLX memory and wired-memory limits are advisory. These changes do not reserve
physical RAM against other applications, perform live microbatch resizing during
an update, or guarantee against allocation failure under changing system load.

## Inspecting a run

Startup prints the microbatch, accumulation count, checkpointing mode, estimated
peak, headroom and ceiling. `training.memory_plan` in the saved training config
records the plan; a measured run also records its selected observed peak.
The existing `logs/autotune_mlx_*.json` report includes candidate measurements,
the allocation budget, headroom, ceiling and selection policy.

Implementation: `telos/training/mlx_memory.py`, `trainer_mlx.py` and
`mlx_tuning.py`. CUDA and XLA batch planning are outside this change.

## Validation

On 2026-11-16, the following focused suite passed **29 tests** with local Metal:

```sh
.venv/bin/python -m pytest -q tests/test_mlx_memory.py \
  tests/test_mlx_tuning.py tests/test_mlx_step.py \
  tests/test_mlx_compiled_accumulation.py tests/test_mlx_unified.py
```

Coverage includes budget arithmetic, fixed headroom, capped budgets, exact
non-power-of-two batches, accumulation and checkpoint fallback, untrainable
models, measured candidate selection, isolated tuning state, existing update
parity/RNG checks, and a two-update synthetic training run that tunes and saves
the final memory plan. The combined v1.7.4 validation is recorded in
`docs/releases/v1.7.4-validation.md`. This is functional validation, not a new throughput or
large-model peak-memory benchmark.
