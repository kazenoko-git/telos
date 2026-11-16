# Full-training attention confirmation protocol

This follow-up uses the merged v1.7.5 preparation source, an isolated MLX
0.32.4 runtime and the same 49,166,848-parameter BF16 AR model with FP32 master
weights/AdamW. Fourteen layers, width 512, eight equal Q/K/V heads and head
dimension 64 are fixed. Packed SwiGLU, compiled accumulation and forced fused
attention forward are held constant. The only compared option is
`MLX_SDPA_VJP_FALLBACK=0` (native backward) versus `1` (fallback backward).
Each job sets this before importing MLX.

For each context length the fresh-process order is native, fallback, fallback,
native. Length 2048 runs 128 updates at effective batch eight; length 256 runs
96 updates at effective batch 64. The production memory planner chooses the
actual microbatch/checkpointing under a 4 GiB ceiling with 1 GiB reserved.
Those plans must agree within each shape. Every initial weight export must
have the same SHA256. Both modes train on the same byte-verified, nonwrapping
real-corpus prefix and use the same seed, learning-rate schedule and updates.

## Warmup and guards

The first unchanged-protocol retry, like the earlier attempt, stopped while
loading/initializing model and optimizer storage. It is retained under
`full_attention_retry` and excluded from performance results.

For the complete `full_attention_steady` comparison, the first **32 updates**
are excluded from timings and establish the swap-growth baseline. This rule
was fixed before the first run in this comparison, and applies to all eight
runs. The 4 GiB MLX allocation guard, 25% system free-memory threshold and
256 MiB swap-headroom threshold remain active throughout setup and training.
After warmup, 512 MiB sustained swap growth also stops a run. System telemetry
is sampled every five seconds; critical system conditions require three
consecutive samples. Advisory MLX limits are not OS allocation caps.

The controller cancels only its own child on another local Python worker or
a 300-second job timeout. Interrupted runs are invalid, not shortened timing
samples. The interactive Mac can have other application GPU activity; system
GPU utilization is not a per-process occupancy measurement.

## Assessment

Primary rates count input tokens over synchronized, completed optimizer-update
times after warmup. Model/process setup and final checkpoint serialization are
outside that rate. Report individual rates, both adjacent-pair native gains,
their geometric mean and first/second timing-half rates. With two pairs, a
pair range is descriptive and is not a confidence interval.

After all timed jobs finish, compare every tensor in model, FP32 masters,
optimizer and MLX RNG groups, checking keys, shapes, dtypes and finite values.
Report numerical distances when they differ; do not treat approximate matches
as bitwise equality. Reconstruct the actual microbatch-shaped full-model
backward graphs with the same checkpointing/kernel options and verify whether
the dedicated native VJP is present.

Evaluate all final models through a common inference path on 32 contiguous
held-out sequences per context length from a fixed tail outside either
training range. Record individual losses and data hashes. This checks finite
short-run behavior and quantifies loss differences; it does not establish
convergence or general model-quality equivalence. Automatic dispatch/defaults
are outside this validation task.
