# MLX optimization agenda — 2026-11-18

The production memory planner owns microbatch/accumulation selection. Preserve
its memory ceiling and effective batch; do not add a batch-size sweep or increase
RAM to obtain a throughput result.

1. Measure compiled accumulation off/on with asynchronous accumulation enabled
   on the released trainer. Record the planner's actual microbatch, accumulation,
   checkpointing policy, allocation and clocks. The older frozen 8×8 workload is
   not a substitute for this check.
2. Verify attention dispatch. Télos already calls `mx.fast.scaled_dot_product_attention`;
   `force_fused_attention` controls a forward diagnostic option. Native/fallback
   backward selection belongs to MLX; Télos has no measured per-shape backward
   dispatcher. Keep the existing attention probes and full-training evidence,
   and verify the graph on the installed runtime before proposing a policy.
3. Profile accumulated-update clipping and optimizer work. Scaling/clipping is
   currently built outside the compiled optimizer function. Prototype moving
   those operations inside it, retaining first-update initialization, FP32
   master weights, moment dtypes, dynamic learning rate and explicit clip/scale
   arguments. Require state/gradient checks and a measured benefit before
   changing production behavior.
4. Last: a projection–cross-entropy primitive that controls buffer lifetimes.
   This is a potential v1.8 project, not part of the preceding default changes.
   Existing Python tiling did not establish a memory or speed win. Begin with
   exact masking/rounding semantics, a correctness reference and an explicit
   peak-allocation gate before considering a custom Metal implementation.

Do not combine component speedups into an end-to-end claim. Retain failed and
inconclusive measurements. Further releases require an explicit release decision.
