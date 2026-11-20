# Structural Optimization Validation — 2026-11-20

This note records the controlled candidate screening run on Apple M5 Pro
(`checkout 3eb97d4d`) and the follow-up PyTorch/CUDA/XLA structural changes.

## Candidates Dropped Based on Local MLX Evidence

Two repeat runs of `scratch/exp_v18x_candidates.py` on M5 Pro (MLX 0.32.2,
`d_model=512`, 14 layers, `vocab=8192`, `T=256`, `batch=8`, `accum=8`) ruled out:

- **MLX FP32 grad clipping + AdamW:** `-22.86%` and `-22.99%` regression.
- **MLX 2D packed SwiGLU Metal kernel:** `-51.30%` and `-0.04%` (no win).
- **MLX 3D QKV + RoPE Metal kernel:** `+8.58%` then `+0.52%` (did not hold).
- **MLX online-softmax CE kernel:** `+10.47% / +15.32%` isolated, but `+0.12% / -0.38%` end-to-end on 50M AR with a `0.0017` final BF16 loss shift.

## Surviving MLX Lead (Pending Fresh-Process Metal Validation)

- **`mlx_reliability.py` (`reliability_metrics`):** Measured `+5.08% / +5.53%`
  in the shared-process harness, but shared-process peak memory rose due to
  evaluating both causal and infill graphs simultaneously. A fresh-process ABBA
  validator (`scratch/validate_mlx_reliability.py`) evaluates causal outputs and
  releases causal logits before the infill pass, and remains pending
  non-sandboxed Metal execution before any MLX edit is promoted.

## Implemented PyTorch / CUDA / XLA Structural Changes

All changes below were verified for forward and backward parity locally; physical
CUDA and TPU throughput gains remain unverified until measured on hardware.

| Area | Change | Local Verification |
| --- | --- | --- |
| RoPE (`components.py`, `transformer.py`) | Hoist `cos`/`sin` dtype cast and unsqueeze once per forward; remove dead `is_xla` string check | Bitwise exact forward & backward (`test_hoisted_rope_matches_per_layer_cast_bitwise`) |
| RMSNorm (`components.py`) | Use native `F.rms_norm` (PyTorch >= 2.4) with FP32 variance fallback | FP32 parity `<= 1e-5`; BF16 max error vs FP32 ref improved `0.1021 -> 0.0628` |
| AR CE (`diffusion/ar.py`) | Single-shot `F.cross_entropy(..., reduction="mean")` removing 4096-token chunk+cat loop | Bitwise exact (`0.0` loss diff, `0.0` grad diff) above 4096 tokens |
| GQA (`components.py`) | Broadcast `expand` + `reshape` instead of `repeat_interleave` | Bitwise exact forward & backward |
| Routing cache (`corosred_unified.py`) | `torch.topk(..., largest=False)` instead of full `torch.argsort`; direct `>= 0.0` instead of `sigmoid >= 0.5` | Exact mask count & BOS preservation |
| CUDA prefetcher (`prefetch.py`) | Stage pinned host buffers in `int32` when token dtype fits, casting to `torch.long` on copy stream | Exact `torch.long` output for `uint16`/`int32` |
| CUDA trainer (`trainer_pytorch.py`) | Enable cuDNN TF32, `cudnn.benchmark = True`, and DDP `gradient_as_bucket_view=True` | Unit/regression suite |
| XLA cache & graph hygiene (`xla_utils.py`, `trainer_pytorch.py`, `transformer.py`) | Opt-in `configure_xla_persistent_cache`, fuse COROSred `xm.all_reduce` before `xm.mark_step()`, skip per-step `xm.set_rng_state` on deterministic AR, opt-in `xla_mark_step_per_microbatch`, native `torch_xla` checkpointing | Mocked-XLA call-order & dispatch tests (`tests/test_xla_structural_optimizations.py`) |
