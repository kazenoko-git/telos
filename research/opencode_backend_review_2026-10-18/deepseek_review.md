## Independent read-only review — Telos 50M TPU/XLA vs CUDA quality gap

### 1. What I inspected, and the strongest new finding

I read all seven bounded reports, then the original sources: `telos/training/trainer_pytorch.py` (init/precision, optimizer, `train()` step loop, save/load), `telos/training/xla_utils.py`, `telos/training/core.py`, `telos/train/cli.py` (incl. `_mp_train_worker` and the resume path), `telos/diffusion/corosred_unified.py`, `telos/training/schedule.py`, `telos/training/lr_schedule.py`, `telos/models/transformer.py`, `telos/models/components.py`; plus saved `config.json`s, `hf_50m_inventory.json`, `other_50m_checkpoint_metadata.json`, `hf_50m_file_history.json`, checkpoint hashes/sizes, and git history (`a2f09a0`, `588f0b9`, `86743d0`, `774173`, `1bee3ca`, `8fc4048`, `b286a90`, `ebc7cae`).

**Strongest new finding (confirmed code behavior):** the launcher makes TPU resume *deliberately discard* AdamW state. `telos/train/cli.py:359-368`:

```python
if device != "xla" and not getattr(trainer,"is_tpu",False) and "optimizer_state_dict" in ckpt_state:
    trainer.optimizer.load_state_dict(ckpt_state["optimizer_state_dict"])   # CUDA/CPU
elif getattr(trainer,"is_tpu",False):
    print("  [Resume TPU] Optimizer moments initialized natively on TPU device ...")
```

On XLA the `elif` branch runs and the first moment (`exp_avg`) and second moment (`exp_avg_sq`) for all 49,430,017 parameters start at zero; on CUDA they are restored exactly. This is consistent with the checkpoint payloads: CUDA `50m_lightning/*.pt` are ~610 MB (weights + two fp32 moment tensors), TPU `50m_python/*.pt` are ~214 MB (weights only), so TPU payloads *cannot* carry moments. The trainer’s own `load_checkpoint()` (line 482) unconditionally reads `optimizer_state_dict`, so it would `KeyError` on a TPU file — TPU continuation must go through this `--init-checkpoint/--resume` path, which is exactly where moments are dropped.

**Why this is new relative to the reports:** the parity audit mentioned that “exact resume is unavailable” but explicitly dismissed it (“does not explain a run trained uninterrupted from scratch”). That dismissal assumes no resume occurred. The historical IDE review shows TPU sessions were *routinely* continued with weight-only init: e.g. `--init-checkpoint .../checkpoint_step_1000.pt --resume 1000` and `--resume 12000` on 8-core Kaggle TPU launches (`research/antigravity_archive_review_2026-10-18/unified/84c05c0f-….prose.txt:12547,12582,13462`). So “trained from scratch, uninterrupted” is an assumption for the TPU runs, not evidence.

**Direction/magnitude.** After each resume, `optimizer.state` is empty, so the first update is the fresh-AdamW step `≈ lr·sign(g)` at every parameter, relaxing to normal AdamW over ~1/(1−β₁) ≈ 10 steps. That is a genuine discontinuity/noise injection absent on CUDA. It plausibly explains *trajectory divergence*, but I cannot show it is *systematically worse*, and 0.4–0.7 nats by step 4000 is larger than a few 10-step transients typically produce. Counterevidence/missing: no local artifact proves the specific `50m_python`/`50m_seed` unified runs resumed, and the `50m_tpu_lrcadence_schedstep1` control would have needed resume too.

### 2. Ranked candidate mechanisms

**Candidate 1 — Resume-time optimizer-state asymmetry (confirmed code; historical exercise unproven).** Evidence: `cli.py:361-368`; 610 MB vs 214 MB payloads; archived TPU `--resume` launches. Predicted: extra large-but-few updates on TPU, plus divergence. Counterevidence: reset is divergent, not obviously inferior; resume not proven for these exact runs; magnitude may be too small.

**Candidate 2 — Runtime/autocast/kernel numerics (unmeasured).** Both paths use FP32 params + BF16 autocast (`trainer_pytorch.py:182-184,356-365`); the model math is identical (`transformer.py:88` computes `is_xla` and never uses it — dead code; attention only differs by the `mask_override` causal flag, `components.py:133-140`). So any backend difference must come from kernel/autocast dispatch, not Telos logic. Note CUDA uniquely enables TF32 (`trainer_pytorch.py:52-54`, CUDA-only) yet is *better*, so “CUDA more precise” is not the story. No matched accelerator capture exists.

**Candidate 3 — LR/objective/metric cadence (10/25 on TPU vs 1 on CUDA).** Evidence: `trainer_pytorch.py:270-271,344-346,522-524`; saved scheduler `update_cadence:10`. But `DynamicMetricTracker` compensates decay (`schedule.py:194`) to a ~100-step time constant, the audit measures only +0.058 % total LR at cadence 10, and the `lrcadence_schedstep1` control still underperformed. Weak.

### 3. Prior conclusions I can refute or strengthen from original evidence

- **Refuted (strengthened):** the in-tree rationale for commit `b286a90` and `trainer_pytorch.py:847-848` — that `xm.optimizer_step()` after `xm.reduce_gradients()` would “multiply gradients by 1/(world_size²) (1/64x).” `reduce_gradients` is a *mean* (SUM then scale 1/world_size), so applying it twice to already-synchronised gradients is **idempotent**, not 1/64. No 64× attenuation ever occurred in the “double reduction” era; that era’s real difference was only reduction *order*. The comment is still wrong in-tree and must not be used to explain the gap.
- **Refuted:** the assumption that the TPU runs were uninterrupted-from-scratch. The archived Kaggle TPU commands use `--init-checkpoint`+`--resume`, and the TPU save path omits moments — so “from scratch” must be established, not assumed.
- **Strengthened:** corpus identity is *not* a confounder. The 5B file was built so its first 2.5B tokens are identical to the 2.5B file (archive prose:21283), and both runs consumed ≈384×512×12,716 ≈ 2.5B tokens. Also, the “cadence-1” TPU control (`50m_tpu_lrcadence_schedstep1/config.json`) set only `lr_cadence` and `sched_cadence`; the metric `cadence` key is absent, so gamma-gate cadence stayed 25 (though its EMA is compensated).

### 4. Smallest next discriminating check (no new experiments in this pass)

- **Local, cheap first:** determine per run whether resume actually occurred — reconcile each TPU directory’s step files against the HF commit history/mtimes (I already found local `50m_python/checkpoint_final.pt` = `59ee1988…`, which is *not* HF’s `5e62ca9f…`, and local `step_4000/8000/12000` were written within ~3 minutes, i.e. copied) and hunt for launch logs recording `--resume` for `50m_python`/`50m_seed`. If resume is confirmed, the mechanism moves from conditional to actionable.
- **If only hardware resolves it:** on identical FP32 params + BF16 autocast, capture on both XLA and CUDA, at a step boundary (e.g. step 1 and ~1000): (a) `p.grad` immediately after reduction (XLA `xm.reduce_gradients` vs DDP-averaged), (b) the global clip coefficient from `clip_grad_norm_`, and (c) after a forced resume-with-reset vs restored-moments, the first three updates’ `exp_avg`/`exp_avg_sq` and the resulting Δweights. Compare those specific tensors, not endpoints.

### 5. Is the cause established?

**No.** I confirmed one real backend-dependent divergence in update semantics (TPU resume silently zeroes AdamW moments while CUDA restores them) and refuted a phantom 1/64 mechanism, but no local artifact proves the historical TPU runs exercised the resume path, and no matched accelerator forward/backward/reduction/clip/update capture exists. The 50 MB vs 610 MB payload sizes and the code guard make the resume asymmetry the leading concrete candidate, but the root cause of the persistent quality gap remains **unestablished**.
