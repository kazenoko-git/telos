# Full-Router Follow-Up Evaluation & Bounded 64-Step Training (2026-11-27)

All experiments ran sequentially on a single NVIDIA H100 80GB under `cpu=1, memory=8192` (2 vCPU, 8 GiB requested host RAM), `retries=0`, `max_containers=1`, `scaledown_window=2`, and `timeout=1200` using `telos-ml==1.9.1rc2`, `transformers==5.25.0.dev0` (commit `8e094f8970a60ca83052e2c199783b247b8fc13e`), and `torch==2.15.0+cu132`.

## 1. Fresh-Base Reload & Stock Parity Verification

Loading a fresh base checkpoint from `/vol/hf` (`local_files_only=True`, `dtype=torch.bfloat16`, `attn_implementation="eager"`) and patching with `patch_moe_for_window_distill(..., window_size=16, lora_rank=8, selector_rank=8, router_training="full")`:

| Model | Revision | Adapter tensors | Exact tensors | Zero-reload logits exact | Base gate frozen | Teacher == unpatched stock (`max_abs_diff`) |
| --- | --- | ---: | --- | --- | --- | --- |
| `thinkingmachines/Inkling-2-Mini` | `c9d028302cd0` | 360 | Yes | Yes | Yes | Yes (`0.0`) |
| `Qwen/Qwen3.5-35B-A3B` | `2508b2437e38` | 500 | Yes | Yes | Yes | Yes (`0.0`) |
| `google/gemma-4-26B-A4B-it` | `4d7ae4984b7d` | 320 | Yes | Yes | Yes | Yes (`0.0`) |

Re-evaluating the original 8 `rc2` held-out prompts (`max_length=128`) on the freshly reloaded models reproduced [rc2/summary.json](rc2/summary.json) `after` metrics (`ce`, `kl`, `top1_match`) identically.

## 2. Locked 64-Prompt Disjoint Held-Out Evaluation (`max_length=256`)

All 64 held-out prompts were drawn from `/vol/dolly.json` after excluding the union of all `rc1` and `rc2` train and held-out prompts. Phase 2 (`long_64step`) trained on 128 fresh disjoint prompts from `/vol/dolly_1024.json` (steps=64, batch_size=1, grad_accum=2, `max_length=256`, 6-step linear warmup + cosine LR `1.5e-4 → 1.5e-5`, checkpoints saved at steps 16, 32, 48, 64).

### Inkling-2-Mini (`18.98M` full-router trainable params; `6.23M` low-rank)

| Variant | NLL (CE) | Teacher KL | Top-1 Match |
| --- | ---: | ---: | ---: |
| Stock (token routing) | 2.1499 | 0.0000 | 100.00% |
| Untrained W=16 | 3.2176 | 1.6213 | 54.47% |
| Low-Rank W=16 (8 steps, `5e-5`) | 3.1950 | 1.5980 | 54.96% |
| Full-Router W=16 (8 steps, `5e-5`) | 3.1184 | 1.5818 | 55.24% |
| Full-Router W=16 (32 steps, `1.5e-4` cos) | **2.5786** | 1.1360 | 61.81% |
| Full-Router W=16 (64 steps, `1.5e-4` cos) | 2.6092 | **1.0851** | **62.39%** |

### Qwen3.5-35B-A3B (`50.08M` full-router trainable params; `10.35M` low-rank)

| Variant | NLL (CE) | Teacher KL | Top-1 Match |
| --- | ---: | ---: | ---: |
| Stock (token routing) | 2.4111 | 0.0000 | 100.00% |
| Untrained W=16 | 3.7736 | 1.6033 | 54.24% |
| Low-Rank W=16 (8 steps, `5e-5`) | 3.6842 | 1.5336 | 55.28% |
| Full-Router W=16 (8 steps, `5e-5`) | 3.5202 | 1.5062 | 55.77% |
| Full-Router W=16 (32 steps, `1.5e-4` cos) | 2.9137 | 1.1011 | 62.24% |
| Full-Router W=16 (64 steps, `1.5e-4` cos) | **2.7523** | **1.0485** | **63.15%** |

### Gemma-4-26B-A4B-it (`27.38M` full-router trainable params; `7.87M` low-rank)

| Variant | NLL (CE) | Teacher KL | Top-1 Match |
| --- | ---: | ---: | ---: |
| Stock (token routing) | 4.0915 | 0.0000 | 100.00% |
| Untrained W=16 | 4.8821 | 2.0282 | 59.87% |
| Low-Rank W=16 (8 steps, `5e-5`) | 4.6018 | 1.8719 | 59.81% |
| Full-Router W=16 (8 steps, `5e-5`) | 4.5351 | 1.8267 | 59.86% |
| Full-Router W=16 (32 steps, `1.5e-4` cos) | 3.2059 | 1.3478 | 62.71% |
| Full-Router W=16 (64 steps, `1.5e-4` cos) | **3.0728** | **1.2893** | **63.74%** |

## 3. Greedy Free-Generation Qualitative Summary (48 new tokens, `use_cache=False`)

- **Untrained W=16 & 8-step pilots**: Severe degeneration / immediate token repetition loops on Inkling (all 4 probes) and Qwen (3 of 4 probes inside `<think>`), and noticeable phrasing slips/repetition on Gemma.
- **64-step Full-Router**:
  - **Inkling**: Clean coherent sentences on 3 of 4 prompts (vegan reasons, PGA Championship, Aberdeen intro with minor repetition/factual drift), though kosher salt still hits a `"table table..."` loop after a fluent opening sentence.
  - **Qwen**: Recovers partial reasoning structure on kosher salt and Aberdeen (`"Granite City"`), while still entering `"1. 1. 1..."` numbering loops on vegan reasons and PGA Championship inside `<think>`.
  - **Gemma**: Fluent, accurate responses on vegan reasons and PGA Championship; minor `"salt" that is a "salt"` loop on kosher salt and `"Gran Granite"` slip on Aberdeen.

Raw generation texts for every prompt and variant are in [eval_phase1/summary.json](eval_phase1/summary.json) and [long_64step/summary.json](long_64step/summary.json).

## 4. Host RSS & GPU Memory Measurements

| Model | Phase 1 GPU alloc / res (GiB) | Phase 1 active RSS (GiB) | Phase 1 load peak RSS (GiB) | Phase 2 GPU alloc / res (GiB) | Phase 2 sampled peak RSS (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `inkling` | 50.90 / 52.34 | 3.95 → 5.56 | 25.81 | 51.55 / 52.36 | 25.69 |
| `qwen` | 67.55 / 67.58 | 3.78 → 6.22 | 70.73 | 68.15 / 68.32 | 72.20 |
| `gemma` | 50.21 / 50.35 | 3.86 → 5.65 | 51.93 | 68.15* / 68.32* | 53.56 |

\*In Phase 2, Modal reused the warm `qwen` container for `gemma` within `scaledown_window=2`, so `torch.cuda.max_memory_allocated()` and `ru_maxrss` carried over Qwen's prior peak; `sampled_peak_rss_gib` (`53.56 GiB`) was sampled per call and matches Gemma's isolated Phase 1 peak (`51.93 GiB`). `torch.cuda.reset_peak_memory_stats()` has been added to both runners for clean per-call GPU peak accounting. Active training/eval host RSS stayed between `5.0–6.2 GiB` across all models, while `from_pretrained` caused transient `25.7–72.2 GiB` host RSS spikes that released immediately after load.

## 5. Elapsed Compute, Artifacts, and Modal Credit Reconciliation

- **Step-64 manifests & checkpoints**: [long_64step/artifacts.json](long_64step/artifacts.json) records local paths, SHA-256 hashes, byte sizes, and Modal volume `/results/<name>/long_64step/step_{016,032,048,064}.safetensors`.
- **Reconciled Modal billing (`2026-11-27`)**:
  - Prior rc1/rc2 pilots (`ap-BUDfxk`, `ap-MfgkGD`, `ap-a8XJIZ`): `$0.6970` compute (+ `$0.1025` egress covered by allowance)
  - Phase 1 eval & low-rank baselines (`ap-w3LBL5`, `ap-MDr7tG`, `ap-5Y9moX`, `ap-fM4R94`): `$0.9876` compute
  - Phase 2 64-step training (`ap-GqEIxr`): `$0.8997` compute (`345.2s` Inkling + `259.6s` Qwen + `174.4s` Gemma)
  - **Total itemized app compute**: `$2.5843` (`modal billing summary` shows `$2.70`–`$2.83` credits applied in November 2026, `$0.00` billed).
  - **Remaining credit**: Starting from `$30.00` (`$29.30` verified in dashboard at `$0.70` used, expiring Nov 30, 2026), **`~$27.17–$27.30` remains** (`~$2.13` spent across all Phase 1 + Phase 2 work).
