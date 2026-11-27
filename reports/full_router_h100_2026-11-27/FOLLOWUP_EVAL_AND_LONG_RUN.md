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

## 6. CE-Matching 80-Step Run (`lora_rank=16`, `grad_accum=4`, `lr=3e-4`, `cov_weight=0.5`)

To close the remaining CE/KL gap toward Stock on `disjoint_64`, [modal_router_ce_match.py](../../scripts/modal_router_ce_match.py) increased attention LoRA rank (`8 → 16`), doubled batch accumulation (`2 → 4`, 320 unique disjoint examples), raised peak LR (`1.5e-4 → 3e-4` with 8-step warmup and cosine decay to `3e-5`), and doubled `cov_weight` (`0.25 → 0.5`). All three models passed exact `step_080.safetensors` reload verification (`reload_parameters_exact = True`).

| Model | Stock CE | 64-step CE / KL / Top-1 | 80-step (`step_040`) CE / KL / Top-1 | 80-step (`step_080`) CE / KL / Top-1 | Remaining CE gap vs Stock |
| --- | ---: | ---: | ---: | ---: | ---: |
| `inkling` | 2.1499 | 2.6092 / 1.0851 / 62.39% | 2.5280 / 1.0484 / 63.08% | **2.4484 / 0.9911 / 63.59%** | +0.2985 |
| `qwen` | 2.4111 | 2.7523 / 1.0485 / 63.15% | 2.6502 / 1.0304 / 65.16% | **2.6327 / 0.9637 / 65.89%** | +0.2216 |
| `gemma` | 4.0915 | 3.0728 / 1.2893 / 63.74% | 2.9332 / 1.2645 / 63.16% | **2.8068 / 1.1980 / 64.68%** | -1.2847 |

Artifacts and SHA-256 hashes are recorded in [ce_match_80step/artifacts.json](ce_match_80step/artifacts.json) and [ce_match_80step/summary.json](ce_match_80step/summary.json).

## 7. Out-of-Distribution 96-Token Capability Test Suite

[modal_router_test_suite.py](../../scripts/modal_router_test_suite.py) tested fresh-base reload of `step_080.safetensors` vs Stock across 8 out-of-distribution prompts (`logic_apples`, `python_palindrome`, `json_extract`, `science_photosynthesis`, `history_printing`, `math_train`, `summary_bullet`, `creative_haiku`, `max_new_tokens=96`, `use_cache=False`, 6 consecutive W=16 windows):

| Model | Stock `distinct_4gram` / EOS stops | `step_080` `distinct_4gram` / EOS stops | Elapsed (s) | Est. compute (USD) |
| --- | ---: | ---: | ---: | ---: |
| `inkling` | 0.986 / 7 of 8 | 0.694 / 5 of 8 | 230.8 | 0.277 |
| `qwen` | 0.934 / 0 of 8 | 0.528 / 1 of 8 | 234.6 | 0.281 |
| `gemma` | 0.996 / 4 of 8 | 0.869 / 5 of 8 | 110.4 | 0.132 |

- **Gemma (`step_080`)**: Solved `logic_apples` (`"9"`), emitted valid Tokyo JSON, wrote accurate photosynthesis/database-index/haiku responses, and showed loops/errors only on `python_palindrome` (code-fence loop), `history_printing` (year `100014`), and `math_train`.
- **Inkling (`step_080`)**: Answered photosynthesis and printing press accurately before trailing off, hit formatting/logic errors on JSON/palindrome/math/apples, and hit token loops on database bullets and haiku.
- **Qwen (`step_080`)**: Solved `logic_apples` (`"9"`) and started photosynthesis/math logically, but entered `1. 1. 2. 1...` numbering loops inside `Thinking Process:` on 5 of 8 prompts.

Full prompt-by-prompt generations are in [test_suite_96tok/summary.json](test_suite_96tok/summary.json).

## 8. Updated Modal Credit Reconciliation (`2026-11-27`)

- `ap-QXydNK...` (interrupted Inkling ce_match run during local DNS drop): `$0.3111` compute
- `ap-r4VEzm...` (`ce_match_80step`, all 3 models completed): `$1.4356` compute
- `ap-7j3nID...` (`test_suite_96tok`, all 3 models completed): `$0.6722` compute
- **Cumulative November 2026 summary (`modal billing summary --json`)**: `$5.24` metered (`$0.10` network egress covered by allowance), **`$5.14` credits applied**, `$0.00` billed — leaving **`~$24.86`** of the `$30.00` monthly credit (expiring Nov 30, 2026).

## 9. Stage-2 Mixed-Domain Training (`step 80 → step 500`)

[modal_router_stage2.py](../../scripts/modal_router_stage2.py) resumed from `ce_match_80step/step_080.safetensors` with AdamW optimizer-state persistence (`optimizer_XXX.pt`) across bounded chunks (`timeout=1200`, `cpu=1, memory=8192`) on `/vol/stage2_mixed.json` (3,000 non-overlapping rows: 1,400 unseen Dolly + 600 Python code + 600 GSM8K math + 600 No Robots; `grad_accum=4`, `lora_rank=16`, `lr=3e-5 → 2e-4` over 12 warmup steps then cosine decay to `2e-5`).

| Model | Stock CE | Step `80` CE / KL / Top-1 | Intermediate (`step 248` / `220` / `360`) | **Step `500` CE / KL / Top-1** | Remaining CE gap vs Stock |
| --- | ---: | ---: | --- | ---: | ---: |
| `inkling` | 2.1499 | 2.4484 / 0.9911 / 63.59% | `step 248`: 2.3784 / 0.8958 / 66.35% · `step 416`: 2.3422 / 0.8372 / 67.67% | **2.3178 / 0.8359 / 68.24%** | +0.1679 |
| `qwen` | 2.4111 | 2.6327 / 0.9637 / 65.89% | `step 220`: 2.5712 / 0.9313 / 65.81% · `step 360`: 2.5704 / 0.8857 / 67.65% | **2.5400 / 0.8608 / 67.85%** | +0.1289 |
| `gemma` | 4.0915 | 2.8068 / 1.1980 / 64.68% | `step 220`: 2.7228 / 1.0657 / 67.03% · `step 360`: 2.7028 / 0.9969 / 67.25% | **2.6818 / 0.9857 / 67.76%** | -1.4097 |

All three models passed exact `step_500.safetensors` parameter reload verification (`reload_parameters_exact = True`). Local `step_500.safetensors` SHA-256 hashes and Modal checkpoint paths are in [stage2_500step/artifacts.json](stage2_500step/artifacts.json) and [stage2_500step/summary.json](stage2_500step/summary.json).

## 10. Step-500 Out-of-Distribution 96-Token Test Suite

Re-running [modal_router_test_suite.py](../../scripts/modal_router_test_suite.py) on fresh base models with `step_500.safetensors`:

| Model | Stock `distinct_4gram` / EOS | `step_080` `distinct_4gram` / EOS | **`step_500` `distinct_4gram` / EOS** | Elapsed (s) / Est. USD |
| --- | ---: | ---: | ---: | ---: |
| `inkling` | 0.986 / 7 of 8 | 0.694 / 5 of 8 | **0.926 / 7 of 8** | 302.5s / \$0.363 |
| `qwen` | 0.934 / 0 of 8 | 0.528 / 1 of 8 | **0.730 / 0 of 8** | 311.8s / \$0.374 |
| `gemma` | 0.996 / 4 of 8 | 0.869 / 5 of 8 | **0.883 / 5 of 8** | 131.4s / \$0.158 |

- **Inkling (`step_500`)**: Solved `math_train` (`210 km`, stopped cleanly on EOS), `history_printing` (`1450`), `science_photosynthesis`, `summary_bullet`, and `creative_haiku` without loops; minor errors remained on `logic_apples` (`8`), `python_palindrome`, and `json_extract` (`""city"`, `"Japanese"`).
- **Qwen (`step_500`)**: Eliminated loops on 6 of 8 prompts (solving `logic_apples`=`9`, `history_printing`=`1450`, `science_photosynthesis`, `python_palindrome` setup, `math_train` fraction setup with `230 km` arithmetic slip), while `json_extract` and `summary_bullet` still hit `1. 1. 1...` numbering loops inside `Thinking Process:`.
- **Gemma (`step_500`)**: Exact `9` on `logic_apples`, valid Tokyo JSON, clean photosynthesis, DB bullets, haiku, and `2 + 1/3` hour fraction conversion; still hit a `-one-one` loop on `python_palindrome` and `1000`/`1040` on `history_printing`.

Raw outputs are recorded in [test_suite_step500/summary.json](test_suite_step500/summary.json).

## 11. Final Reconciled Modal Billing (`2026-11-27`)

- `ap-qVuRS4...` (Stage-2 mixed dataset CPU prep): `$0.0003`
- `ap-KX0pTU...` (Stage-2 `step 80 → 500` across all 11 H100 chunks): `$5.7893` compute
- `ap-HZJD3z...` (`test_suite_step500`, all 3 models): `$0.8624` compute
- **Cumulative November 2026 (`modal billing summary --json`)**: `$12.61` metered (`$0.44` free storage allowance, `$0.10` egress allowance), **`$12.07` credits applied**, `$0.00` billed — leaving **`~$17.93`** of the `$30.00` monthly credit (expiring Nov 30, 2026).

## 12. Window-8 + Resident-MLP LoRA Diagnostic (`w8_diag`)

Commit `5292de96` added `adapt_resident_mlp=True` to `patch_moe_for_window_distill` (wrapping Qwen3.5 `shared_expert` and Gemma4 dense parallel `mlp` `gate_proj`/`up_proj`/`down_proj` with LoRA while keeping routed experts frozen) and `mlp.shared_expert` resolution in `window_routing.py`. [modal_router_w8_and_rollouts.py](../../scripts/modal_router_w8_and_rollouts.py) tested `W=8` + `adapt_resident_mlp=True` over 80 mixed-domain steps and generated initial stock on-policy rollouts:

| Model | Untrained `W=8` CE / KL / Top-1 | `W=8` `step_080` CE / KL / Top-1 | `W=8` `step_080` 96-tok `d4` / EOS | `W=16` `step_500` `d4` / EOS |
| --- | ---: | ---: | ---: | ---: |
| `inkling` | 3.0533 / 1.4378 / 57.25% | 2.4082 / 0.9041 / 65.83% | **0.947** / 7 of 8 | 0.926 / 7 of 8 |
| `qwen` | 3.5298 / 1.3254 / 58.24% | 2.5732 / **0.8400 / 68.10%** | **0.868** / 1 of 8 | 0.730 / 0 of 8 |
| `gemma` | 4.8257 / 1.7881 / 61.51% | 2.7612 / 1.0123 / 67.95% | **0.962** / 5 of 8 | 0.883 / 5 of 8 |

In 80 steps, `W=8` + `adapt_resident_mlp=True` outperformed 500 steps of `W=16` on free-generation coherence across all three models and beat `W=16` step-500 KL/top-1 on Qwen (`0.8400`, `68.10%`).

## 13. W=8 On-Policy + Mixed Distillation (`step 80 → step 320`)

[modal_router_w8_onpolicy.py](../../scripts/modal_router_w8_onpolicy.py) topped up stock on-policy rollouts (`34` Inkling, `85` Qwen, `97` Gemma) and resumed `W=8` + `adapt_resident_mlp=True` from `w8_80step/step_080` through **step 320** (50% on-policy rollouts with `ce_weight=0.0`, 50% mixed domain prompts with `ce_weight=0.05`, `kl_weight=1.0`, `cov_weight=0.5`, `lr=3e-5 → 2e-4 → 2e-5`):

| Model | Stock CE / `d4` | `step_200` CE / KL / Top-1 | `step_320` CE / KL / Top-1 | `step_320` 96-tok `d4` / EOS |
| --- | ---: | ---: | ---: | ---: |
| `inkling` | 2.1499 / 0.986 | 2.4267 / 0.8193 / 68.02% | **2.4167 / 0.8084 / 68.09%** | **0.946** / 7 of 8 |
| `qwen` | 2.4111 / 0.934 | 2.5746 / 0.8002 / 68.08% | **2.5570 / 0.7642 / 70.29%** | **0.933** / 0 of 8 |
| `gemma` | 4.0915 / 0.996 | — | **2.8909 / 0.9097 / 70.18%** | **0.992** / 5 of 8 |

- **Gemma (`step_320`)**: Zero loops across all 8 out-of-distribution probes (`distinct_4gram = 0.992` vs `0.996` Stock), solving `logic_apples=9`, `is_palindrome`, Tokyo JSON, photosynthesis, `1440` printing press, `20/60 = 0.333` hour math, DB bullets, and haiku cleanly.
- **Qwen (`step_320`)**: Zero `1. 1. 1...` numbering loops across all 8 probes (`distinct_4gram = 0.933` vs `0.934` Stock), matching Stock's `Thinking Process:` structure across every prompt.
- **Inkling (`step_320`)**: Zero repetition loops (`distinct_4gram = 0.946`, `7/8` EOS), with minor arithmetic/syntax errors (`8 sheep`, `630 km`, `.lowercase`) remaining where Inkling lacks a resident dense/shared MLP.

All `step_320.safetensors` adapters passed exact reload verification (`reload_parameters_exact = True`) and are indexed in [w8_onpolicy_320/artifacts.json](w8_onpolicy_320/artifacts.json).

## 14. Cumulative Modal Credit Reconciliation (`2026-11-27`)

- `ap-7nbJRk...` (`w8_diag` + initial rollouts): `$2.5720` compute
- `ap-D0fmia...` (`w8_onpolicy_320` + rollout top-up + final 96-token test suite): `$4.2792` compute
- **Cumulative November 2026 (`modal billing summary --json`)**: `$19.34` metered (`$0.44` free storage allowance, `$0.10` egress allowance), **`$18.80` credits applied**, `$0.00` billed — leaving **`~$11.20`** of the `$30.00` monthly credit (expiring Nov 30, 2026).

## 15. Option 1: 12-Task Verifiable Capability Benchmark (`max_new_tokens=384`)

To test end-to-end task preservation (exact integers, GSM8K-style word problems, syllogism deduction, sandbox-executed Python functions `is_palindrome` / `merge_intervals` / `two_sum`, strict JSON schema, SQL, science, and history), all three models were evaluated at `max_new_tokens=384` against fresh Stock baselines (`use_cache=True` for Stock, exact sequential `use_cache=False` for `W=8` adapters; apples-to-apples conclusion grading applied to `logic_syllogism`):

| Model & Checkpoint | Passed | EOS Stops | Mean `distinct_4gram` | Mean Tokens |
| --- | ---: | ---: | ---: | ---: |
| `inkling` Stock | **12/12** | 12/12 | 0.982 | 54.5 |
| `inkling` `W=8 step_320` | 7/12 | 12/12 | 0.968 | 58.4 |
| `inkling` `W=8 step_600` | 7/12 | 12/12 | **0.973** | 45.6 |
| `gemma` Stock | **12/12** | 12/12 | 0.963 | 119.5 |
| `gemma` **`W=8 step_320`** | **10/12** | 12/12 | 0.950 | 106.0 |
| `gemma` `W=8 step_500` | 7/12 | 12/12 | 0.945 | 106.8 |
| `qwen` Stock (default thinking) | 6/12 | 1/12 | 0.840 | 379.5 |
| `qwen` `W=8 step_320` (default thinking) | 6/12 | 4/12 | 0.765 | 360.7 |
| `qwen` `W=8 step_500` (default thinking) | 6/12 | 2/12 | 0.798 | 372.3 |
| `qwen` Stock (`enable_thinking=False`) | **11/12** | 12/12 | 0.957 | 112.5 |
| `qwen` `W=8 step_500` (`enable_thinking=False`) | 6/12 | 12/12 | **0.970** | 88.7 |

Raw prompt-by-prompt outputs and grading details are in [option12/bench12_baseline.json](option12/bench12_baseline.json) and [option12/bench12_final.json](option12/bench12_final.json).

## 16. Option 2: Stage-3 Push (`Inkling → step_600`, `Qwen/Gemma → step_500`)

Resuming from `step_320.safetensors` with expanded Inkling stock on-policy rollouts (`34 → 54`, `$0.466`), strictly unseen mixed-domain rows (`1480:2400`), and cosine LR (`2e-5 → 1.4e-4 → 1.5e-5`, `W=8`, `adapt_resident_mlp=True`), all three models improved held-out `disjoint_64` NLL/KL/Top-1 beyond `step_320` and passed exact reload verification (`reload_parameters_exact = True`):

| Model & Stage | Stock CE | `step_320` CE / KL / Top-1 | Stage-3 (`step_600` / `step_500`) CE / KL / Top-1 | Stage-3 Artifact & SHA-256 |
| --- | ---: | ---: | ---: | --- |
| `inkling` (`step_600`) | 2.1499 | 2.4167 / 0.8084 / 68.09% | **2.3227 / 0.7326 / 70.18%** | `50ea939a2869dcd382d48093ec1ae2d3ba0b4fdbc5af5b33cfbd36ef65dab94c` |
| `qwen` (`step_500`) | 2.4111 | 2.5570 / 0.7642 / 70.29% | **2.5017 / 0.7346 / 71.00%** | `b87d3bc624d47acac6b17be4818bd02889106809b745ea655bd99a7ba1303bf4` |
| `gemma` (`step_500`) | 4.0915 | 2.8909 / 0.9097 / 70.18% | **2.8266 / 0.8895 / 70.49%** | `9fd746180354557875a16e518d21897b15f528c0d9f49eba006cfa61994bed65` |

**Key takeaway**: While coherence loops are completely resolved across all three models (`distinct_4gram` `0.945–0.973`, `12/12` EOS on Inkling/Gemma/Qwen-nothink) and Stage 3 improved held-out Dolly Top-1 above `70%` on every model, **`step_500` regressed Gemma on exact `bench12` (`10/12 → 7/12`)** and did not lift Inkling past `7/12`. At `~0.73–0.89` per-token KL (`~70–71%` Top-1), multi-step coding/math precision remains vulnerable to isolated token typos (e.g., `.isalalpha()`, `GROUPBy`, arithmetic slips), making **`gemma w8_step320` (`10/12`)** stronger than `gemma step_500` on end-to-end tasks. Artifacts are indexed in [option12/artifacts.json](option12/artifacts.json) and saved under `models/full_router_w8_stage3_h100_2026-11-27/`.

## 17. Final Modal Billing Reconciliation (`2026-11-27`)

- Cumulative November 2026 (`modal billing summary`): **`$27.55` metered** (`$0.44` free storage, `$0.10` network egress allowance, **`$27.01` credits applied**, **`$0.00` billed**), leaving **`~$2.99`** of the `$30.00` monthly credit (expiring Nov 30, 2026). All ephemeral apps stopped.
