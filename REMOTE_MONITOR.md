# Télos Live Remote Monitor & Benchmark Dashboard
> **Last Updated**: `2026-09-22 19:16:51 IST` (Epoch: `1790084811`)
> **Control Channel**: Gemini Antigravity Remote Control

## 1. System Hardware & Memory Telemetry

| Metric | Current Value | Target Threshold / Constraint |
| :--- | :--- | :--- |
| **CPU Utilization** | **`3.8%`** | Multi-core Apple Silicon |
| **Unified RAM Used** | **`10.88 GB`** / `24.00 GB` (`68.3%`) | Safe Headroom: `7.61 GB` free |
| **GPU / Metal Utilization** | **`1%`** | Apple Silicon Metal Core |
| **Context Length Lock** | **`8,192 Tokens`** | Strict Zero-OOM Ceiling |

## 2. LM Studio Engine State
```text
No models actively loaded in LM Studio.
```

## 3. Active Benchmark Processes
| PID | Script | CPU % | RAM % | Started At |
| :--- | :--- | :--- | :--- | :--- |
| `17380` | **`system_remote_monitor.py`** | `0.0%` | `0.1%` | `00:15:16` |

## 4. Multi-Model Benchmark Scorecard (7 Suites)

| Model | HumanEval | Tool-Use | Cyber | GPQA | MMLU | MATH | ARC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Apple AFM 3 Core Advanced** | `62.8%` | `73.3%` | `12.0%` | `39.4%` | `69.8%` | `52.0%` | `85.3%` |
| **IBM Granite 4.2 3B MLX** | `60.4%` | `70.0%` | `22.0%` | `28.8%` | `61.8%` | `64.7%` | `86.6%` |
| **Liquid AI LFM 2.5 8B MLX** | `45.7%` | `63.3%` | `24.0%` | `20.7%` | `73.1%` | `43.6%` | `59.6%` |
| **Microsoft Phi 4 Mini** | `26.2%` | `0.0%` | `10.0%` | `16.2%` | pending Pending | pending Pending | pending Pending |
| **Google Gemma 4 E4B** | `75.0%` | `76.7%` | `52.0%` | `23.2%` | pending Pending | pending Pending | pending Pending |

## 5. Live Terminal Logs

### `logs/run_lfm_8b_full_benches.log` (LFM 2.5 8B Execution)
```text
Saved to: logs/eval_report_lfm8b_master_summary.json
==========================================================================================
Benchmark Suite                               | Tasks  | Pass@1 (%) | 95% CI             | AST Valid
-----------------------------------------------------------------------------------------------
OpenAI HumanEval (Python Code)                |    164 |     45.73% | [38.4%, 53.0%]     |     61.6%
BFCL Tool-Use & Function Calling              |     30 |     63.33% | [46.6%, 80.0%]     |     90.0%
Cybersecurity OWASP/CWE Auditing              |     50 |     24.00% | [12.0%, 36.0%]     |    100.0%
GPQA Diamond (PhD Science — 8,096 Tokens)     |    198 |     20.71% | [15.2%, 26.3%]     |    100.0%
MMLU Science & STEM                           |    833 |     73.11% | [70.0%, 75.9%]     |    100.0%
Competition MATH (Hendrycks — 4,096 Tokens)   |    700 |     43.57% | [39.9%, 47.3%]     |    100.0%
ARC-Challenge Science Reasoning               |   1172 |     59.56% | [56.8%, 62.5%]     |    100.0%
===============================================================================================


==========================================================================================
  ALL BENCHMARKS COMPLETED FOR LIQUID AI LFM 2.5 8B A1B MLX
==========================================================================================
```

### `logs/run_post_school_master_pipeline.log` (Master Pipeline Standby)
```text
Numeric Precision:    54.3% of generated numbers were ground-truth matches
  Exact Constant Match: 57.8% of tasks preserved ALL required numbers
  Zero Numbers Emitted: 1.7% of tasks omitted numbers entirely (e.g. 'return val')
  Hallucinated Numbers: 80.6% of tasks generated invented numbers
=========================================================================================================

[done] Saved evaluation report to logs/eval_report_gemma4_e4b_gpqa_diamond_full.json


>>> Repairing truncated tasks in eval_report_gemma4_e4b_gpqa_diamond_full.json for Google Gemma 4 E4B (4B)...
  Found 141 truncated tasks out of 198 total tasks.
    [20/141] Repaired... Recovered: 10 (50.0%)
```

---
*Automated telemetry updated every 30 minutes via scheduled daemon.*
