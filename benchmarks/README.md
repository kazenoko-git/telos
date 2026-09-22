# Télos Institutional Benchmark Suite

This directory contains comprehensive evaluation results, comparative scorecards, and methodology breakdowns across frontier edge models and Télos architectures.

---

## 📊 Benchmark Reports

| Report | Description & Scope |
| :--- | :--- |
| **[AFM vs LFM vs IBM Comparison](afm_vs_lfm_vs_ibm.md)** | Head-to-head institutional evaluation comparing **Apple AFM-3 Core Advanced**, **IBM Granite 4.2 3B (MLX)**, and **LiquidAI LFM 8B A1B** across 7 benchmark suites. |
| **[Apple Foundation Models (AFM)](afm.md)** | Native Swift IPC evaluation report for on-device Apple Intelligence models (AFM-3 Core & Core Advanced). |
| **[IBM Granite 4.2 3B](ibm_granite.md)** | Quantized Apple Silicon (MLX) & LM Studio evaluation report for IBM Granite 4.2 3B. |
| **[LiquidAI LFM 8B A1B](liquid_lfm.md)** | Hybrid architecture performance report for LiquidAI LFM 8B A1B. |

---

## 🏆 Master Leaderboard Summary (7-Suite Completion)

| Model | HumanEval (Pass@1) | BFCL Tool-Use | Cyber Auditing | GPQA Diamond | MMLU Science | Competition MATH | ARC Challenge | 7-Suite Average |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Apple AFM-3 Core Advanced** | **62.80%** | **73.33%** | 12.00% | **39.39%** | 69.75% | 52.00% | 85.32% | **56.37%** |
| **IBM Granite 4.2 3B (MLX)** | 60.37% | 70.00% | 22.00% | 28.79% | 61.82% | **64.71%** | **86.60%** | 56.33% |
| **LiquidAI LFM 8B A1B** | 45.73% | 63.33% | **24.00%** | 20.71% | **73.11%** | 43.57% | 59.56% | 47.14% |

---

## 🌐 Interactive Comparison Dashboard

Generate the self-contained HTML dashboard with Chart.js charts and radar plots:

```bash
python scripts/generate_benchmark_dashboard.py
```

View live visualizations online at **[telos.research.wingit.tech](https://telos.research.wingit.tech)**.
