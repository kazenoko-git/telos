# Institutional Benchmark Report: Apple AFM-3 vs. IBM Granite 4.2 3B vs. LiquidAI LFM 8B

**Author**: Ivan Samuel  
**Affiliation**: Wing It Research  
**Website**: [telos.research.wingit.tech](https://telos.research.wingit.tech)

---

## Executive Summary

We conducted an institutional-grade 7-suite comparative evaluation comparing three premier edge and open-weights models:
1. **Apple AFM-3 Core Advanced** (Native Swift IPC `FoundationModels.framework` on Apple Silicon)
2. **IBM Granite 4.2 3B** (MLX 4-bit / LM Studio execution)
3. **LiquidAI LFM 8B A1B** (Hybrid architecture execution)

---

## Comprehensive Scorecard

| Suite | Tasks | Apple AFM-3 Core Adv. | IBM Granite 4.2 3B | LiquidAI LFM 8B A1B | Metric |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **OpenAI HumanEval** | 164 | **62.80%** | 60.37% | 45.73% | Pass@1 (AST Valid Code) |
| **BFCL Tool-Use** | 30 | **73.33%** | 70.00% | 63.33% | Pass@1 (JSON Schema Match) |
| **Cybersecurity Auditing** | 50 | 12.00% | 22.00% | **24.00%** | Pass@1 (CWE Remediation) |
| **GPQA Diamond** | 198 | **39.39%** | 28.79% | 20.71% | Pass@1 (Multiple Choice) |
| **MMLU Science** | 119 | 69.75% | 61.82% | **73.11%** | Pass@1 (Multiple Choice) |
| **Competition MATH** | 140 | 52.00% | **64.71%** | 43.57% | Pass@1 (Boxed Numerical/LaTeX) |
| **ARC-Challenge** | 150 | 85.32% | **86.60%** | 59.56% | Pass@1 (Reasoning Choice) |
| **Overall Average** | **841** | **56.37%** | **56.33%** | **47.14%** | Macro Average |

---

## Key Domain Insights

### 1. Code Generation (HumanEval) & Tool-Use (BFCL)
- **Apple AFM-3 Core Advanced** achieved the highest coding efficiency (**62.80% Pass@1**), outperforming Granite 4.2 3B (**60.37%**) and LFM 8B (**45.73%**).
- On function calling / tool-use, AFM-3 achieved **73.33%**, tightly followed by Granite 4.2 at **70.00%**.

### 2. Mathematics & Logical Reasoning (Competition MATH & ARC)
- **IBM Granite 4.2 3B** led in Competition MATH with **64.71%** Pass@1, demonstrating strong chain-of-thought mathematical reasoning capabilities.
- Granite 4.2 3B and AFM-3 Core Advanced performed neck-and-neck on ARC-Challenge (**86.60%** vs **85.32%**).

### 3. Scientific Knowledge & Graduate Reasoning (GPQA Diamond & MMLU)
- **AFM-3 Core Advanced** dominated GPQA Diamond with **39.39%** accuracy, outperforming Granite 4.2 3B (**28.79%**) and LFM 8B (**20.71%**).
- **LiquidAI LFM 8B** achieved its highest domain performance on MMLU Science with **73.11%**.

### 4. Cybersecurity OWASP/CWE Auditing
- **LiquidAI LFM 8B** (**24.00%**) and **IBM Granite 4.2 3B** (**22.00%**) outperformed AFM-3 Core Advanced (**12.00%**) on secure code remediation, where strict prompt safety filters often caused refusal responses in AFM-3.

---

## Evaluation Reproducibility Commands

```bash
# Run Apple AFM-3 Core Advanced evaluation
telos eval --model afm-3-core-advanced --backend swift --type all

# Run IBM Granite 4.2 3B evaluation
telos eval --model ibm-granite-4.2-3b --backend mlx --type all

# Run LiquidAI LFM 8B evaluation
telos eval --model lfm-8b-a1b --type all
```
