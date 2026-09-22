# IBM Granite 4.2 3B Benchmark Report

**Model**: IBM Granite 4.2 3B Instruct  
**Runtime**: MLX LM / LM Studio REST API  
**Hardware**: Apple Silicon

---

## Performance Summary

| Suite | Metric | Pass Rate |
| :--- | :--- | :---: |
| **HumanEval** | Pass@1 (164 tasks) | **60.37%** |
| **BFCL Tool-Use** | Pass@1 (30 tasks) | **70.00%** |
| **Competition MATH** | Pass@1 (140 tasks) | **64.71%** |
| **ARC-Challenge** | Pass@1 (150 tasks) | **86.60%** |
| **MMLU Science** | Pass@1 (119 tasks) | **61.82%** |
| **GPQA Diamond** | Pass@1 (198 tasks) | **28.79%** |
| **Cybersecurity** | Pass@1 (50 tasks) | **22.00%** |

---

## Highlights

- **Math & Logic Dominance**: Leads the 3B parameter class in Competition MATH (**64.71%**) and ARC-Challenge (**86.60%**).
- **Code Execution**: Achieved **60.37%** Pass@1 on OpenAI HumanEval.
