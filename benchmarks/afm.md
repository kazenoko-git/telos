# Apple Foundation Models (AFM) Benchmark Report

**Model**: Apple AFM-3 Core Advanced & AFM-3 Core  
**Runtime**: Native Swift IPC (`telos.afm` over `FoundationModels.framework`)  
**Hardware**: Apple Silicon M-Series (M5 / M4 / M3)

---

## Performance Summary

| Suite | Metric | Pass Rate |
| :--- | :--- | :---: |
| **HumanEval** | Pass@1 (164 tasks) | **62.80%** |
| **BFCL Tool-Use** | Pass@1 (30 tasks) | **73.33%** |
| **GPQA Diamond** | Pass@1 (198 tasks) | **39.39%** |
| **MMLU Science** | Pass@1 (119 tasks) | **69.75%** |
| **Competition MATH** | Pass@1 (140 tasks) | **52.00%** |
| **ARC-Challenge** | Pass@1 (150 tasks) | **85.32%** |
| **Cybersecurity** | Pass@1 (50 tasks) | **12.00%** |

---

## Integration Architecture

Télos accesses AFM-3 directly via a lightweight compiled Swift binary (`telos/afm/_swift/afm_bridge.swift`) communicating over bidirectional JSON IPC. The bridge requires no cloud servers, API keys, or manual model downloads.

```bash
# Check availability status
telos afm status

# Generate completion
telos afm generate "Explain rotary position embeddings in two sentences."
```
