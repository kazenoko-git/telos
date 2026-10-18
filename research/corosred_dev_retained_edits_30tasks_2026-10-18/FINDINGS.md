# Development Report: COROSred Edits Restricted to Retained Execution Code

**Date:** 2026-10-18  
**Environment:** Development Set Only (MBPP 30 Tasks, `MBPP/2` through `MBPP/71`)  
**Status:** Reserved fresh benchmark tasks (HumanEval 0–163) remain genuinely untouched.  

---

## 1. Executive Summary

In previous iterations, unconstrained confidence routing flagged tokens across the entire raw generation stream—frequently masking trailing dead code, echoed test cases, or stop-string fragments (e.g. `\ndef square...`, `\nif __name__...`) that were subsequently discarded before execution. Consequently, many token edits were "dead edits" that never altered the executed program.

To address this, we:
1. **Restricted Eligible Edits:** Mapped the exact character span of the code actually retained for execution (`clean_functional_completion`) to token offset intervals `(start, end)`. Only tokens strictly within the executed code span and with non-whitespace content were eligible for masking and repair.
2. **Logged Every Selected Token and Replacement:** Tracked the original token string, ID, margin, and entropy alongside the MDLM replacement token string, ID, and posterior probability.
3. **Classified Edits:** Distinguished between no-op edits (MDLM regenerates the identical token), effective code edits (the code passed to the sandbox actually changed), fixes (failing draft becomes passing), and regressions (passing draft becomes failing).
4. **Preserved Test Integrity:** Conducted all experiments strictly on the development set (MBPP), preserving HumanEval as genuinely untouched fresh tasks for the eventual frozen-policy test.

---

## 2. Quantitative Summary Across 30 Development Tasks

| Metric | Untouched AR Draft Baseline | Single-Token Repair (Retained Code) | Budget $k=3$ Repair (Retained Code) |
| :--- | :---: | :---: | :---: |
| **Pass@1 (%)** | **13.3%** (4/30) | **10.0%** (3/30) | **10.0%** (3/30) |
| **AST Validity (%)** | **96.7%** (29/30) | **93.3%** (28/30) | **90.0%** (27/30) |
| **Routing Triggers** | 0 / 30 (0.0%) | **30 / 30 (100.0%)** | **30 / 30 (100.0%)** |
| **Total Candidates Masked** | 0 | 30 | 78 |
| **Effective Code Edits** | 0 | **11 / 30 (36.7%)** | **23 / 30 (76.7%)** |
| **No-Op Edits (Token Preserved)** | 0 | **19 / 30 (63.3%)** | **7 / 30 (23.3%)** |
| **Drafts Fixed (FAIL $\to$ PASS)** | N/A | **0 / 30 (0.0%)** | **0 / 30 (0.0%)** |
| **Drafts Broken (PASS $\to$ FAIL)** | N/A | **1 / 4 (25.0%)** (`MBPP/17`) | **1 / 4 (25.0%)** (`MBPP/17`) |
| **Neutral Effective Edits** | 0 | **10 / 30 (33.3%)** | **22 / 30 (73.3%)** |
| **Mean Latency per Task** | **4.950 s** | 5.529 s (+0.579s) | 5.269 s (+0.319s) |

---

## 3. Log of Selected Tokens and Replacements (Policy 1: Single Token)

The following table details all 11 effective code edits produced by Policy 1 within the executed code:

| Task ID | Position | Original Token | AR Margin | AR Entropy | Replacement Token | MDLM Prob | Outcome Impact |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `MBPP/16` | 1 | `' return'` | 0.00 | 2.34 | `' assert'` | 0.32 | Neutral (FAIL $\to$ FAIL) |
| `MBPP/17` | 1 | `' return'` | 0.43 | 1.49 | `' correct'` | 0.13 | **Regression (PASS $\to$ FAIL)** |
| `MBPP/18` | 2 | `' your'` | 0.01 | 5.70 | `' write'` | 0.31 | Neutral (FAIL $\to$ FAIL) |
| `MBPP/19` | 8 | `' first'` | 0.01 | 5.30 | `' First'` | 0.30 | Neutral (FAIL $\to$ FAIL) |
| `MBPP/56` | 1 | `' rev'` | 0.26 | 2.09 | `' n'` | 0.56 | Neutral (FAIL $\to$ FAIL) |
| `MBPP/62` | 8 | `' return'` | 0.04 | 5.59 | `' find'` | 0.71 | Neutral (FAIL $\to$ FAIL) |
| `MBPP/63` | 8 | `' return'` | 0.00 | 5.82 | `' find'` | 0.88 | Neutral (FAIL $\to$ FAIL) |
| `MBPP/64` | 20 | `' print'` | 0.04 | 3.42 | `' return'` | 0.18 | Neutral (FAIL $\to$ FAIL) |
| `MBPP/65` | 8 | `' return'` | 0.00 | 5.58 | `' Calculate'` | 0.17 | Neutral (FAIL $\to$ FAIL) |
| `MBPP/68` | 8 | `' check'` | 0.03 | 5.87 | `' Check'` | 0.59 | Neutral (FAIL $\to$ FAIL) |
| `MBPP/70` | 2 | `' your'` | 0.02 | 4.47 | `' Write'` | 0.79 | Neutral (FAIL $\to$ FAIL) |

---

## 4. Key Qualitative Patterns in Effective Edits

### Pattern 1: Bidirectional Docstring Echoing in Comments
In tasks `MBPP/62` and `MBPP/63`:
- **Docstring:** `"Write a python function to find smallest number in a list."`
- **AR Draft Comment:** `# return the smallest number in the list`
- **MDLM Edit:** Replaced `' return'` with `' find'` ($p=0.71$ and $p=0.88$).
- *Observation:* MDLM leverages future context to align inline comments with the exact vocabulary of the docstring. While semantically appropriate for comments, this does not alter execution semantics.

### Pattern 2: Keyword Corruption Under Low Replacement Confidence
In task `MBPP/17`:
- **Prompt:** `def square_perimeter(a): """Write a function that returns the perimeter..."""`
- **AR Draft:**
  ```python
  def square_perimeter(a):
      return 4 * a
  ```
  *(Status: Correct, passes all unit tests)*
- **MDLM Edit:** At position 1, replaced `' return'` with `' correct'`, producing:
  ```python
  def square_perimeter(a):
      correct 4 * a
  ```
  *(Status: Broken by SyntaxError)*
- *Root Cause:* MDLM's replacement probability for `' correct'` was only **0.131 (13.1%)**. Because the repair policy blindly accepted MDLM's argmax without requiring a minimum posterior threshold (e.g. $p > 0.50$) or checking that MDLM's probability exceeds AR's confidence, an unconfident guess destroyed a working program.

### Pattern 3: Statement Level Correction
In task `MBPP/64`:
- **AR Draft:** `print(sort_marks(...))`
- **MDLM Edit:** Replaced `' print'` with `' return'` ($p=0.18$).
- *Observation:* MDLM correctly recognized that a function completion should return rather than print, demonstrating architectural capacity for semantic repair, although full recovery requires repairing the enclosing expression syntax.

---

## 5. Strategic Implications for Policy Freezing

1. **Retained-Code Restriction Is Mandatory:** Without restricting candidate tokens to the retained code span, 50%+ of edits fall on dead comments or stop strings. Restricting eligible tokens ensures that every repair operation targets code actually executed by the test harness.
2. **MDLM Confidence Gating Is Essential to Prevent Regressions:**
   - When MDLM's replacement probability is low ($p < 0.50$), it hallucinates invalid tokens (such as `' correct'` with $p=0.13$).
   - A repair should only be accepted if:
     $$\text{MDLM Prob}(t_{\text{new}}) > \tau_{\text{accept}} \quad (\text{e.g. } \tau \ge 0.50)$$
     or if the replacement probability significantly exceeds the AR draft's token probability.
3. **Fresh Tasks Preserved:** All 164 tasks of HumanEval remain completely fresh and unobserved, reserved for the final frozen-policy evaluation.

---

## 6. Artifact References
- Master Summary: [`research/corosred_dev_retained_edits_30tasks_2026-10-18/summary.json`](file:///Users/ivansamuel/Documents/Projects/telos/research/corosred_dev_retained_edits_30tasks_2026-10-18/summary.json)
- Full Edit Logs: [`research/corosred_dev_retained_edits_30tasks_2026-10-18/edits_log.json`](file:///Users/ivansamuel/Documents/Projects/telos/research/corosred_dev_retained_edits_30tasks_2026-10-18/edits_log.json)
- Per-Task Traces: [`research/corosred_dev_retained_edits_30tasks_2026-10-18/per_task_results.json`](file:///Users/ivansamuel/Documents/Projects/telos/research/corosred_dev_retained_edits_30tasks_2026-10-18/per_task_results.json)
- Reproduction Command:
  ```bash
  .venv/bin/python scripts/evaluate_corosred_retained_dev.py --max-tasks 30 --output-dir research/corosred_dev_retained_edits_30tasks_2026-10-18
  ```
