# Empirical Investigation: End-to-End Evaluation of Qwen3-0.6B AR, MDLM, and COROSred

**Date:** 2026-10-18  
**Hardware:** Apple M5 Pro (16-core, 24 GB Unified Memory, MPS backend)  
**Evaluator:** Télos Functional Benchmark Harness (`telos.eval.runner.evaluate_functional`)  
**Suite:** HumanEval (40 Tasks, Sandboxed Subprocess Execution, Timeout: 3.0s)  
**Status:** Completed  

---

## 1. Executive Summary

We evaluated whether existing matched autoregressive (AR) and masked-diffusion language model (MDLM) checkpoints can provide a viable, practical starting point for **COROSred** (Confidence-Routed Selective Re-Diffusion: AR draft $\to$ reference-free uncertainty routing $\to$ bidirectional repair) without additional training.

The three evaluated systems were:
1. **Native Qwen3-0.6B AR Baseline:** Greedy causal decoding with KV caching, stopping on EOS (`<|im_end|>`, `<|endoftext|>`) or function/class stop strings.
2. **Native Qwen3-0.6B MDLM Baseline:** Iterative block diffusion (`block_size=32`, `steps=32`, low-confidence remasking) on the identical prefix prompts.
3. **Two-Model COROSred:** AR draft $\to$ reference-free uncertainty detection (`margin < 0.5` or `entropy > 1.0`) $\to$ bidirectional masked repair via MDLM on surrounding generated context.

### Headline Results Across 40 Functional Tasks

| Metric | Native AR Baseline | Native MDLM Baseline | Two-Model COROSred |
| :--- | :---: | :---: | :---: |
| **Pass@1 (%)** | **42.5%** (17/40) | **0.0%** (0/40) | **20.0%** (8/40) |
| **AST Validity (%)** | **100.0%** (40/40) | **7.5%** (3/40) | **87.5%** (35/40) |
| **Mean Latency / Task** | 5.534 s | 4.694 s | 5.524 s |
| **Mean Generation Latency** | 3.671 s (AR draft) | 4.694 s (diffusion) | 3.671s (AR) + 0.285s (repair) |
| **Routing Frequency** | N/A | N/A | **100.0%** (40/40 tasks) |
| **MDLM Forward Passes** | 0 | 1,280 (32 steps $\times$ 40) | 40 (1 per routed task) |
| **Drafts Fixed (Incorrect $\to$ Correct)** | N/A | N/A | **0 / 40** (0.0%) |
| **Drafts Broken (Correct $\to$ Incorrect)** | N/A | N/A | **9 / 17** (52.9% of working drafts) |

---

## 2. Experimental Setup & Pinned Artifacts

### Pinned Checkpoint Revisions
- **AR Model:** `Qwen/Qwen3-0.6B`  
  - Repository Revision: `c1899de289a04d12100db370d81485cdf75e47ca`  
  - Architecture: 28 layers, hidden dim 1024, 16 query heads, 8 KV heads (GQA), QK-RMSNorm, RoPE theta 1,000,000.  
  - Weights: `~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/model.safetensors`
- **MDLM Model:** `dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1`  
  - Repository Revision: `c8d24a3f4adaeef46881b450e1bf7d1005203bd7`  
  - Architecture: Identical backbone to Qwen3-0.6B, converted with bidirectional attention and `<|mask|>` (token ID 151669).  
  - Weights: `~/.cache/huggingface/hub/models--dllm-hub--Qwen3-0.6B-diffusion-mdlm-v0.1/snapshots/c8d24a3f4adaeef46881b450e1bf7d1005203bd7/model.safetensors`
- **Tokenizer:** `c1899de289a04d12100db370d81485cdf75e47ca/tokenizer.json` (vocab size 151,643; mask token 151,669).

### Predeclared Decoding & Routing Parameters
- `temperature`: 0.0 (greedy decoding)
- `max_new_tokens`: clamped to 128 (preventing runaway decoding on open-ended function prompts)
- `stop_strings`: `["\nclass ", "\ndef ", "\nif __name__"]`
- `mdlm_block_size`: 32 tokens
- `mdlm_steps`: 32 steps (8 steps per block)
- `corosred_margin_threshold`: 0.5 (top-1 prob minus top-2 prob)
- `corosred_entropy_threshold`: 1.0 nats

---

## 3. Detailed Results & Execution Breakdown

### Outcome Categories (40 Tasks)

| Outcome Category | Native AR | Native MDLM | COROSred |
| :--- | :---: | :---: | :---: |
| **PASSED** | 17 | 0 | 8 |
| **FAILED_ASSERTION** | 18 | 1 | 19 |
| **SYNTAX_ERROR** | 0 | 37 | 5 |
| **RUNTIME_EXCEPTION** | 4 | 2 | 8 |
| **TIMEOUT** | 1 | 0 | 0 |
| **MEMORY_EXCEEDED** | 0 | 0 | 0 |
| **Total** | 40 | 40 | 40 |

---

## 4. Why Native MDLM Collapses from Scratch on Code (0% Pass@1, 7.5% AST)

When generating free-form Python function completions from raw docstrings, the native MDLM baseline failed on 37/40 tasks with syntax errors (`SYNTAX_ERROR`):
- **Repetitive punctuation collapse:** Without an autoregressive inductive bias or conversational chat-template prompt formatting, the diffusion model fills the unmasked blocks with degenerate repetitive tokens (e.g. repeated commas `,`, quotes `"`, or repeated identifiers `the the the`).
- **Instruction-Tuning Mismatch:** The `dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1` checkpoint was distilled on conversational chat datasets (`smoltalk`, `tulu-3`), not raw prefix completion. In unprompted function completion, it lacks the inductive prior to enforce Python grammar block-by-block.

---

## 5. Anatomy of COROSred Degradation (42.5% $\to$ 20.0%)

In COROSred, routing triggered on **100% of tasks (40/40)** with an average of **23.0 uncertain tokens** flagged per sequence. While the repair step was fast (0.285s / task, 7.2% of total generation time), it caused catastrophic regressions across 9 previously working programs.

### 1. Scope Coherence & Variable Renaming Drift
- **Task `HumanEval/4` (mean_absolute_deviation):**
  - AR draft:
    ```python
    mean = sum(numbers) / len(numbers)
    return sum(abs(x - mean) for x in numbers) / len(numbers) if len(numbers) > 0 else 0.0
    ```
  - COROSred repair:
    ```python
    MAD = sum(numbers) / len(numbers)
    return sum(abs(x - mean) for x in numbers) / len(numbers) if len(numbers) != 0 else 0.0
    ```
  - **Failure:** MDLM changed `mean` to `MAD` at definition, but left the downstream usage of `mean` unchanged in surrounding context. Result: `NameError: name 'mean' is not defined` (`RUNTIME_EXCEPTION`).

### 2. Operator & Logic Corruption
- **Task `HumanEval/31` (is_prime):**
  - AR draft: `for i in range(3, int(n**0.5)+1, 2):` (Correct)
  - COROSred repair: `for i in range(3, int(n**0.5)**1, 2):` (Changed `+` to `**`, resulting in `int(n**0.5)**1` and `TypeError` / logic failure).
  - Also mutated `if n < 2: return False; if n == 2: return True` into `if n <= 2: return False; elif n == 2: return True` (making `n == 2` dead code!).

- **Task `HumanEval/8` (sum_product):**
  - AR draft: `return (total_sum, product) if numbers else (0, 1)`
  - COROSred repair: `return (total_sum, product) * numbers else (0, 1)` (Changed `if` to `*`, breaking valid syntax and logic).

- **Task `HumanEval/15` (string_sequence):**
  - AR draft: `return ' '.join(str(i) for i in range(n + 1))`
  - COROSred repair: `return ' '.join(str(i) for i in range(n, 1))` (Changed `+` to `,`, generating empty sequence for $n > 1$).

### 3. Container Type Mutation
- **Task `HumanEval/28` (concatenate):**
  - AR draft: `result = ''`
  - COROSred repair: `result = []` (Mutated accumulator type from `str` to `list`, while keeping `result += s`, returning a list rather than string).

### 4. Syntax Corruption on Working Token Sequences
- **Task `HumanEval/21` (rescale_to_unit):**
  - AR draft: `scaled = [ (x - min_val) / (max_val - min_val) for x in numbers ]`
  - COROSred repair: `scaled = [xx - min_val) / (max_val - min_val) for x in numbers ]` (Changed `(x` to `xx`, missing open parenthesis).

- **Task `HumanEval/23` (strlen):**
  - AR draft: `return len(string)` (Passed)
  - COROSred repair: Corrupted trailing tests into mismatched quotes and syntax errors (`SYNTAX_ERROR`).

---

## 6. Distinguishing Functional Generation from Reconstruction Probes

In our earlier isolated reconstruction probes (where 100% of surrounding context was ground-truth code):
- MDLM showed strong local denoising capability on isolated single-token blanks when the surrounding prefix and suffix were ground-truth code.

However, in **actual functional generation**:
1. The surrounding context is an **unverified AR draft**, not ground truth.
2. Code has strict global syntactic constraints (lexical scoping, matching parentheses, exact operator semantics).
3. The diffusion model's token distribution is not conditioned on causal execution or scope coherence across spans.
4. An uncalibrated uncertainty heuristic (`margin < 0.5` or `entropy > 1.0`) triggers on benign syntactic variants, inviting MDLM to overwrite valid tokens with locally plausible but globally breaking replacements.

---

## 7. Conclusions & Strategic Implications for Telos

1. **Off-the-shelf zero-shot COROSred does NOT provide an immediate drop-in improvement without joint fine-tuning.**
   - While the two checkpoints share identical architecture and parameter shapes, MDLM was trained as a standalone diffusion model, not a span-repair adapter for an AR model.
2. **Confidence Routing Requires Calibration for Code:**
   - Heuristics based solely on AR probability margin or entropy over-trigger on normal programming token ambiguity (naming choices, comments, formatting). A specialized reliability head or syntactic filtering (e.g. only routing within unclosed blocks or AST errors) is essential.
3. **Span Infilling vs. Isolated Point Token Remasking:**
   - Point-wise unmasking without re-evaluating the full abstract syntax tree induces variable renaming drift and operator corruption.
4. **Latency Advantage of One-Pass Repair:**
   - The one-pass bidirectional repair took only **0.285s** (7.2% of total runtime), proving that the computational budget of COROSred is negligible. What is lacking is not speed or parameter fit, but token-level repair calibration.

---

## 8. Artifact Locations
- Master summary: `research/corosred_humaneval_40tasks_2026-10-18/comparative_summary.json`
- Per-task comparisons: `research/corosred_humaneval_40tasks_2026-10-18/per_task_comparisons.json`
- Raw reports:
  - `research/corosred_humaneval_40tasks_2026-10-18/ar_eval_report.json`
  - `research/corosred_humaneval_40tasks_2026-10-18/mdlm_eval_report.json`
  - `research/corosred_humaneval_40tasks_2026-10-18/corosred_eval_report.json`
- Reproduction command:
  ```bash
  .venv/bin/python scripts/evaluate_telos_corosred_generation.py --suite humaneval --max-tasks 40 --output-dir research/corosred_humaneval_40tasks_2026-10-18
  ```
