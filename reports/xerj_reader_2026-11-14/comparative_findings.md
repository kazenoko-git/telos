# Comparative Evaluation Findings: Reader-1 and Language Model Baselines (v2 Corrected)

14 November 2026 · 512 synthetic cases · 7 model architectures · SHA-256 `7aa446c23aa360a783b2ba5935e98e9305d67528ea55a7fea9a8f4cd30db529a`

This report records the comparative diagnostics on a frozen 512-case test suite across 12 structured document extraction categories. The evaluation disaggregates 240 previously inspected historical cases from 272 newly generated held-out cases to evaluate generalization. The evaluated cohort comprises seven fully evaluated models completing all 512 cases:
1. **Ministral 3 8B Instruct** (local Q4_K_M · LM Studio)
2. **Google Gemma 4 E4B** (local Q4 · LM Studio)
3. **LiquidAI LFM3 Omni 2B** (local 4-bit MLX direct resident worker)
4. **Ternary Bonsai 4B** (local 1.58-bit ternary MLX · LM Studio)
5. **XERJ Reader-1** (API + symbolic executor replay)
6. **Qwen 3.5 0.8B MLX** (local 4-bit MLX · LM Studio)
7. **Qwen 3.5 2B OptiQ** (local OptiQ 4-bit · LM Studio)

*Note on Gemini 3.5 Flash-Lite*: Gemini 3.5 Flash-Lite was removed from the benchmark cohort because its evaluation was incomplete (107/512 completed, 25 quota errors, 380 unattempted due to hitting the Google AI Studio 500 requests/day free-tier cap). To maintain rigorous parity where every reported model completes all 512 benchmark cases, Gemini has been excised from active scoreboards and figures.

Historical v1 results are preserved in `comparative_512_run_v1.json`. Following an independent audit (`independent_audit.md`) that identified 6 harness-level contract defects, this document details the corrected v2 benchmark results recorded in `comparative_512_v2_run.json`.

---

## 1. Audit Context and Harness Fixes (v1 to v2)

An independent read-only audit of the initial v1 run uncovered six harness issues that distorted model rankings:

1. **Money Schema Mismatch**: The initial adapter JSON schema forbade object values for `value`, whereas the evaluation oracle required `{"amount": str, "currency": str}` for 21 money cases in `corrections`. All general LLMs failed these 21 cases on formatting, while Reader-1 passed natively.
   - *Fix*: Updated the JSON schema and prompt contract to accept `{"amount": "...", "currency": "..."}`, and added strict money normalization.
2. **Percentage Syntax Normalization**: General models returning strings like `"10%"` were failed against numeric gold targets (`10`), conflating syntax choices with arithmetic failure.
   - *Fix*: Normalized percentage strings by stripping `%` and converting to Decimal for comparison, preserving genuine calculation errors.
3. **Date Representation**: The prompt requested type `date` without specifying ISO 8601, causing models that generated `"14 March 2024"` to fail against ISO targets.
   - *Fix*: Explicitly requested ISO `YYYY-MM-DD` in prompt instructions and added standard date parsing.
4. **LFM3 List Parsing Crash**: Decoding JSON list responses caused an unhandled `AttributeError: 'list' object has no attribute 'get'`, dropping all 43 list cases as transport errors and discarding raw text.
   - *Fix*: Handled list roots cleanly and captured `raw_response` before parsing across all adapters.
5. **Gemini Quota Throttling**: The 15 requests/minute free-tier quota on Google AI Studio caused 124 HTTP 429 drops during the v1 run.
   - *Fix*: Added a 4.5s thread-safe rate limiter and exponential backoff respecting `Retry-After`.
6. **Error Taxonomy**: Parser crashes and quota drops were previously pooled into `transport_error`.
   - *Fix*: Reclassified into distinct categories: `malformed_response`, `contract_error`, and `rate_limit_error`.

---

## 2. V1 vs V2 Comparative Scoreboard

Scores require matching semantic status (`ANSWER`, `CONFLICT`, `ABSENT`, etc.) and typed value. All 7 reported models completed all 512 frozen benchmark cases.

| Model Checkpoint | Architecture & Runtime | V1 Combined (512) | V2 Combined (512) | V2 Inspected (240) | V2 New (272) | Median Latency | Valid Offsets |
|---|---|---|---|---|---|---|---|
| `ministral-3-8b-instruct-2512` | Local 8B (GGUF Q4_K_M · LM Studio) | 455/512 (88.9%) | **466/512 (91.0%)** | 232/240 (96.7%) | 234/272 (86.0%) | 3618.9 ms | 95.3% |
| `google/gemma-4-e4b` | Local 4B (Q4 · LM Studio) | 377/512 (73.6%) | **431/512 (84.2%)** | 200/240 (83.3%) | 231/272 (84.9%) | 3338.8 ms | 100.0% |
| `lfm3-omni-2b` | Local 2B (MLX Direct Subprocess Worker) | 325/512 (63.5%) | **430/512 (84.0%)** | 192/240 (80.0%) | 238/272 (87.5%) | 578.8 ms | 93.4% |
| `ternary-bonsai-4b-mlx` | Local 4B (1.58-bit Ternary · LM Studio) | 366/512 (71.5%) | **412/512 (80.5%)** | 201/240 (83.8%) | 211/272 (77.6%) | 1106.7 ms | 92.2% |
| `xerj-reader-1` | Cloud API + Symbolic Executor Replay | 330/512 (64.5%) | **330/512 (64.5%)** | 162/240 (67.5%) | 168/272 (61.8%) | 293.0 ms | 100.0% (Native) |
| `qwen3.5-0.8b-mlx` | Local 0.8B (MLX 4-bit · LM Studio) | — | **282/512 (55.1%)** | 126/240 (52.5%) | 156/272 (57.4%) | 631.9 ms | 84.0% |
| `qwen3.5-2b-optiq` | Local 2B (OptiQ 4-bit · LM Studio) | — | **129/512 (25.2%)** | 60/240 (25.0%) | 69/272 (25.4%) | 1171.8 ms | 95.7% |

### Analysis of Score Shifts
- **LiquidAI LFM3 Omni 2B (+20.5%)**: Fixing the list parser crash and money schema caused LFM3's score to surge from 63.5% to 84.0%, achieving 43/43 on lists and 43/43 on corrections.
- **Ternary Bonsai 4B (+9.0%)**: Restoring money object compatibility moved corrections accuracy from 51% (22/43) to 100% (43/43), lifting overall performance to 80.5%.
- **Google Gemma 4 E4B (+10.6%)**: Money schema and percentage syntax normalization lifted Gemma to 84.2%, demonstrating balanced performance across edge categories.
- **XERJ Reader-1 (Stable at 64.5%)**: Reader-1 was unaffected by the general LLM schema bugs, reproducing its exact 162/240 (67.5%) inspected score and 168/272 (61.8%) on held-out cases.
- **Qwen 3.5 0.8B MLX (55.1%)**: The sub-1B model demonstrates remarkable strength on structured scalar extraction and corrections (100% on corrections, 100% on lists, 97.6% on Unicode, 95.3% on irrelevant instructions) at 631.9 ms latency, but lacks abstention calibration (0% on conflicts, 0% on missing).
- **Qwen 3.5 2B OptiQ (25.2%)**: Shows perfect abstention calibration (100% on conflicts, 100% on missing), but under JSON Schema constrained decoding, the model aggressively over-structured scalar values into nested objects (e.g. `{"total_tasks": 19}` instead of `19`), causing scalar oracle mismatches.

---

## 3. Corrected v2 Category Performance Across All 12 Capabilities

| Category | Ministral 3 8B | Gemma 4 E4B | LFM3 Omni 2B | Ternary Bonsai 4B | Reader-1 | Qwen 3.5 0.8B | Qwen 3.5 2B |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **arithmetic** | 35/43 (81%) | 40/43 (93%) | 35/43 (81%) | 32/43 (74%) | 35/43 (81%) | 25/43 (58%) | 0/43 (0%) |
| **conflicts** | 43/43 (100%) | 20/43 (47%) | 15/43 (35%) | 1/43 (2%) | 0/43 (0%) | 0/43 (0%) | 43/43 (100%) |
| **corrections** | 43/43 (100%) | 43/43 (100%) | 43/43 (100%) | 43/43 (100%) | 43/43 (100%) | 43/43 (100%) | 21/43 (49%) |
| **dates** | 35/43 (81%) | 26/43 (60%) | 29/43 (67%) | 27/43 (63%) | 43/43 (100%) | 26/43 (60%) | 0/43 (0%) |
| **irrelevant instructions** | 30/43 (70%) | 43/43 (100%) | 43/43 (100%) | 43/43 (100%) | 38/43 (88%) | 41/43 (95%) | 0/43 (0%) |
| **lists** | 43/43 (100%) | 43/43 (100%) | 43/43 (100%) | 43/43 (100%) | 43/43 (100%) | 43/43 (100%) | 22/43 (51%) |
| **missing** | 43/43 (100%) | 25/43 (58%) | 31/43 (72%) | 43/43 (100%) | 22/43 (51%) | 0/43 (0%) | 43/43 (100%) |
| **percentages** | 38/43 (88%) | 33/43 (77%) | 36/43 (84%) | 33/43 (77%) | 0/43 (0%) | 0/43 (0%) | 0/43 (0%) |
| **scalar** | 42/42 (100%) | 42/42 (100%) | 42/42 (100%) | 42/42 (100%) | 42/42 (100%) | 32/42 (76%) | 0/42 (0%) |
| **table lookup** | 42/42 (100%) | 42/42 (100%) | 42/42 (100%) | 40/42 (95%) | 10/42 (24%) | 23/42 (55%) | 0/42 (0%) |
| **unicode** | 42/42 (100%) | 42/42 (100%) | 42/42 (100%) | 42/42 (100%) | 42/42 (100%) | 41/42 (98%) | 0/42 (0%) |
| **units** | 30/42 (71%) | 32/42 (76%) | 29/42 (69%) | 23/42 (55%) | 12/42 (29%) | 8/42 (19%) | 0/42 (0%) |

---

## 4. Substantive Model Differences (Beyond Harness Effects)

With harness mismatches resolved, true model capability boundaries emerge:

1. **Unresolved Contradictions (`conflicts`)**:
   - `xerj-reader-1` (0/43), `ternary-bonsai-4b` (1/43), and `qwen3.5-0.8b` (0/43) consistently pick one of two conflicting facts rather than returning the required `CONFLICT` status.
   - `ministral-3-8b` (43/43) and `qwen3.5-2b` (43/43) reliably identify contradictory evidence and abstain.
2. **Percentages and Ratios (`percentages`)**:
   - General LLMs (Ministral, Gemma, LFM3, Bonsai) score 77% to 88% on percentage calculations under normalized syntax.
   - `xerj-reader-1` scores **0/43**. Detailed trace inspection shows it consistently extracts a raw count (e.g. `120`) instead of executing division `DIV(...)`.
   - `qwen3.5-0.8b` and `qwen3.5-2b` score 0/43 on percentages due to transport/context drops (0.8B) and dictionary wrapping (2B).
3. **Multi-Condition Markdown Tables (`table_lookup`)**:
   - `ministral-3-8b`, `gemma-4`, and `lfm3-omni` achieve **42/42 (100%)**.
   - `xerj-reader-1` scores **10/42 (24%)**. Reader-1 frequently fails when two row/column filters must be applied simultaneously.
   - `qwen3.5-0.8b` reaches **23/42 (55%)**, demonstrating table lookup capability in a compact sub-1B architecture.
4. **Calendar and Dates (`dates`)**:
   - `xerj-reader-1` scores **43/43 (100%)** using its verified date normalization engine.
   - Small edge models struggle with multi-month date offsets (`gemma-4` 60%, `ternary-bonsai` 63%, `lfm3-omni` 67%, `qwen3.5-0.8b` 60%).

---

## 5. Latency, Efficiency, and Evidence Offset Grounding

| Model | Architecture | Median Model Latency | Client Median | Valid Offsets | Total Input Tokens | API Cost |
|---|---|---|---|---|---|---|
| `xerj-reader-1` | Cloud API + Symbolic Replay | **293.0 ms** | 1008.6 ms | **100.0%** (512/512) | 23,414 | $0.000234 |
| `lfm3-omni-2b` | Local 2B (MLX Resident Subprocess) | **578.8 ms** | 609.4 ms | **93.4%** (478/512) | 88,432 | $0.00 |
| `qwen3.5-0.8b-mlx` | Local 0.8B (MLX 4-bit · LM Studio) | **631.9 ms** | 632.1 ms | **84.0%** (430/512) | 92,150 | $0.00 |
| `ternary-bonsai-4b-mlx` | Local 4B (1.58-bit Ternary · LM Studio) | **1106.7 ms** | 1115.3 ms | **92.2%** (472/512) | 108,194 | $0.00 |
| `qwen3.5-2b-optiq` | Local 2B (OptiQ 4-bit · LM Studio) | **1171.8 ms** | 1172.0 ms | **95.7%** (490/512) | 93,420 | $0.00 |
| `google/gemma-4-e4b` | Local 4B (Q4 · LM Studio) | **3338.8 ms** | 3348.1 ms | **100.0%** (512/512) | 107,890 | $0.00 |
| `ministral-3-8b-instruct-2512` | Local 8B (Q4 · LM Studio) | **3618.9 ms** | 3622.4 ms | **95.3%** (488/512) | 106,712 | $0.00 |

### Key System Takeaways
- **Low-Latency Edge**: `lfm3-omni-2b` (578.8 ms) and `qwen3.5-0.8b-mlx` (631.9 ms) offer fast local inference, completing 512 cases in under 6 minutes each on Apple Silicon.
- **Reader-1 Specialization**: `xerj-reader-1` maintains the fastest raw execution (293.0 ms), lowest token count (23,414 tokens), and 100% executor replay consistency (128/128 sampled), but is bounded on complex multi-predicate table reasoning.
- **Evidence Offsets & Provenance**: Only XERJ Reader-1 produces native model-emitted character offsets directly from its model architecture. For all general-purpose models (Ministral 3 8B, Gemma 4 E4B, Ternary Bonsai 4B, LFM3 Omni 2B, Qwen 3.5 0.8B, and Qwen 3.5 2B), character offsets are **reconstructed post-hoc** by locating the model quotation in the source document (`document.find`). This measures literal quotation grounding, not native character coordinate prediction.
- **Money Normalization Audit**: The money parser was audited to ensure explicit 3-letter currency codes (e.g. CAD, USD, EUR, GBP) and symbols ($, €, £, ¥) are parsed without silently assuming USD for non-USD currencies (e.g. CAD 15 parses as CAD; bare 15 leaves currency as None). Gold cases in this benchmark use USD, so model passes reflect genuine extractions from document context.

---

## 6. Artifacts and Reproduction Commands

All evaluation artifacts are preserved in `/Users/ivansamuel/Documents/Projects/telos-evaluation-v1.6.0`:

- **Dataset**: `reports/xerj_reader_2026-11-14/suite_512_cases.json` (SHA-256: `7aa446c23aa360a783b2ba5935e98e9305d67528ea55a7fea9a8f4cd30db529a`)
- **Manifest**: `reports/xerj_reader_2026-11-14/suite_512_manifest.json`
- **Adapters**: `reports/xerj_reader_2026-11-14/model_adapters.py` (with 6 audit fixes and Qwen support)
- **v1 Run Baseline**: `reports/xerj_reader_2026-11-14/comparative_512_run_v1.json`
- **v2 Run Results**: `reports/xerj_reader_2026-11-14/comparative_512_v2_run.json`
- **Qwen 0.8B Run**: `reports/xerj_reader_2026-11-14/comparative_512_qwen08b.json`
- **Qwen 2B Run**: `reports/xerj_reader_2026-11-14/comparative_512_qwen2b.json`
- **Interactive Dashboard**: `reports/xerj_reader_2026-11-14/comparative_display.html`
- **Publication Figure**: `figures/xerj_reader_comparative_2026-11-14.png`
- **Deployment Guide**: `DEPLOYMENT.md`

### Reproduction Commands
```bash
# Evaluate Qwen models individually
python3 reports/xerj_reader_2026-11-14/run_comparative_suite.py --models qwen3.5-0.8b-mlx --output comparative_512_qwen08b.json
python3 reports/xerj_reader_2026-11-14/run_comparative_suite.py --models qwen3.5-2b-optiq --output comparative_512_qwen2b.json

# Merge constituent v2 runs
python3 reports/xerj_reader_2026-11-14/merge_comparative_runs.py --v2

# Regenerate standalone HTML comparative display
python3 reports/xerj_reader_2026-11-14/generate_comparative_display.py

# Regenerate publication overview figure
/Users/ivansamuel/Documents/Projects/telos/.venv/bin/python scripts/generate_reader_comparative_figure.py
```
