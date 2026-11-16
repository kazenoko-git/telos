# Five local model evaluations: what the saved results show

Ivan Samuel · Wing It Research · Review draft prepared 28 October 2026

We evaluated Apple’s on-device Foundation Models interface, Ternary Bonsai 4B,
IBM Granite, LiquidAI LFM, and Google Gemma using the Telos harness. The retained
reports cover the same 3,147 task IDs per model across seven suites. The table
below is regenerated from individual recorded outcomes, rather than the older
master summaries. This verifies archive consistency; it does not independently
validate every answer, scorer, or original model configuration.

**These are saved success rates, not a controlled pass@1 leaderboard.** Some
reports combine original responses with successful selective retries, and the
models used different generation budgets, prompts, and answer extraction.

| Suite | Tasks | AFM default | Bonsai | Granite | LFM | Gemma |
|---|---:|---:|---:|---:|---:|---:|
| HumanEval | 164 | 62.80% (103/164) | 71.95% (118/164) | 60.37% (99/164) | 45.73% (75/164) | 82.32% (135/164) |
| Telos tool-use | 30 | 73.33% (22/30) | 63.33% (19/30) | 70.00% (21/30) | 63.33% (19/30) | 80.00% (24/30) |
| Telos cybersecurity | 50 | 12.00% (6/50) | 16.00% (8/50) | 22.00% (11/50) | 24.00% (12/50) | 56.00% (28/50) |
| GPQA Diamond | 198 | 39.39% (78/198) | 26.77% (53/198) | 28.79% (57/198) | 20.71% (41/198) | 43.94% (87/198) |
| MMLU Science selection | 833 | 69.75% (581/833) | 62.55% (521/833) | 61.82% (515/833) | 73.11% (609/833) | 58.58% (488/833) |
| Competition MATH selection | 700 | 52.00% (364/700) | 56.71% (397/700) | 64.71% (453/700) | 43.57% (305/700) | 67.14% (470/700) |
| ARC-Challenge | 1,172 | 85.32% (1000/1172) | 54.61% (640/1172) | 86.60% (1015/1172) | 59.56% (698/1172) | 66.72% (782/1172) |

Gemma has the highest recorded success rate in four suites: HumanEval, tool-use,
cybersecurity, and GPQA. LFM leads the recorded MMLU selection; Granite leads
the recorded MATH selection and ARC. These are descriptive comparisons of the
retained reports, not evidence of model superiority under matched conditions.

## What was measured

- HumanEval: recorded Python execution outcomes, not merely AST validity.
- Tool-use: Telos’s 30-task local tool-name and argument-matching suite. This is
  **not an official Berkeley Function Calling Leaderboard result**. The stored
  configuration references a different 100-task JSON file, but the evaluator
  uses the 30-task Python suite; the exported denominator is the actual 30 rows.
- Cybersecurity: a 50-task local CWE/remediation rubric using identifier and
  required/forbidden code-pattern checks. This is not a proof of exploit resistance
  or comprehensive security correctness. No causal refusal explanation is claimed.
- Science/math: the retained GPQA, MMLU, MATH and ARC task selections, scored by
  the local answer extraction and comparison logic. MMLU and MATH selections
  should not be presented as their entire parent benchmarks.

## Model identity and execution

- Apple: the historical bridge used `SystemLanguageModel.default`. The label
  `afm-3-core-advanced` was a harness identifier, not a verified model selector.
  Publish this as Apple AFM, system-default model, variant unverified. Current
  hardware or current SDK capabilities cannot establish the historical variant.
- Bonsai: saved loaded-model metadata identifies
  `prism-ml/Ternary-Bonsai-4B-mlx-2bit`, 2-bit, via LM Studio. The configuration’s
  generic “4-bit MLX” text is contradicted by that metadata.
- Gemma: saved loaded-model metadata identifies `google/gemma-4-e4b`, 4-bit,
  via LM Studio. The reported file size is not measured runtime memory.
- Granite and LFM: retained run labels identify Granite 4.2 3B MLX and
  LFM 2.5 8B A1B MLX through LM Studio. Exact checkpoint revisions and loaded
  quantization are not established by the summaries alone.
- Phi-4 Mini has only three suites in its master summary and is excluded from
  this seven-suite comparison.

## Retry and budget limits

The reports explicitly flag successful recoveries at 4,096 tokens: AFM has 28
GPQA and 49 MATH rows; Granite has 35 GPQA and 204 MATH rows. Granite also has
16 MMLU and 572 ARC rows marked `repaired_16000_tokens`. That marker’s name is
not reliable evidence of its token budget: the current repair script requests
4,096 tokens. These counts are markers in the saved outcomes, not complete
counts of attempts. Failed retry attempts may not be fully represented.

Gemma and Bonsai configurations specify 1,024 tokens for ARC, 2,048 for MMLU,
HumanEval and tool-use, 8,192 for GPQA and MATH, and 4,096 for cybersecurity.
Reasoning/final-answer handling also differs between adapters. A clean matched
comparison requires fresh immutable first-attempt records under a common policy.
No new inference was run to prepare this publication draft.

## Corrections and reproducibility

Bonsai’s master summary disagrees with five detailed success rates: HumanEval
73.78 → 71.95; GPQA 32.32 → 26.77; MMLU 63.75 → 62.55; MATH 58.14 → 56.71;
ARC 55.38 → 54.61. The older social graphic must not be reused. Gemma’s summary
also simplifies HumanEval failure categories and claims 100% AST validity,
whereas the detailed report contains syntax failures. Detailed reports govern
this draft; historical files have been preserved.

`results.json` records report paths, SHA-256 hashes, original timestamps,
retry markers and all detected summary discrepancies. `results.csv` contains
suite scores. `task_outcomes.csv` contains task IDs and outcomes without prompts
or model completions. The exporter verifies report denominators, unique IDs,
identical task-ID sets across all five models, and task-derived scores.

From the Telos repository, regenerate with:

```sh
python3 scripts/prepare_public_evaluation_archive.py
```

These results do not establish equal-compute performance, memory efficiency,
statistical superiority, or a “first external evaluation” claim. Source datasets
and scorer versions need a historical provenance review before describing this
as a fully reproducible benchmark release. Raw dataset questions and responses
are deliberately not included in this publication bundle.
