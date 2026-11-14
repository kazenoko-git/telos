# Comparative Evaluation Report: 512-Case Benchmark Suite

Generated UTC: 2026-11-14T13:36:34.705705+00:00
Frozen Cases File: suite_512_cases.json (SHA-256: `7aa446c23aa360a783b2ba5935e98e9305d67528ea55a7fea9a8f4cd30db529a`)
Total Benchmark Cases: 512 (240 inspected historical + 272 new held-out)

## Overall Comparative Scoreboard

| Model | Runtime | Inspected 240 Acc | New 272 Acc | Combined 512 Acc | Offset Valid % | Total Cost | Median Latency |
|---|---|---|---|---|---|---|---|
| `gemini-3.5-flash-lite` | local/API | 64.17% (154/240) | 65.07% (177/272) | **64.65%** (331/512) | 100.0% | $0.000000 | 1141.2 ms |

## Outcome Classifications (Combined 512 Cases)

| Model | Correct Ans | Correct Abst | Incorrect Ans | Incorrect Abst | Malformed | Transport Err |
|---|---|---|---|---|---|---|
| `gemini-3.5-flash-lite` | 265 | 66 | 53 | 2 | 0 | 126 |

## Category Accuracies (Combined 512 Cases)

| Category | `gemini-3.5-flash-lite` |
|---|---|
| arithmetic | 35/43 (81.4%) |
| conflicts | 36/43 (83.72%) |
| corrections | 14/43 (32.56%) |
| dates | 28/43 (65.12%) |
| irrelevant_instructions | 33/43 (76.74%) |
| lists | 30/43 (69.77%) |
| missing | 30/43 (69.77%) |
| percentages | 0/43 (0.0%) |
| scalar | 35/42 (83.33%) |
| table_lookup | 29/42 (69.05%) |
| unicode | 27/42 (64.29%) |
| units | 34/42 (80.95%) |

