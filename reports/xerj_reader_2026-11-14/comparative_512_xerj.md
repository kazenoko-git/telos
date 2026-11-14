# Comparative Evaluation Report: 512-Case Benchmark Suite

Generated UTC: 2026-11-14T13:03:04.709016+00:00
Frozen Cases File: suite_512_cases.json (SHA-256: `7aa446c23aa360a783b2ba5935e98e9305d67528ea55a7fea9a8f4cd30db529a`)
Total Benchmark Cases: 512 (240 inspected historical + 272 new held-out)

## Overall Comparative Scoreboard

| Model | Runtime | Inspected 240 Acc | New 272 Acc | Combined 512 Acc | Offset Valid % | Total Cost | Median Latency |
|---|---|---|---|---|---|---|---|
| `xerj-reader-1` | local/API | 67.5% (162/240) | 61.76% (168/272) | **64.45%** (330/512) | 100.0% | $0.000234 | 293.0 ms |

## Outcome Classifications (Combined 512 Cases)

| Model | Correct Ans | Correct Abst | Incorrect Ans | Incorrect Abst | Malformed | Transport Err |
|---|---|---|---|---|---|---|
| `xerj-reader-1` | 308 | 22 | 146 | 15 | 21 | 0 |

## Category Accuracies (Combined 512 Cases)

| Category | `xerj-reader-1` |
|---|---|
| arithmetic | 35/43 (81.4%) |
| conflicts | 0/43 (0.0%) |
| corrections | 43/43 (100.0%) |
| dates | 43/43 (100.0%) |
| irrelevant_instructions | 38/43 (88.37%) |
| lists | 43/43 (100.0%) |
| missing | 22/43 (51.16%) |
| percentages | 0/43 (0.0%) |
| scalar | 42/42 (100.0%) |
| table_lookup | 10/42 (23.81%) |
| unicode | 42/42 (100.0%) |
| units | 12/42 (28.57%) |

