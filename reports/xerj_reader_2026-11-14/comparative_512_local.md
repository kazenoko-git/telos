# Comparative Evaluation Report: 512-Case Benchmark Suite

Generated UTC: 2026-11-14T13:54:11.565882+00:00
Frozen Cases File: suite_512_cases.json (SHA-256: `7aa446c23aa360a783b2ba5935e98e9305d67528ea55a7fea9a8f4cd30db529a`)
Total Benchmark Cases: 512 (240 inspected historical + 272 new held-out)

## Overall Comparative Scoreboard

| Model | Runtime | Inspected 240 Acc | New 272 Acc | Combined 512 Acc | Offset Valid % | Total Cost | Median Latency |
|---|---|---|---|---|---|---|---|
| `gemini-3.5-flash-lite` | local/API | 0.0% (0/240) | 0.0% (0/272) | **0.0%** (0/512) | 100.0% | $0.000000 | 0.0 ms |
| `ternary-bonsai-4b-mlx` | local/API | 73.33% (176/240) | 69.85% (190/272) | **71.48%** (366/512) | 97.27% | $0.000000 | 1213.05 ms |
| `lfm3-omni-2b` | local/API | 58.33% (140/240) | 68.01% (185/272) | **63.48%** (325/512) | 93.95% | $0.000000 | 555.42 ms |
| `ministral-3-8b-instruct-2512` | local/API | 92.08% (221/240) | 86.03% (234/272) | **88.87%** (455/512) | 94.53% | $0.000000 | 2624.6 ms |
| `google/gemma-4-e4b` | local/API | 73.75% (177/240) | 73.53% (200/272) | **73.63%** (377/512) | 99.41% | $0.000000 | 1999.07 ms |

## Outcome Classifications (Combined 512 Cases)

| Model | Correct Ans | Correct Abst | Incorrect Ans | Incorrect Abst | Malformed | Transport Err |
|---|---|---|---|---|---|---|
| `gemini-3.5-flash-lite` | 0 | 0 | 0 | 0 | 0 | 512 |
| `ternary-bonsai-4b-mlx` | 323 | 43 | 95 | 43 | 0 | 8 |
| `lfm3-omni-2b` | 253 | 72 | 115 | 14 | 0 | 58 |
| `ministral-3-8b-instruct-2512` | 369 | 86 | 52 | 3 | 0 | 2 |
| `google/gemma-4-e4b` | 324 | 53 | 105 | 21 | 0 | 9 |

## Category Accuracies (Combined 512 Cases)

| Category | `gemini-3.5-flash-lite` | `ternary-bonsai-4b-mlx` | `lfm3-omni-2b` | `ministral-3-8b-instruct-2512` | `google/gemma-4-e4b` |
|---|---|---|---|---|---|
| arithmetic | 0/43 (0.0%) | 33/43 (76.74%) | 34/43 (79.07%) | 37/43 (86.05%) | 41/43 (95.35%) |
| conflicts | 0/43 (0.0%) | 0/43 (0.0%) | 29/43 (67.44%) | 43/43 (100.0%) | 22/43 (51.16%) |
| corrections | 0/43 (0.0%) | 22/43 (51.16%) | 22/43 (51.16%) | 22/43 (51.16%) | 22/43 (51.16%) |
| dates | 0/43 (0.0%) | 7/43 (16.28%) | 13/43 (30.23%) | 34/43 (79.07%) | 4/43 (9.3%) |
| irrelevant_instructions | 0/43 (0.0%) | 43/43 (100.0%) | 23/43 (53.49%) | 40/43 (93.02%) | 43/43 (100.0%) |
| lists | 0/43 (0.0%) | 43/43 (100.0%) | 0/43 (0.0%) | 43/43 (100.0%) | 43/43 (100.0%) |
| missing | 0/43 (0.0%) | 43/43 (100.0%) | 43/43 (100.0%) | 43/43 (100.0%) | 31/43 (72.09%) |
| percentages | 0/43 (0.0%) | 28/43 (65.12%) | 6/43 (13.95%) | 37/43 (86.05%) | 13/43 (30.23%) |
| scalar | 0/42 (0.0%) | 42/42 (100.0%) | 42/42 (100.0%) | 42/42 (100.0%) | 42/42 (100.0%) |
| table_lookup | 0/42 (0.0%) | 40/42 (95.24%) | 42/42 (100.0%) | 42/42 (100.0%) | 42/42 (100.0%) |
| unicode | 0/42 (0.0%) | 42/42 (100.0%) | 42/42 (100.0%) | 42/42 (100.0%) | 42/42 (100.0%) |
| units | 0/42 (0.0%) | 23/42 (54.76%) | 29/42 (69.05%) | 30/42 (71.43%) | 32/42 (76.19%) |

