# Comparative Evaluation Report: 512-Case Benchmark Suite

Generated UTC: 2026-11-14T16:00:25.260615+00:00
Frozen Cases File: suite_512_cases.json (SHA-256: `7aa446c23aa360a783b2ba5935e98e9305d67528ea55a7fea9a8f4cd30db529a`)
Total Benchmark Cases: 512 (240 inspected historical + 272 new held-out)

## Overall Comparative Scoreboard

| Model | Runtime | Inspected 240 Acc | New 272 Acc | Combined 512 Acc | Offset Valid % | Total Cost | Median Latency |
|---|---|---|---|---|---|---|---|
| `ternary-bonsai-4b-mlx` | local/API | 83.75% (201/240) | 77.57% (211/272) | **80.47%** (412/512) | 92.19% | $0.000000 | 1106.68 ms |
| `lfm3-omni-2b` | local/API | 80.0% (192/240) | 87.5% (238/272) | **83.98%** (430/512) | 93.36% | $0.000000 | 578.76 ms |
| `ministral-3-8b-instruct-2512` | local/API | 96.67% (232/240) | 86.03% (234/272) | **91.02%** (466/512) | 95.31% | $0.000000 | 3618.95 ms |
| `google/gemma-4-e4b` | local/API | 83.33% (200/240) | 84.93% (231/272) | **84.18%** (431/512) | 100.0% | $0.000000 | 3338.85 ms |

## Outcome Classifications (Combined 512 Cases)

| Model | Correct Ans | Correct Abst | Incorrect Ans | Incorrect Abst | Malformed | Contract Err | Transport Err |
|---|---|---|---|---|---|---|---|
| `ternary-bonsai-4b-mlx` | 368 | 44 | 55 | 42 | 0 | 0 | 3 |
| `lfm3-omni-2b` | 384 | 46 | 52 | 23 | 0 | 0 | 7 |
| `ministral-3-8b-instruct-2512` | 380 | 86 | 31 | 13 | 0 | 0 | 2 |
| `google/gemma-4-e4b` | 386 | 45 | 48 | 23 | 0 | 0 | 10 |

## Category Accuracies (Combined 512 Cases)

| Category | `ternary-bonsai-4b-mlx` | `lfm3-omni-2b` | `ministral-3-8b-instruct-2512` | `google/gemma-4-e4b` |
|---|---|---|---|---|
| arithmetic | 32/43 (74.42%) | 35/43 (81.4%) | 35/43 (81.4%) | 40/43 (93.02%) |
| conflicts | 1/43 (2.33%) | 15/43 (34.88%) | 43/43 (100.0%) | 20/43 (46.51%) |
| corrections | 43/43 (100.0%) | 43/43 (100.0%) | 43/43 (100.0%) | 43/43 (100.0%) |
| dates | 27/43 (62.79%) | 29/43 (67.44%) | 35/43 (81.4%) | 26/43 (60.47%) |
| irrelevant_instructions | 43/43 (100.0%) | 43/43 (100.0%) | 30/43 (69.77%) | 43/43 (100.0%) |
| lists | 43/43 (100.0%) | 43/43 (100.0%) | 43/43 (100.0%) | 43/43 (100.0%) |
| missing | 43/43 (100.0%) | 31/43 (72.09%) | 43/43 (100.0%) | 25/43 (58.14%) |
| percentages | 33/43 (76.74%) | 36/43 (83.72%) | 38/43 (88.37%) | 33/43 (76.74%) |
| scalar | 42/42 (100.0%) | 42/42 (100.0%) | 42/42 (100.0%) | 42/42 (100.0%) |
| table_lookup | 40/42 (95.24%) | 42/42 (100.0%) | 42/42 (100.0%) | 42/42 (100.0%) |
| unicode | 42/42 (100.0%) | 42/42 (100.0%) | 42/42 (100.0%) | 42/42 (100.0%) |
| units | 23/42 (54.76%) | 29/42 (69.05%) | 30/42 (71.43%) | 32/42 (76.19%) |

