# Comparative Evaluation Report: 512-Case Benchmark Suite

Generated UTC: 2026-11-14T18:09:57.162810+00:00
Frozen Cases File: suite_512_cases.json (SHA-256: `7aa446c23aa360a783b2ba5935e98e9305d67528ea55a7fea9a8f4cd30db529a`)
Total Benchmark Cases: 512 (240 inspected historical + 272 new held-out)

## Overall Comparative Scoreboard

| Model | Runtime | Inspected 240 Acc | New 272 Acc | Combined 512 Acc | Offset Valid % | Total Cost | Median Latency |
|---|---|---|---|---|---|---|---|
| `qwen3.5-0.8b-mlx` | local/API | 52.5% (126/240) | 57.35% (156/272) | **55.08%** (282/512) | 83.98% | $0.000000 | 631.9 ms |

## Outcome Classifications (Combined 512 Cases)

| Model | Correct Ans | Correct Abst | Incorrect Ans | Incorrect Abst | Malformed | Contract Err | Rate Limit Err | Transport Err |
|---|---|---|---|---|---|---|---|---|
| `qwen3.5-0.8b-mlx` | 282 | 0 | 172 | 0 | 0 | 0 | 0 | 58 |

## Category Accuracies (Combined 512 Cases)

| Category | `qwen3.5-0.8b-mlx` |
|---|---|
| arithmetic | 25/43 (58.14%) |
| conflicts | 0/43 (0.0%) |
| corrections | 43/43 (100.0%) |
| dates | 26/43 (60.47%) |
| irrelevant_instructions | 41/43 (95.35%) |
| lists | 43/43 (100.0%) |
| missing | 0/43 (0.0%) |
| percentages | 0/43 (0.0%) |
| scalar | 32/42 (76.19%) |
| table_lookup | 23/42 (54.76%) |
| unicode | 41/42 (97.62%) |
| units | 8/42 (19.05%) |

