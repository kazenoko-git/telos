# Comparative Evaluation Report: 512-Case Benchmark Suite

Generated UTC: 2026-11-14T18:22:10.872673+00:00
Frozen Cases File: suite_512_cases.json (SHA-256: `7aa446c23aa360a783b2ba5935e98e9305d67528ea55a7fea9a8f4cd30db529a`)
Total Benchmark Cases: 512 (240 inspected historical + 272 new held-out)

## Overall Comparative Scoreboard

| Model | Runtime | Inspected 240 Acc | New 272 Acc | Combined 512 Acc | Offset Valid % | Total Cost | Median Latency |
|---|---|---|---|---|---|---|---|
| `qwen3.5-2b-optiq` | local/API | 25.0% (60/240) | 25.37% (69/272) | **25.2%** (129/512) | 95.7% | $0.000000 | 1171.85 ms |

## Outcome Classifications (Combined 512 Cases)

| Model | Correct Ans | Correct Abst | Incorrect Ans | Incorrect Abst | Malformed | Contract Err | Rate Limit Err | Transport Err |
|---|---|---|---|---|---|---|---|---|
| `qwen3.5-2b-optiq` | 43 | 86 | 362 | 0 | 0 | 0 | 0 | 21 |

## Category Accuracies (Combined 512 Cases)

| Category | `qwen3.5-2b-optiq` |
|---|---|
| arithmetic | 0/43 (0.0%) |
| conflicts | 43/43 (100.0%) |
| corrections | 21/43 (48.84%) |
| dates | 0/43 (0.0%) |
| irrelevant_instructions | 0/43 (0.0%) |
| lists | 22/43 (51.16%) |
| missing | 43/43 (100.0%) |
| percentages | 0/43 (0.0%) |
| scalar | 0/42 (0.0%) |
| table_lookup | 0/42 (0.0%) |
| unicode | 0/42 (0.0%) |
| units | 0/42 (0.0%) |

