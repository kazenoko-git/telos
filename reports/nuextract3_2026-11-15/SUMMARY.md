# NuExtract3 Benchmark Evaluation

Evaluation of `numind/NuExtract3` (4-bit affine quantized on Apple Silicon MLX) across the frozen 512-case Télos document suite.

> [!IMPORTANT]
> Direct QA, extraction-only, and calculator pipeline tracks are strictly separated.
> - **Direct QA**: Original document and question mapped to native extraction JSON schema without intermediate plan.
> - **Extraction-Only**: Extracts exact source operands using human-specified frozen field descriptions; scored offline by `specialist_contract.py`.
> - **Calculator Pipeline**: Model-extracted operands fed into the deterministic calculation engine; scored against gold answers.

## Overall Track Scoreboard

| Track | Tasks | Exact Extraction | Grounded Ext. | Calculator Pipeline | Direct QA | Format Failures | Client Latency (p50 / p95) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Direct QA** | 512/512 | — | — | — | **466/512** (91.0%) | 0 | 1054 ms / 1693 ms |
| **Extraction + Calculator** | 512/512 | **426/512** (83.2%) | 512/512 (100.0%) | **443/512** (86.5%) | — | 0 | 298 ms / 422 ms |

## Historical Context Comparison

| Model | Model Size | Exact Extraction | Calculator Pipeline | Direct QA | Notes |
|---|---|---:|---:|---:|---|
| **NuExtract3** (MLX 4-bit) | 4B (Qwen 3.5 base) | *See above* | *See above* | *See above* | Native structured schema, non-thinking mode |
| Reader-1 Baseline | Reference | — | — | 330/512 (64.5%) | Frozen baseline reference |
| NuExtract-1.5-tiny | 0.5B | 421/512 (82.2%) | 416/512 (81.2%) | — | Previous generation tiny specialist |
| GLiNER2 Base | 205M | 394/512 (77.0%) | 382/512 (74.6%) | — | Bi-encoder token extraction specialist |
| SmolLM2-Instruct | 360M | 247/512 (48.2%) | 272/512 (53.1%) | 128/512 (25.0%) | General tiny LM baseline |

## Performance by Category

| Category | Cases | Direct QA | Extraction Exact | Calculator Pipeline |
|---|---:|---:|---:|---:|
| `arithmetic` | 43 | 40/43 | 43/43 | 43/43 |
| `conflicts` | 43 | 43/43 | 26/43 | 43/43 |
| `corrections` | 43 | 43/43 | 30/43 | 30/43 |
| `dates` | 43 | 36/43 | 43/43 | 43/43 |
| `irrelevant_instructions` | 43 | 12/43 | 9/43 | 9/43 |
| `lists` | 43 | 43/43 | 43/43 | 43/43 |
| `missing` | 43 | 43/43 | 43/43 | 43/43 |
| `percentages` | 43 | 41/43 | 43/43 | 43/43 |
| `scalar` | 42 | 42/42 | 31/42 | 31/42 |
| `table_lookup` | 42 | 42/42 | 42/42 | 42/42 |
| `unicode` | 42 | 42/42 | 31/42 | 31/42 |
| `units` | 42 | 39/42 | 42/42 | 42/42 |

## Answer-Present vs. Absence / Conflict Breakdown

| Group | Cases | Direct QA | Extraction Exact | Calculator Pipeline |
|---|---:|---:|---:|---:|
| `answer_present` | 426 | 380/426 | 357/426 | 357/426 |
| `absence_or_conflict` | 86 | 86/86 | 69/86 | 86/86 |

## Latency & Runtime Details

- **Hardware**: Apple Silicon (Apple M5 Pro, 24 GB Unified Memory, AC Power, Power Mode: High)
- **Runtime**: `mlx-lm` 0.32.2 on `mlx` 0.32.2, Python 3.12
- **Model**: `numind/NuExtract3-mlx-4bits` (revision `29c38269f94054282bf9ea97a20dfc6bb8bbefea`)
- **Precision**: 4-bit affine quantized (group_size: 64)
- **Thinking Mode**: `False` (frozen non-thinking mode for deterministic structured extraction)
- **Decoding**: Greedy (`temperature: 0.0`, `max_new_tokens: 256`)
- **Timing Definitions**:
  - `client_ms`: End-to-end wall-clock latency per case (tokenization, MLX generation, JSON parsing).
  - `generation_ms`: MLX inference generation step only.

## Verification & Integrity Assurances

- Zero prompt leaks: Suite gold answers and private reference fields were never exposed during inference.
- Complete auditability: All 512 case outputs, raw completion text, parsed JSON, and execution logs are preserved on disk.
- Independent replay: Extraction grades and deterministic pipeline executions re-verified offline via `audit_nuextract3.py`.
