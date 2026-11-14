# Tiny extraction models

Direct QA, extraction-only, and calculator results are separate tracks. Pipeline operations and requested source fields are human-specified.

| Model / track | Tasks | Exact extraction | Calculator answer | Direct QA |
|---|---:|---:|---:|---:|
| fastino/gliner2-base-v1 / extraction_and_pipeline | 512/512 | 214/512 | 214/512 | — |
| HuggingFaceTB/SmolLM2-360M-Instruct / raw and normalized records; per-case outputs saved incrementally | 96/512 | 5/96 | 5/96 | — |

Extraction means exact source fields; calculator means final status/value correctness after deterministic execution. Source-substring grounding is not semantic entailment. Empty missing fields are evaluated separately in summary.json. Incomplete runs are not full-suite scores.

See ../xerj_reader_2026-11-14/tiny_specialist_protocol.md for the frozen protocol.
