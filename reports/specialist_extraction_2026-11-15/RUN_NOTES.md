# Completed tiny-model runs

All three requested models completed the frozen 512-case suite. SmolLM2 additionally completed a separate direct-QA run. Raw outputs and prompts remain in the model artifacts; no existing comparison or first-pass record was replaced.

| Model | Exact source extraction | Calculator pipeline | Extraction format failures |
|---|---:|---:|---:|
| SmolLM2 360M Instruct | 10/512 | 10/512 | 275 |
| NuExtract 1.5 Tiny | 88/512 | 89/512 | 27 |
| GLiNER2 Base | 214/512 | 214/512 | 0 |

These are strict question-conditioned extraction results under the recorded templates, not general model benchmark scores. Extraction requires all requested source operands to match the reference. Pipeline correctness requires the final status/value after a human-specified operation. Neither track is merged into the existing direct-QA leaderboard.

SmolLM2 direct QA achieved 1/512 status/value contract successes. There were 404 malformed JSON/object outputs and 11 unsupported statuses; the remaining valid-contract answers can still be wrong. Unconstrained MLX decoding differs from the structured decoding used by prior LM Studio baselines. For example arithmetic-03 states the correct total in prose but fails the JSON contract. Do not interpret 1/512 as a measurement of raw arithmetic ability. An independent city-name prompt produced `Paris`, confirming ordinary generation works.

Re-encoding retained output indicates 176 SmolLM2 extraction completions and six NuExtract completions reached the 256-token budget; these are tokenizer-derived cap flags, not native finish reasons (the one-shot API did not expose those). All malformed output is retained. Native NuExtract uses its official Template/Text format with string placeholders for singular fields and arrays for plural fields. This interface was frozen after preflight, before its full run; the minimal required output fits the 256-token budget for all tasks (maximum measured 164 tokens). No full-suite answer examples were supplied.

GLiNER was run on CPU with four threads, default threshold 0.5, native structured extraction, and native span coordinates. Its API confidence-orders spans. Canonicalizing those positions to document order changed the recorded extraction/pipeline result from 204 to 214; the unchanged inference output is separately archived in gliner2_base_native_order.json. This is an adapter correction, not a new inference attempt or selection of favorable answers. Every one of its 756 returned native spans matches its document offsets.

Absence/conflict cases are separated from the 426 answer-present cases in summary.json. Of exact extraction successes, GLiNER has 139/426 answer-present and 75/86 absence/conflict; NuExtract has 45/426 and 43/86; SmolLM2 has 10/426 and 0/86. This prevents empty-output abstention successes from concealing low recall. Literal substring grounding is recorded separately and does not establish semantic support.

All 2,048 completed task records (three extraction runs plus Smol direct QA) have unique known case IDs within their track. audit_completed_tracks.py independently recomputes every saved correctness decision, extraction grade, and valid-field executor result. It verifies suite/task-plan hashes and checks for serialized reference/gold fields in prompts. The six contract boundary tests passed. Model revisions, weight hashes, settings and isolated runtimes are recorded; no training dependencies were changed. The active Smol extraction process did not capture its startup source commit/hash, which is explicitly disclosed; exact prompts/settings and model hash are retained.

Reproduce summaries with summarize_tracks.py; replay grades with audit_completed_tracks.py; render the cream/logo overview with render_overview.py using the project environment containing Matplotlib. The source-operand protocol is ../xerj_reader_2026-11-14/tiny_specialist_protocol.md. The raw per-model artifacts are the files named in summary.json.

NuExtract3 remains deferred to the next round. Nothing was emailed, externally published, or released.
