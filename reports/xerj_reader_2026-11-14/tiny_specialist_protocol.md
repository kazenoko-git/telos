# Tiny-model evaluation protocol

## Models and tracks

- HuggingFaceTB/SmolLM2-360M-Instruct: direct question answering, source-operand extraction, and deterministic pipeline.
- numind/NuExtract-1.5-tiny: source-operand extraction and deterministic pipeline, using its extraction template.
- fastino/gliner2-base-v1: source-operand extraction and deterministic pipeline, using its native schema interface.
- NuExtract3 is deferred until these runs are complete.

Use the existing frozen 512 tasks. Preserve original benchmark records and keep all new outputs in distinct artifacts. Record downloaded model revision/weight hash, runtime versions, device, prompt/template, generation settings, elapsed client duration, and every failed attempt. No gold answer or reference operand enters inference.

## Separate metrics

**Direct QA** uses the original question and requested answer type. Status/value correctness is scored against the original frozen gold. This is the track comparable to the existing general-model status/value leaderboard. It does not require semantically supporting evidence; report grounding separately.

**Extraction-only** measures source operands, not computed answers. All models receive the document, original question, and the same named-field descriptions from `pipeline_task_plans.json`. Model-specific formatting is allowed and saved. Field values must be lists of literal document strings; absent fields are empty lists. The scorer measures exact reference fields and source-substring grounding separately. Ordered-list tasks preserve order; other fields treat distinct values as unordered. Report exact-case counts overall, per category, and separately for answer-present versus absence/conflict cases so trivial empty outputs cannot conceal failures.

**Calculator pipeline** consumes only extracted fields and a public human-written operation plan. Arithmetic, percentage, unit and calendar calculations use Decimal/datetime. Missing values and competing values use the same deterministic resolution policy as direct fields. No regex operand oracle runs on model outputs to supply missing inputs. The private reference fields are used only by the extraction grader. Executor results are evaluated against the original answer/status gold.

The pipeline is a configured extraction-plus-calculation system. It is not evidence that the extractor learned to generate a program, and its results must not be mixed into the direct-model leaderboard. Its operation plans and field descriptions are extra human scaffolding relative to original Reader-1/direct-QA runs. They are frozen before the full inference runs, after inspecting known suite templates, so this is a diagnostic comparison rather than a blind held-out benchmark.

## Validation and reporting

Commit `specialist_contract.py`, `build_specialist_plans.py`, and the 512 frozen plans before full inference. The reference extraction fields are built from document literals, and all 512 reference executions must agree with original gold. This is a reproducibility/contract check, not independent real-world validation. Test that changing extracted operands changes executor output and that public plans exclude references.

Retain malformed responses as malformed, runtime failures as runtime failures, and retries separately. Do not silently award final-answer credit to a wrapped/object answer where a scalar is required. For extraction adapters, save both native output and normalized named fields. Record any changes discovered by smoke tests before full runs. Select no model or prompt based on favorable full-suite outcomes.

Show separate direct-QA, extraction-only, and pipeline columns with denominators. Report task coverage and errors, not just completed-response accuracy. Grounding is literal quote matching, not semantic entailment. Model/runtime claims must be limited to the device and configuration actually measured. No release, external publication, or email is part of this run.
