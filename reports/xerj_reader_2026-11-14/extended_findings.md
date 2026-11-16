# Reader-1 expanded diagnostic results

14 November 2026 · 240 synthetic cases · xerj-reader-1

**162/240 correct (67.5%)** under expected status-and-value checks fixed before this run. The extension contains 20 cases per category, with two template forms and varied parameters. It was chosen after pilot failures; it is not a held-out benchmark or an estimate of accuracy on real documents. Do not pool it with the pilot and diagnostic repeats as if all cases were independently sampled.

| Category | Correct |
|---|---|
| arithmetic | 20/20 |
| conflicts | 0/20 |
| corrections | 20/20 |
| dates | 20/20 |
| irrelevant instructions | 18/20 |
| lists | 20/20 |
| missing | 11/20 |
| percentages | 0/20 |
| scalar | 20/20 |
| table lookup | 8/20 |
| unicode | 20/20 |
| units | 5/20 |

All 240 requests returned model responses with no transport errors. Every fourth case was selected for executor replay before the run: all 60 sampled replays matched in status and typed value. Evidence offsets exactly matched document slices in all 240 responses, including responses with empty evidence. These consistency checks do not verify the semantic relevance of the selected evidence.

42 incorrect responses had confidence at least 0.90. A threshold alone would not remove the observed wrong selections. No threshold was used when scoring this run.

Input usage: 9,224 tokens. At the documented $0.01 per million input tokens, the estimated model charge is $0.00009224. Output and execute calls are documented as free. This is a list-price estimate, not a balance reconciliation. Median client-observed model latency was 930.905 ms; it excludes the local rate-limiter wait. Requests were spaced at least 1.15 seconds apart across model and execute endpoints, with three workers. One attempt was made per case.

## Interpretation

Scalar extraction, arithmetic, corrections, spelled-month dates, ordered lists and Unicode cases passed all 20 cases in each category. The exact templates and parameters matter: this does not establish perfect performance across these capabilities.

Unresolved-conflict cases scored 0/20 against the accepted CONFLICT/AMBIGUOUS statuses. Percentage calculations scored 0/20; inspected examples extracted a raw operand instead of composing the requested percentage calculation. Multi-condition table lookups scored 8/20, and throughput calculations scored 5/20. These outcomes are more informative as failure reproductions than as a broad model ranking.

Cases containing irrelevant quoted instructions scored 18/20. This category tests answers in their presence, not comprehensive prompt-injection resistance; no causal claim that an instruction caused either failure is established. Missing-value cases scored 11/20. A type-check refusal can safely withhold a number while still failing the semantic-status requirement; inspect the actual status and formula before calling an error an invented value.

## Artifacts and reproduction

- `extended_cases.json`: frozen inputs and expected answers/statuses; SHA-256 in `extended_manifest.json` and `extended-run.json`.
- `generate_extended.py`: deterministic Decimal/datetime generator with schema checks. Generated directly in Codex after Agy Gemini 3.8 Flash High returned no generator before its three-minute timeout; no Agy output is claimed as evidence.
- `run_extended.py`: bounded concurrency, shared rate limiter, atomic per-completion snapshots, run timestamps and source hashes.
- `extended-run.json`: all 240 raw model responses and 60 executor replay outputs.
- `extended_failures.json`: self-contained cases and outputs for the 78 failed checks.
- `display.html`: portable cream notebook with expected values, source highlighting and formula inspection.
- `../../figures/xerj_reader_evaluation_2026-11-14.png`: publication-style overview.

```sh
python3 reports/xerj_reader_2026-11-14/run_extended.py --output extended-repeat.json
telos eval report reports/xerj_reader_2026-11-14/extended-run.json --cases reports/xerj_reader_2026-11-14/extended_cases.json --output reader-report.html
```

The pilot used 40 model calls and 40 replays. With this extension, the retained evaluation contains 280 model calls and 100 executor replays, excluding the initial Cloudflare-rejected request. Total reported input usage is 11,899 tokens, for estimated list-price cost $0.00011899. All submitted documents were synthetic. No feedback email has been sent.
