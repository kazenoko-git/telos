# Reader-1 API evaluation

All documents are synthetic. These results do not measure accuracy on real Télos logs.

Answer correctness: 2/6 model responses.
API or replay errors: 0.
Exact evidence offsets: 6/6 responses (empty evidence passes this check).
Answers with evidence: 6/6 responses (abstentions exempt).
Deterministic replay: 6/6 match in status and value.
Input tokens: 406. Estimated charge at $0.01/M input tokens: $0.00000406.

Evidence offset validity establishes text consistency, not semantic relevance. Replay tests consistency of execution, not correctness of the model-selected program.

| Category | Correct / responses |
|---|---|
| table | 0 / 2 |
| table-diagnostic | 2 / 4 |

| Case | Correct | Status | Value | Confidence | Client ms |
|---|---|---|---|---|---|
| pipe-repeat-1 | False | ANSWER | "2.90" | 0.9842 | 1487.9 |
| pipe-repeat-2 | False | ANSWER | "2.90" | 0.9842 | 1445.25 |
| csv-table-11 | False | ANSWER | "2.90" | 0.9828 | 1663.39 |
| csv-table-12 | False | ABSENT | null | 0.946 | 1206.92 |
| prose-table-11 | True | ANSWER | "2.20" | 0.9738 | 937.92 |
| prose-table-12 | True | ANSWER | "3.10" | 0.9599 | 1140.97 |

Raw responses, evidence and replay outputs: table-probes-run.json.
Expected answers were specified in table_probes.json before this run. Missing-value tests accept ABSENT or UNKNOWN; the unresolved-conflict test accepts CONFLICT or AMBIGUOUS.
Single pass per case; small hand-authored suite, no general benchmark claim.
