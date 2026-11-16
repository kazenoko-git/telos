# Reader-1 API evaluation

All documents are synthetic. These results do not measure accuracy on real Télos logs.

Answer correctness: 0/0 model responses.
API or replay errors: 1.
Exact evidence offsets: 0/0 responses (empty evidence passes this check).
Answers with evidence: 0/0 responses (abstentions exempt).
Deterministic replay: 0/0 match in status and value.
Input tokens: 0. Estimated charge at $0.01/M input tokens: $0.00000000.

Evidence offset validity establishes text consistency, not semantic relevance. Replay tests consistency of execution, not correctness of the model-selected program.

| Category | Correct / responses |
|---|---|

| Case | Correct | Status | Value | Confidence | Client ms |
|---|---|---|---|---|---|
| baseline-01 | error | error | null | None |  |

Raw responses, evidence and replay outputs: run.json.
Expected answers were specified in cases.json before API execution. Missing-value tests accept ABSENT or UNKNOWN; the unresolved-conflict test accepts CONFLICT or AMBIGUOUS.
Single pass per case; small hand-authored suite, no general benchmark claim.
