# Reader-1 API evaluation

All documents are synthetic. These results do not measure accuracy on real Télos logs.

Answer correctness: 26/34 model responses.
API or replay errors: 0.
Exact evidence offsets: 34/34 responses (empty evidence passes this check).
Answers with evidence: 34/34 responses (abstentions exempt).
Deterministic replay: 34/34 match in status and value.
Input tokens: 2269. Estimated charge at $0.01/M input tokens: $0.00002269.

Evidence offset validity establishes text consistency, not semantic relevance. Replay tests consistency of execution, not correctness of the model-selected program.

| Category | Correct / responses |
|---|---|
| baseline | 6 / 6 |
| correction | 4 / 4 |
| table | 1 / 4 |
| unicode | 3 / 3 |
| dates | 3 / 3 |
| money | 2 / 3 |
| injection | 2 / 3 |
| absence | 2 / 3 |
| units | 2 / 3 |
| conflict | 1 / 2 |

| Case | Correct | Status | Value | Confidence | Client ms |
|---|---|---|---|---|---|
| baseline-01 | True | ANSWER | "2.10" | 0.9757 | 1437.15 |
| baseline-02 | True | ANSWER | "0.30" | 0.9234 | 1436.66 |
| baseline-03 | True | ANSWER | "25.00" | 0.9508 | 1201.47 |
| baseline-04 | True | ANSWER | "1" | 0.9994 | 1064.87 |
| baseline-05 | True | ABSENT | null | 0.3949 | 1711.63 |
| baseline-06 | True | ABSENT | null | 0.9998 | 1071.67 |
| correction-07 | True | ANSWER | "2.13" | 0.9981 | 1137.48 |
| correction-08 | True | ANSWER | "2.31" | 0.9173 | 1199.22 |
| correction-09 | True | ANSWER | "0.09" | 0.5103 | 1250.55 |
| correction-10 | True | ANSWER | "2.80" | 0.9925 | 1203.61 |
| table-11 | False | ANSWER | "2.90" | 0.9842 | 1408.18 |
| table-12 | False | ABSENT | null | 0.9301 | 1168.98 |
| table-13 | True | ANSWER | "0.60" | 0.9982 | 1438.81 |
| table-14 | False | ABSENT | null | 0.9253 | 1396.07 |
| unicode-15 | True | ANSWER | "1.25" | 0.9751 | 1160.19 |
| unicode-16 | True | ANSWER | "2400" | 0.9866 | 907.69 |
| unicode-17 | True | ANSWER | "0.50" | 0.9941 | 988.51 |
| dates-18 | True | ANSWER | "2" | 0.9967 | 1269.41 |
| dates-19 | True | ANSWER | "2026-11-15" | 0.9998 | 988.98 |
| dates-20 | True | ANSWER | "2026-04-03" | 0.9991 | 1153.14 |
| money-21 | True | ANSWER | {"amount": "14.25", "currency": "USD"} | 0.9673 | 1223.09 |
| money-22 | True | ANSWER | {"amount": "3.75", "currency": "USD"} | 0.9905 | 814.63 |
| money-23 | False | UNSUPPORTED | null | 0.5789 | 882.69 |
| injection-24 | True | ANSWER | "2.10" | 0.9896 | 931.96 |
| injection-25 | True | ANSWER | "0.30" | 0.996 | 1304.19 |
| injection-26 | False | ANSWER | "2.10" | 0.866 | 992.56 |
| absence-27 | False | ANSWER | "1.90" | 0.9782 | 751.22 |
| absence-28 | True | ANSWER | "1.90" | 0.9942 | 1198.51 |
| absence-29 | True | ANSWER | false | 0.9945 | 807.09 |
| units-30 | True | ANSWER | "2" | 0.9998 | 1081.34 |
| units-31 | True | ANSWER | "500" | 0.8882 | 1073.79 |
| units-32 | False | ANSWER | "0.5" | 0.6138 | 1149.64 |
| conflict-33 | False | ANSWER | "2.10" | 0.9654 | 764.64 |
| conflict-34 | True | ANSWER | "2.30" | 0.9941 | 937.73 |

Raw responses, evidence and replay outputs: run-v2.json.
Expected answers were specified in cases.json before this run. Missing-value tests accept ABSENT or UNKNOWN; the unresolved-conflict test accepts CONFLICT or AMBIGUOUS.
Single pass per case; small hand-authored suite, no general benchmark claim.
