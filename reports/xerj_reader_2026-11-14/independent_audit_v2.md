# Independent v2 recheck

2026-11-14. Read-only checks of repaired code and saved v2 results. No new inference or changes to benchmark records.

## Verified

- Recomputed status/value correctness for every saved record using the current classifier: no correctness mismatches across six models.
- Full-suite correct counts: Ministral 466/512; Gemma 431/512; LFM 430/512; Bonsai 412/512; Reader 330/512.
- Object values are now allowed by the local schema. Money, percentage, and date normalization exist. All local models now pass 43/43 corrections; LFM passes 43/43 lists.
- No passed local money cases were found with a missing currency in the parsed raw value.
- Archived v1 file SHA-256 matches the source fingerprint from the first independent audit.

## Remaining findings

1. Gemini is incomplete and the report describes its denominator incorrectly. Its 132 records comprise 107 non-error responses and 25 HTTP 429 errors. Correct is 106/132 attempted records, not 106/132 completed answers. Only arithmetic, conflicts, corrections, and three date cases were attempted. Keep it out of the full-suite ranking; do not compare a selected 107-response success rate to 512-case scores.
2. Sixteen Gemini quota errors retain the old transport_error classification. Current classifier changes these to rate_limit_error. This does not change correctness totals, but contradicts the claimed clean separation in the report. Saved summary totals need regeneration from recorded responses.
3. normalize_money still invents USD for bare numbers, missing currency, and unrecognized currency strings. Executed probes: 15, "15", "CAD 15", and {"amount":"15"} all normalize to USD 15. No affected passed local money cases were found in this run; this is a remaining contract defect, not a demonstrated score inflation here. Require explicit currency or return a contract error.
4. comparative_findings.md claims Gemini produces native character offsets. Gemini still passes through normalize_evidence, which finds/repairs offsets with document.find. This establishes quote matching, not native model pointer accuracy or semantic evidence support.
5. scripts/generate_reader_comparative_figure.py still hardcodes v1 diagnostic callouts: Ministral 455/512 and 88.9%, LFM 63.5% and 555ms, and old Gemini findings. These must be derived from the selected run; regenerating the figure currently preserves stale text.
6. Gemini retries occur inside its adapter as well as the outer runner. Inner attempts are not returned as individual attempt records, so the saved first_attempt/retries fields do not represent every HTTP attempt. Preserve those if claiming complete first-attempt accounting.

The repaired local results are substantially more credible. The above reporting and contract issues remain before publication. This recheck does not certify every gold answer, exact checkpoint identity, or visual rendering of the generated figure/dashboard.
