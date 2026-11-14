# Independent comparison audit

2026-11-14. Read-only audit of the saved 512-case comparison; no inference, source fixes, rescoring, or changes to recorded outputs. Source fingerprints and independently counted outcomes are in independent_audit.json.

**Verdict: do not publish the current totals as a clean capability ranking.**

## Findings

1. **Money output contract is incompatible with the scorer.** model_adapters.py:44 permits string, number, boolean, array, or null for `value`, but forbids objects. The scorer requires `{amount, currency}` for all 21 money correction cases. All five general models fail these 21; Reader passes all 21. Example corrections-02: gold is USD 15; Ministral returns 15 and explicitly identifies USD 15 in its reason. A typed strict score is valid only if the model is given a compatible schema. Fix the contract before rerunning; do not simply award currency based on the document.

2. **Percentage serialization is scored as a reasoning failure.** normalize_value handles `number`, but not `pct`. Gemini percentages-01 returns `10%` with the correct calculation; gold is numeric 10. Gemma and LFM have the same issue in sampled cases. Reader returns 120 on that case, a substantive wrong answer. Their shared 0/43 percentage scores do not mean the same thing. Define percentage points versus fractions explicitly, normalize syntax only, and preserve genuine arithmetic failures.

3. **Dates have an undocumented representation requirement.** Gold uses ISO dates; prompts only name type `date`. For example Gemini returns `14 March 2024` for the corresponding ISO gold date. This is not equivalent to the separate inclusive-day counting mistakes visible in Bonsai/Gemma. Specify ISO output and/or use a documented unambiguous date parser.

4. **LFM lists hit an adapter exception, with raw generations discarded.** All 43 list cases contain AttributeError: `'list' object has no attribute 'get'`; the adapter calls `.get` on decoded JSON without validating the root. The retained error raw_response is empty, so these records cannot establish whether the generated lists were correct. Count this as a contract/parsing failure, not transport or demonstrated list reasoning failure. Preserve the raw generation before parsing and rerun the affected cases separately. Fifteen additional LFM errors are JSONDecodeError.

5. **Gemini is quota-limited, not fully measured.** Of 512 records, 124 final errors are HTTP 429, one is HTTP 503, and one is an AttributeError. The 126 `transport_error` entries depress the total. The retry loop waits only 1 then 2 seconds; saved 429 responses request longer delays and name a 15 requests/minute free-tier quota. Pace requests and honor RetryInfo/Retry-After. A completed-request-only score would be descriptive, not a replacement full-suite score, because missing cases are not random.

6. **Error taxonomy and provenance need clearer labels.** JSON/parser/adapter failures are classified as transport_error. Evidence offsets for general models are located or repaired with document.find; this checks quote matching, not native model-produced offset accuracy or whether a quote entails the answer. Report those separately from Reader native evidence and symbolic replay. Provider/generation latency and full client latency must remain separate.

## Checks that passed

- Exactly 512 unique gold IDs and 512 unique result IDs per model.
- Original 240 cases retain all original fields and values.
- Suite SHA-256 matches the manifest: 7aa446c23aa360a783b2ba5935e98e9305d67528ea55a7fea9a8f4cd30db529a.
- The 240 Reader response IDs differ from the historical run; the saved comparison is not a direct copy of those historical responses.
- Independently recounted correct totals: Ministral 455, Gemma 377, Bonsai 366, Gemini 331, Reader 330, LFM 325.

## Next run

Freeze a revised shared output contract and parser tests before inference. Cover money objects, percentage syntax, dates, list-valued answers, malformed roots, and quota errors. Retain every raw response even when parsing fails. Keep this run intact; publish corrected runs with their own identifiers and first-attempt/retry accounting. Do not produce adjusted headline totals from a few hand-inspected examples.

This audit does not validate every new gold answer, checkpoint identity, or model-runtime tuning. The suite remains a small number of parameterized synthetic templates; 512 cases do not imply 512 independent task designs.
