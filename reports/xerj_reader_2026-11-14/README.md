# XERJ Reader-1 evaluation

This standalone evaluation uses 34 hand-authored synthetic questions. It does not change Télos training code or send private project documents. Expected answers were fixed before API execution.

## Run

Set `XERJ_API_KEY` in the environment or the existing project `.env`. The runner reads only that key and never prints it.

```sh
python3 reports/xerj_reader_2026-11-14/evaluate.py
python3 reports/xerj_reader_2026-11-14/evaluate.py --run --output another-run.json
```

The first command validates the dataset without API calls. The second calls `/v1/answers` once per case and replays the resulting formula through `/v1/execute`. Outputs are written beside the script. Existing result files are never overwritten. The runner stops on authentication, balance, rate-limit or availability errors. It uses two seconds of spacing between cases and no automatic retries.

## Interpretation

Answer correctness compares status and typed value against `cases.json`, allowing numerically equivalent decimal representations. Missing-value cases accept ABSENT or UNKNOWN. The unresolved-conflict case accepts CONFLICT or AMBIGUOUS. Correct abstention is scored separately from returned numeric answers by category in the report.

Every evidence quote is checked against Python Unicode code-point offsets in the exact submitted document. This checks textual consistency, not semantic relevance. Deterministic replay checks status and value equality; a wrong formula can still replay perfectly.

Raw JSON retains responses, evidence, programs, latency, confidence and usage. The adjacent Markdown report summarizes scores and list-price cost. Client latency includes network and request time; provider latency is kept separately in each raw response.

The first attempt (`run.json`) was rejected by Cloudflare with error 1010 before a model response. An explicit evaluation User-Agent was added, and `run-v2.json` records the subsequent run.

## Sources

- https://platform.xerj.ai/docs/api-answers
- https://platform.xerj.ai/docs/api-execute
- https://platform.xerj.ai/docs/concepts

These are small synthetic checks, not a representative benchmark or an estimate of production accuracy. No feedback is sent to XERJ by this runner beyond the API test requests.
