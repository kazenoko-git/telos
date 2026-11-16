#!/usr/bin/env python3
"""Small, reproducible Reader-1 evaluation. No third-party dependencies."""
import argparse
import json
import os
import time
import urllib.error
import urllib.request
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = 'https://platform.xerj.ai'


def key_from_env():
    key = os.environ.get('XERJ_API_KEY', '')
    if not key:
        env = ROOT.parent.parent / '.env'
        if env.exists():
            for line in env.read_text().splitlines():
                name, sep, value = line.partition('=')
                if sep and name.strip() in ('XERJ_API_KEY', 'export XERJ_API_KEY'):
                    key = value.strip().strip('\"\'')
    return key


def request(path, payload, key):
    data = json.dumps(payload, ensure_ascii=False).encode()
    req = urllib.request.Request(BASE + path, data=data, headers={
        'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json',
        'User-Agent': 'telos-reader-evaluation/1.0'})
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=90) as response:
        result = json.load(response)
    return result, round((time.monotonic() - started) * 1000, 2)


def equal(actual, expected):
    if isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(equal(actual.get(k), v) for k, v in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(equal(a, b) for a, b in zip(actual, expected))
    try:
        return Decimal(str(actual)) == Decimal(str(expected))
    except InvalidOperation:
        return actual == expected


def grade(case, response):
    status = response.get('status')
    correct = status in case.get('statuses', ['ANSWER'])
    if 'expected' in case:
        correct = correct and equal(response.get('value'), case['expected'])
    evidence = response.get('evidence', [])
    offsets_ok = all(isinstance(e.get('start'), int) and isinstance(e.get('end'), int)
                     and 0 <= e['start'] < e['end'] <= len(case['document'])
                     and case['document'][e['start']:e['end']] == e.get('quote') for e in evidence)
    # Text matching proves provenance consistency, not that the selected facts are relevant.
    has_evidence = status != 'ANSWER' or bool(evidence)
    return {'answer_correct': correct, 'evidence_offsets_valid': offsets_ok,
            'answer_has_evidence': has_evidence}


def write_report(results, output, cases_file='cases.json'):
    completed = [r for r in results if 'grade' in r]
    tokens = sum(r['response'].get('usage', {}).get('input_tokens', 0) for r in completed)
    counts = Counter(r['category'] for r in completed)
    correct = Counter(r['category'] for r in completed if r['grade']['answer_correct'])
    replayed = [r for r in results if 'replay_matches' in r]
    lines = ['# Reader-1 API evaluation', '',
             'All documents are synthetic. These results do not measure accuracy on real Télos logs.', '',
             f"Answer correctness: {sum(r['grade']['answer_correct'] for r in completed)}/{len(completed)} model responses.",
             f"API or replay errors: {sum('error' in r for r in results)}.",
             f"Exact evidence offsets: {sum(r['grade']['evidence_offsets_valid'] for r in completed)}/{len(completed)} responses (empty evidence passes this check).",
             f"Answers with evidence: {sum(r['grade']['answer_has_evidence'] for r in completed)}/{len(completed)} responses (abstentions exempt).",
             f"Deterministic replay: {sum(r['replay_matches'] for r in replayed)}/{len(replayed)} match in status and value.",
             f'Input tokens: {tokens}. Estimated charge at $0.01/M input tokens: ${tokens / 100000000:.8f}.', '',
             'Evidence offset validity establishes text consistency, not semantic relevance. Replay tests consistency of execution, not correctness of the model-selected program.', '',
             '| Category | Correct / responses |', '|---|---|']
    lines.extend(f'| {category} | {correct[category]} / {total} |' for category, total in counts.items())
    lines.extend(['', '| Case | Correct | Status | Value | Confidence | Client ms |', '|---|---|---|---|---|---|'])
    for record in results:
        response = record.get('response', {})
        value = json.dumps(response.get('value'), ensure_ascii=False).replace('|', '\\|')
        lines.append(f"| {record['case_id']} | {record.get('grade', {}).get('answer_correct', 'error')} | {response.get('status', 'error')} | {value} | {response.get('confidence')} | {record.get('client_ms', '')} |")
    lines.extend(['', 'Raw responses, evidence and replay outputs: ' + output.name + '.',
                  f'Expected answers were specified in {cases_file} before this run. Missing-value tests accept ABSENT or UNKNOWN; the unresolved-conflict test accepts CONFLICT or AMBIGUOUS.',
                  'Single pass per case; small hand-authored suite, no general benchmark claim.', ''])
    output.with_suffix('.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true', help='Send the synthetic cases to XERJ')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--output', default='run.json')
    parser.add_argument('--cases', default='cases.json')
    args = parser.parse_args()
    cases = json.loads((ROOT / args.cases).read_text())
    if args.limit is not None:
        cases = cases[:args.limit]
    assert len({c['id'] for c in cases}) == len(cases)
    assert all(len(c['question']) <= 300 and c['document'] for c in cases)
    print(f'{len(cases)} synthetic cases validated; no private project data included.', flush=True)
    if not args.run:
        return
    key = key_from_env()
    if not key:
        raise SystemExit('Set XERJ_API_KEY in the environment or project .env. No calls made.')
    output = ROOT / args.output
    if output.exists():
        raise SystemExit(f'Refusing to overwrite {output}; choose --output.')
    results = []
    for case in cases:
        payload = {k: case[k] for k in ('document', 'question', 'type', 'ref_date') if k in case}
        payload['model'] = 'xerj-reader-1'
        record = {'case_id': case['id'], 'category': case['category']}
        try:
            response, elapsed = request('/v1/answers', payload, key)
            record.update(response=response, client_ms=elapsed, grade=grade(case, response))
            program = response.get('symbolic', {}).get('program')
            if program is not None:
                replay, replay_ms = request('/v1/execute', dict(payload, program=program), key)
                record.update(replay=replay, replay_client_ms=replay_ms,
                              replay_matches=(replay.get('status') == response.get('status')
                                              and equal(replay.get('value'), response.get('value'))))
            print(case['id'], response.get('status'), repr(response.get('value')),
                  'PASS' if record['grade']['answer_correct'] else 'FAIL', flush=True)
        except urllib.error.HTTPError as error:
            # Only store the provider error body; never request headers or the key.
            record['error'] = {'http_status': error.code, 'body': error.read().decode(errors='replace')[:2000]}
            print(case['id'], 'HTTP', error.code, flush=True)
        except (OSError, ValueError) as error:
            record['error'] = {'kind': type(error).__name__, 'message': str(error)}
            print(case['id'], type(error).__name__, flush=True)
        results.append(record)
        output.write_text(json.dumps({'cases_file': args.cases, 'results': results}, indent=2, ensure_ascii=False) + '\n')
        if record.get('error', {}).get('http_status') in (401, 402, 403, 429, 503):
            print('Stopped on access, balance, rate-limit or availability error; no automatic retry.', flush=True)
            break
        # Answers and execute share the preview request limit.
        time.sleep(2)
    answered = [r for r in results if 'grade' in r]
    print(f"Correct: {sum(r['grade']['answer_correct'] for r in answered)}/{len(answered)} responses; "
          f"errors: {sum('error' in r for r in results)}")
    print('Saved:', output)
    write_report(results, output, args.cases)


if __name__ == '__main__':
    main()
