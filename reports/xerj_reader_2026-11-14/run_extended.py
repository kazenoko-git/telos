#!/usr/bin/env python3
"""Rate-limited concurrent diagnostics with immutable case fingerprints."""
import argparse
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
import time
import urllib.error

from evaluate import ROOT, grade, key_from_env, request, write_report


class RateLimiter:
    def __init__(self, interval=1.15):
        self.lock = threading.Lock()
        self.next_start = 0.0
        self.interval = interval

    def wait(self, stopped):
        with self.lock:
            start = max(time.monotonic(), self.next_start)
            self.next_start = start + self.interval
        if stopped.wait(max(0, start - time.monotonic())):
            raise CancelledError()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', default='extended_cases.json')
    parser.add_argument('--output', default='extended-run.json')
    parser.add_argument('--workers', type=int, choices=range(1, 5), default=3)
    parser.add_argument('--replay-every', type=int, default=4)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.replay_every < 1:
        parser.error('--replay-every must be positive')
    output = ROOT / args.output
    if output.exists():
        parser.error('Output already exists; choose another --output.')
    key = key_from_env()
    if not key:
        parser.error('XERJ_API_KEY is missing; no calls made.')
    raw = (ROOT / args.cases).read_bytes()
    cases = json.loads(raw)
    if args.limit is not None:
        cases = cases[:args.limit]
    assert len({c['id'] for c in cases}) == len(cases)
    assert all(len(c['question']) <= 300 and c['document'] for c in cases)
    limiter, stopped = RateLimiter(), threading.Event()
    metadata = dict(cases_file=args.cases, cases_sha256=hashlib.sha256(raw).hexdigest(),
                    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    started_utc=datetime.now(timezone.utc).isoformat(), requested_cases=len(cases),
                    policy=dict(workers=args.workers, start_interval_seconds=limiter.interval,
                                replay_every=args.replay_every, attempts_per_case=1))

    def one(index, case):
        if stopped.is_set():
            raise CancelledError()
        payload = {k: case[k] for k in ('document', 'question', 'type', 'ref_date') if k in case}
        payload['model'] = 'xerj-reader-1'
        record = dict(case_id=case['id'], category=case['category'], index=index)
        try:
            limiter.wait(stopped)
            response, elapsed = request('/v1/answers', payload, key)
            record.update(response=response, client_ms=elapsed, grade=grade(case, response))
            if index % args.replay_every == 0:
                program = response.get('symbolic', {}).get('program')
                if program is not None:
                    limiter.wait(stopped)
                    replay, elapsed = request('/v1/execute', dict(payload, program=program), key)
                    from evaluate import equal
                    record.update(replay=replay, replay_client_ms=elapsed,
                                  replay_matches=(replay.get('status') == response.get('status')
                                                  and equal(replay.get('value'), response.get('value'))))
        except urllib.error.HTTPError as error:
            record['error'] = dict(http_status=error.code, body=error.read().decode(errors='replace')[:2000])
            if error.code in (401, 402, 403, 429, 503):
                stopped.set()
        except CancelledError:
            if 'response' not in record:
                raise
        except (OSError, ValueError) as error:
            record['error'] = dict(kind=type(error).__name__, message=str(error))
        return record

    records = []
    print(f'Running {len(cases)} synthetic cases; starts spaced {limiter.interval}s apart; replay every {args.replay_every}.', flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(one, i, case) for i, case in enumerate(cases)]
        for future in as_completed(futures):
            try:
                record = future.result()
            except CancelledError:
                continue
            records.append(record)
            ordered = sorted(records, key=lambda r: r['index'])
            snapshot = dict(metadata, results=ordered, completed_cases=len(ordered))
            temp = output.with_suffix('.tmp')
            temp.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + '\n')
            temp.replace(output)
            if len(records) % 10 == 0 or len(records) == len(cases) or 'error' in record:
                scored = [r for r in records if 'grade' in r]
                correct = sum(r['grade']['answer_correct'] for r in scored)
                print(f"{len(records)}/{len(cases)} completed · {correct}/{len(scored)} correct · latest {record['case_id']}", flush=True)
    snapshot = dict(metadata, results=sorted(records, key=lambda r: r['index']),
                    completed_cases=len(records), finished_utc=datetime.now(timezone.utc).isoformat())
    output.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + '\n')
    write_report(snapshot['results'], output, args.cases)
    print('Saved:', output, flush=True)
    if stopped.is_set():
        raise SystemExit('Stopped on access, balance, rate limit or availability failure. Output is partial.')


if __name__ == '__main__':
    main()
