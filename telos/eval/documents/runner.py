"""Resumable, adapter-neutral document evaluation and offline score replay."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import platform
import time
from urllib.parse import urlsplit, urlunsplit
import uuid

from .contracts import execute, grade_extraction, public_plan
from .prompts import render
from .scoring import evidence_info, outcome, parse_direct, parse_fields
from .suite import load_suite, reference_responses

TRACKS = ('direct_qa', 'extraction', 'calculator')
SCHEMA = 'telos.eval.documents.v1'


def selected_tracks(tracks):
    if tracks == 'all':
        return TRACKS
    if isinstance(tracks, str):
        tracks = (tracks,)
    if not tracks or any(track not in TRACKS for track in tracks):
        raise ValueError(f'Tracks must be drawn from {TRACKS}')
    return tuple(dict.fromkeys(tracks))


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def summary(records, track, expected_count):
    key = 'exact_fields' if track == 'extraction' else 'answer_correct'
    good = sum(bool(record['grade'].get(key)) for record in records)
    categories = {}
    for record in records:
        row = categories.setdefault(record['category'], {'attempted': 0, 'correct': 0})
        row['attempted'] += 1
        row['correct'] += bool(record['grade'].get(key))
    return {'metric': 'exact_source_fields' if track == 'extraction' else 'status_value_correctness',
            'correct': good, 'attempted': len(records), 'expected': expected_count,
            'complete': len(records) == expected_count,
            'saved_success_pct': round(good * 100 / len(records), 4) if records else None,
            'outcomes': dict(Counter(record['grade']['outcome'] for record in records)), 'categories': categories}


def grade_record(case, plan, track, record):
    record = deepcopy(record)
    record.update(case_id=case['id'], category=case['category'], subset=case['subset'], track=track)
    if track == 'direct_qa':
        response = record.get('response')
        if not isinstance(response, dict):
            try:
                response = parse_direct(record.get('raw_completion'), case['type'])
            except (ValueError, TypeError) as error:
                response = {'status': 'MALFORMED', 'value': None, 'error': {'kind': 'malformed_response', 'message': str(error)}}
        record['response'] = response
        label = outcome(case, response)
        record['grade'] = {'answer_correct': label in ('correct_answer', 'correct_abstention'), 'outcome': label}
        record['evidence_check'] = evidence_info(case['document'], response)
    else:
        fields = record.get('fields')
        error = record.get('error')
        if fields is None and not error:
            try:
                fields = parse_fields(record.get('raw_completion'), [f['name'] for f in plan['fields']])
            except (ValueError, TypeError) as exc:
                error = {'kind': 'malformed_response', 'message': str(exc)}
        record['fields'] = fields
        record['error'] = error
        extraction = grade_extraction(case['document'], plan, fields)
        record['extraction_grade'] = extraction
        if track == 'extraction':
            label = 'malformed_response' if error and error.get('kind') == 'malformed_response' else ('rate_limit_error' if error.get('http_status') == 429 else 'runtime_error') if error else 'exact_extraction' if extraction['exact_fields'] else 'incorrect_extraction'
            record['grade'] = {**extraction, 'outcome': label}
        else:
            response = execute(case['document'], public_plan(plan), fields)
            record['response'] = response
            label = 'malformed_response' if error and error.get('kind') == 'malformed_response' else ('rate_limit_error' if error.get('http_status') == 429 else 'runtime_error') if error else outcome(case, response)
            record['grade'] = {'answer_correct': label in ('correct_answer', 'correct_abstention'), 'outcome': label}
    return record


def _safe_model_info(adapter):
    info = {'model': str(adapter.model_name), 'backend': str(adapter.backend_name)}
    for key in ('model_revision', 'model_path'):
        if getattr(adapter, key, None):
            info[key] = str(getattr(adapter, key))
    base = getattr(adapter, 'api_base', None)
    if base:
        parsed = urlsplit(base)
        info['api_base'] = urlunsplit((parsed.scheme, parsed.netloc.split('@')[-1], parsed.path, '', ''))
    return info


def _weights(adapter):
    root = Path(getattr(adapter, 'model_path', '') or str(adapter.model_name))
    if not root.is_dir():
        return {}
    result = {}
    for path in sorted(root.glob('*.safetensors')):
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        result[path.name] = digest.hexdigest()
    return result


def evaluate_documents(adapter, *, tracks='direct_qa', output=None, resume=False, max_tasks=None,
                       max_new_tokens=256, temperature=0.0, prompt_format='instruction', thinking=False,
                       max_consecutive_errors=3, hash_weights=True, case_ids=None):
    """Run a supplied BaseModelAdapter; extraction generation is shared with calculator.

    No reference operand or expected answer enters prompt rendering. No provider
    is selected here. Model/runtime exceptions and raw malformed text are retained.
    """
    from ..display import EvaluationProgress
    tracks = selected_tracks(tracks)
    if max_tasks is not None and (not isinstance(max_tasks, int) or isinstance(max_tasks, bool) or max_tasks < 1):
        raise ValueError('max_tasks must be a positive integer')
    if not isinstance(max_new_tokens, int) or max_new_tokens < 1 or not math.isfinite(temperature) or temperature < 0 or max_consecutive_errors < 1:
        raise ValueError('Invalid generation/error limits')
    if prompt_format not in ('instruction', 'native-template'):
        raise ValueError('Unknown prompt format')
    cases, plans, manifest = load_suite()
    if case_ids is not None:
        if len(case_ids) != len(set(case_ids)) or set(case_ids) - {c['id'] for c in cases} or not case_ids:
            raise ValueError('case_ids must be unique known document task IDs')
        cases = [c for c in cases if c['id'] in set(case_ids)]
    if max_tasks:
        cases = cases[:max_tasks]
    settings = {'tracks': list(tracks), 'case_ids': [c['id'] for c in cases], 'max_new_tokens': max_new_tokens,
                'temperature': temperature, 'prompt_format': prompt_format, 'thinking': thinking,
                'max_consecutive_errors': max_consecutive_errors, 'suite_files': manifest['files']}
    identity = {'model': _safe_model_info(adapter), 'settings': settings,
                'weight_sha256': _weights(adapter) if hash_weights else {}}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    report = {'schema': SCHEMA, 'suite': manifest['suite'], 'suite_version': manifest['version'],
              'manifest': manifest, 'run_id': str(uuid.uuid4()), 'identity': identity, 'fingerprint': fingerprint,
              'started_at': time.time(), 'finished_at': None, 'platform': platform.platform(),
              'attempt_policy': 'One adapter call per prediction; built-in HTTP adapters use one request. No automatic recovery.',
              'display_note': 'Synthetic diagnostics. Tracks are separate; calculator operations are human-specified. Evidence checks establish literal matching, not entailment.',
              'tracks': {track: {'records': []} for track in tracks}}
    if resume:
        if output is None or not Path(output).exists():
            raise ValueError('Resume requires an existing --output file')
        report = json.loads(Path(output).read_text())
        if report.get('schema') != SCHEMA or report.get('fingerprint') != fingerprint:
            raise ValueError('Resume configuration/model/suite fingerprint mismatch')
    elif output and Path(output).exists():
        raise FileExistsError('Output exists; use resume or a different output path')
    # Verify native-template capability before spending requests.
    for track in ('direct_qa', 'extraction'):
        if track in tracks or track == 'extraction' and 'calculator' in tracks:
            render(adapter, cases[0], public_plan(plans[cases[0]['id']]), track, prompt_format, thinking)
    known = {}
    for track, saved in report['tracks'].items():
        ids = [r['case_id'] for r in saved['records']]
        if len(ids) != len(set(ids)) or set(ids) - set(settings['case_ids']):
            raise ValueError('Resume records contain duplicate or unknown task IDs')
        known[track] = {r['case_id']: r for r in saved['records']}
    def save():
        for track in tracks:
            report['tracks'][track]['summary'] = summary(report['tracks'][track]['records'], track, len(cases))
        if output:
            atomic_write(output, report)
    save()
    consecutive = 0
    progress = {track: EvaluationProgress(f'Documents {track}', len(cases)) for track in tracks}
    try:
        for case in cases:
            plan = plans[case['id']]
            for inference_track in ('direct_qa', 'extraction'):
                destinations = [t for t in tracks if t == 'direct_qa'] if inference_track == 'direct_qa' else [t for t in tracks if t != 'direct_qa']
                missing = [t for t in destinations if case['id'] not in known[t]]
                if not missing:
                    continue
                prior = next((known[t][case['id']] for t in destinations if case['id'] in known[t]), None)
                if prior is None:
                    prompt = render(adapter, case, public_plan(plan), inference_track, prompt_format, thinking)
                    start = time.perf_counter()
                    trace = []
                    base = {'prompt': prompt, 'generation_id': str(uuid.uuid4()), 'raw_completion': None, 'error': None}
                    try:
                        base['raw_completion'] = adapter.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature,
                                                                 request_attempts=1, request_trace=trace)
                        if not isinstance(base['raw_completion'], str):
                            raise TypeError('Adapter.generate must return text')
                        consecutive = 0
                    except Exception as exc:
                        message = str(exc)
                        secret = getattr(adapter, 'api_key', None)
                        if secret:
                            message = message.replace(str(secret), '<redacted>')
                        base['error'] = {'kind': type(exc).__name__, 'message': message}
                        if trace and trace[-1].get('http_status', 0) >= 400:
                            base['error']['http_status'] = trace[-1]['http_status']
                        base['response'] = {'status': 'ERROR', 'value': None, 'error': base['error']}
                        consecutive += 1
                    base['client_ms'] = round((time.perf_counter() - start) * 1000, 3)
                    base['request_trace'] = trace
                else:
                    base = prior
                for track in missing:
                    record = grade_record(case, plan, track, base)
                    report['tracks'][track]['records'].append(record)
                    known[track][case['id']] = record
                save()
                for track in missing:
                    saved = report['tracks'][track]
                    progress[track].update(len(saved['records']), passed=saved['summary']['correct'])
                if consecutive >= max_consecutive_errors:
                    raise RuntimeError('Stopped after consecutive backend errors; partial results were saved. Use a new output for recovery.')
        report['finished_at'] = time.time()
        save()
        return report
    finally:
        # Progress has no persistent resource and does not change model state.
        pass


def replay_responses(data, *, tracks=None):
    """Re-grade package reports or generic response records without model imports.

    Input records: {model, case_id, track, response:{status,value,...}} for direct
    QA, or {model, case_id, track:'extraction', fields:{name:[literal,...]}}.
    Calculator results are recomputed from fields, never trusted from input grades.
    """
    cases, plans, manifest = load_suite()
    by_id = {c['id']: c for c in cases}
    groups = {}
    if data.get('schema') == SCHEMA:
        model = data['identity']['model']['model']
        inputs = [{**r, 'model': model, 'track': track} for track, run in data['tracks'].items() for r in run['records']]
        expected = len(data['identity']['settings']['case_ids'])
    else:
        inputs = data.get('records')
        expected = 512
        if not isinstance(inputs, list):
            raise ValueError('Expected a document run or an object containing records')
    requested = selected_tracks(tracks) if tracks is not None else None
    seen = set()
    for item in inputs:
        if item.get('case_id') not in by_id:
            raise ValueError(f"Unknown task ID: {item.get('case_id')}")
        track = item.get('track', 'direct_qa')
        if track not in TRACKS:
            raise ValueError(f'Unknown track: {track}')
        model = str(item.get('model', 'Model'))
        key = (model, track, item['case_id'])
        if key in seen:
            raise ValueError('Duplicate model/track/task record')
        seen.add(key)
        destinations = (track,) if data.get('schema') == SCHEMA or track != 'extraction' else ('extraction', 'calculator')
        for target in destinations:
            if requested is not None and target not in requested:
                continue
            group = groups.setdefault((model, target), [])
            case = by_id[item['case_id']]
            # An imported raw completion is authoritative; discard stored parsed
            # responses/fields so changes to raw text cannot reuse an old grade.
            candidate = deepcopy(item)
            if candidate.get('raw_completion') is not None:
                if target == 'direct_qa':
                    candidate.pop('response', None)
                else:
                    candidate.pop('fields', None)
                    candidate.pop('error', None)
            group.append(grade_record(case, plans[case['id']], target, candidate))
    return {'schema': 'telos.eval.documents.replay.v1', 'suite': 'document512', 'manifest': manifest,
            'display_note': 'Offline saved-score replay; tracks remain separate. No inference or provider program execution.',
            'runs': [{'model': model, 'track': track, 'summary': summary(records, track, expected), 'records': records}
                     for (model, track), records in groups.items()]}


def replay_reference():
    return replay_responses(reference_responses())
