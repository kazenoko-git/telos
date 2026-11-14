"""Portable evaluation reports and terminal progress; no web dependencies."""
from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
from contextvars import ContextVar
import html
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time

ASSETS = Path(__file__).with_name('display_assets')
ANSI = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')
DISPLAY_MODE = ContextVar('telos_evaluation_display', default='auto')


@contextmanager
def display_mode(mode):
    if mode not in ('auto', 'plain', 'off'):
        raise ValueError('Display mode must be auto, plain or off.')
    token = DISPLAY_MODE.set(mode)
    try:
        yield
    finally:
        DISPLAY_MODE.reset(token)


def number(value):
    """Reject bools and non-finite numbers in presentation metrics."""
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError, OverflowError):
        return None


def normalize_report(data, *, title='Evaluation report', cases=None, source=''):
    """Read Reader-1 result archives and nested native Télos suite reports.

    Missing results stay unknown. Duplicate native code/functional aliases are
    skipped, and dual hint tracks remain separate instead of being pooled.
    """
    if not isinstance(data, dict):
        raise ValueError('Report must be a JSON object.')
    case_map = {str(c['id']): c for c in (cases or [])}
    rows, suites = [], []
    reader_found = False

    def reader(node, path):
        nonlocal reader_found
        reader_found = True
        for item in node['results']:
            response = item.get('response', {})
            case = case_map.get(str(item.get('case_id')), {})
            grade = item.get('grade', {})
            correct = grade.get('answer_correct')
            rows.append({
                'id': str(item.get('case_id', len(rows) + 1)),
                'model': response.get('model', 'Reader-1'),
                'suite': path or 'Document extraction', 'category': item.get('category', 'General'),
                'question': response.get('question', case.get('question', '')),
                'correct': correct if isinstance(correct, bool) else None,
                'status': response.get('status', 'ERROR' if item.get('error') else 'UNKNOWN'),
                'value': response.get('value'), 'expected': case.get('expected', case.get('statuses')),
                'confidence': number(response.get('confidence')),
                'latency_ms': number(item.get('client_ms')),
                'evidence': response.get('evidence', []), 'document': case.get('document', ''),
                'program': response.get('symbolic', {}).get('program_compact', ''),
                'replay': item.get('replay_matches'),
                'offsets': grade.get('evidence_offsets_valid'),
                'details': item.get('error', response.get('reason')),
                'input_tokens': number(response.get('usage', {}).get('input_tokens')),
            })

    def visit(node, path='', model='Model'):
        if not isinstance(node, dict):
            return
        model = str(node.get('model_checkpoint', node.get('model', model)))
        if isinstance(node.get('rows'), list) and 'tasks_per_model' in node:
            for saved in node['rows']:
                suites.append({'model': saved.get('name', saved.get('model', model)),
                               'suite': saved.get('label', saved.get('suite', path)),
                               'score': number(saved.get('saved_success_pct')),
                               'metric': 'saved_success_pct', 'total': number(saved.get('total')),
                               'ci': None, 'outcomes': {}})
            return
        if isinstance(node.get('results'), list) and any('case_id' in r for r in node['results'] if isinstance(r, dict)):
            reader(node, path)
            return
        if isinstance(node.get('no_hint'), dict) and isinstance(node.get('hint'), dict):
            visit(node['no_hint'], path + '/no_hint', model)
            visit(node['hint'], path + '/hint', model)
            return
        if isinstance(node.get('overall'), dict) and 'total_probes' in node['overall']:
            overall = node['overall']
            if number(overall.get('top1_acc_pct')) is not None:
                converted = dict(node, suite=path or node.get('probe_type', 'Probes'),
                                 total_tasks=overall['total_probes'],
                                 pass_at_1_pct=overall['top1_acc_pct'],
                                 pass_at_1_95ci=overall.get('top1_95ci'),
                                 tasks=[dict(p, id=p.get('id', i + 1), passed=p.get('top1'),
                                             details={'target': p.get('target'), 'rank': p.get('rank'), 'cross_entropy': p.get('target_ce')})
                                        for i, p in enumerate(node.get('probes', []))])
                converted.pop('overall')
                visit(converted, path, model)
            return
        score_key = next((k for k in ('pass_at_1_pct', 'pass_rate_pct') if number(node.get(k)) is not None), None)
        if score_key is not None and 'total_tasks' in node:
            suite = str(node.get('suite', path or 'Evaluation'))
            track = path.rsplit('/', 1)[-1]
            if track in ('hint', 'no_hint'):
                suite += ' · ' + track.replace('_', ' ')
            tasks = node.get('tasks', [])
            suites.append({'model': model, 'suite': suite,
                           'score': number(node[score_key]), 'metric': score_key,
                           'total': number(node.get('total_tasks')),
                           'ci': node.get('pass_at_1_95ci'),
                           'outcomes': node.get('execution_outcomes', {})})
            for task in tasks:
                status = task.get('outcome', 'PASSED' if task.get('passed') is True else 'FAILED' if task.get('passed') is False else 'UNKNOWN')
                known = 'outcome' in task or isinstance(task.get('passed'), bool)
                rows.append({'id': str(task.get('id', len(rows) + 1)), 'model': model,
                             'suite': suite, 'category': str(task.get('category', 'General')),
                             'correct': status == 'PASSED' if known else None,
                             'status': status, 'question': task.get('prompt', ''),
                             'value': task.get('completion'), 'expected': None,
                             'details': task.get('details'), 'confidence': None,
                             'latency_ms': None, 'evidence': [], 'document': '',
                             'program': '', 'replay': None, 'offsets': None, 'input_tokens': None})
            return
        for key, value in node.items():
            if key in ('functional', 'probes', 'anticheat', 'sample') and isinstance(node.get('code'), dict) and key in node['code']:
                continue
            if isinstance(value, dict):
                child_model = key if not path and not node.get('model_checkpoint') else model
                visit(value, f'{path}/{key}'.strip('/'), child_model)

    visit(data)
    if not rows and not suites:
        raise ValueError('No supported evaluation results found. Expected native suite reports or Reader-1 results.')
    default_note = ('Synthetic diagnostics; not production accuracy.'
                    if reader_found else 'Saved results; datasets and retry policies may differ.')
    return {'schema': 'telos.evaluation.display.v1', 'title': str(title), 'source': source,
            'rows': rows, 'suites': suites,
            'note': str(data.get('display_note', default_note))}


def write_html_report(data, output, *, title='Evaluation report', cases=None, source=''):
    report = normalize_report(data, title=title, cases=cases, source=source)
    # A JSON script element is still an HTML parsing context: block </script>.
    payload = json.dumps(report, ensure_ascii=True, allow_nan=False).replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e')
    content = (ASSETS / 'report.html').read_text(encoding='utf-8')
    logo = base64.b64encode((ASSETS / 'logo.png').read_bytes()).decode('ascii')
    replacements = {'TITLE': html.escape(str(title)), 'STYLE': (ASSETS / 'report.css').read_text(),
                       'SCRIPT': (ASSETS / 'report.js').read_text(), 'DATA': payload,
                       'LOGO': logo}
    content = re.sub(r'\{\{(TITLE|STYLE|SCRIPT|DATA|LOGO)\}\}', lambda m: replacements[m[1]], content)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding='utf-8')
    return output


def clean_terminal(text):
    text = ANSI.sub('', str(text))
    return ''.join(c if c >= ' ' and c != '\x7f' else ' ' for c in text)


class EvaluationProgress:
    """One-line TTY progress; sparse, escape-free records in pipes and CI."""
    def __init__(self, label, total, *, stream=None, enabled=True):
        self.stream = stream or sys.stderr
        self.enabled = enabled and DISPLAY_MODE.get() != 'off'
        self.label = clean_terminal(label)
        self.total = max(0, int(total))
        self.started = time.monotonic()
        self.last = -1
        self.tty = DISPLAY_MODE.get() != 'plain' and bool(getattr(self.stream, 'isatty', lambda: False)())
        self.color = self.tty and 'NO_COLOR' not in os.environ and os.environ.get('TERM') != 'dumb'

    def update(self, completed, passed, *, detail='', force=False):
        if not self.enabled:
            return
        completed = max(0, min(self.total, int(completed)))
        if not self.tty and not force and completed not in (0, self.total) and completed % 10:
            return
        if completed == self.last and not force:
            return
        self.last = completed
        width = max(20, shutil.get_terminal_size((80, 24)).columns)
        elapsed = time.monotonic() - self.started
        rate = completed / elapsed if elapsed else 0
        eta = (self.total - completed) / rate if rate else 0
        bar_width = min(20, max(5, width // 7))
        filled = round(bar_width * completed / max(self.total, 1))
        bar = '=' * filled + '.' * (bar_width - filled)
        text = f'TÉLOS · {self.label} [{bar}] {completed}/{self.total} · {passed} passed · {elapsed:.0f}s'
        if completed and completed < self.total:
            text += f' · ETA {eta:.0f}s'
        if detail:
            text += ' · ' + clean_terminal(detail)
        if self.tty:
            text = text[:width - 1]
            if self.color:
                text = '\x1b[38;2;155;91;62m' + text + '\x1b[0m'
            self.stream.write('\r\x1b[2K' + text + ('\n' if completed == self.total else ''))
        else:
            self.stream.write(text + '\n')
        self.stream.flush()


def main(argv=None):
    parser = argparse.ArgumentParser(description='Build a portable Télos evaluation display from saved JSON.')
    parser.add_argument('reports', nargs='+', type=Path)
    parser.add_argument('--output', '-o', type=Path, default=Path('evaluation.html'))
    parser.add_argument('--title', default='Evaluation report')
    parser.add_argument('--cases', type=Path, help='Optional Reader-1 cases JSON for expected answers and source documents')
    args = parser.parse_args(argv)
    try:
        data = {p.stem: json.loads(p.read_text()) for p in args.reports}
        if len(data) != len(args.reports):
            raise ValueError('Input filenames must have distinct stems.')
        cases = json.loads(args.cases.read_text()) if args.cases else None
        path = write_html_report(data, args.output, title=args.title, cases=cases,
                                 source=', '.join(p.name for p in args.reports))
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    print(f'TÉLOS · Saved evaluation display to {path}')


if __name__ == '__main__':
    main()
