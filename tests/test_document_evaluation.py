"""Public contract, checkpoint, packaged references and provider-boundary checks."""
import io
import json
import urllib.error
from unittest.mock import patch

import pytest

from telos.eval.documents.contracts import execute, public_plan
from telos.eval.documents.runner import evaluate_documents, replay_reference, replay_responses
from telos.eval.documents.scoring import normalize_value
from telos.eval.documents.suite import load_suite, reference_responses
from telos.eval.display import normalize_report


class Model:
    model_name = 'test-model'
    backend_name = 'test'
    def __init__(self, fail_at=None):
        self.calls = []
        self.fail_at = fail_at
    def generate(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        if len(self.calls) == self.fail_at:
            raise RuntimeError('synthetic backend failure')
        first = 'east completed 14 tasks' in prompt
        if 'Answer the user question' in prompt:
            return json.dumps({'status': 'ANSWER', 'value': 19 if first else 9})
        return json.dumps({'east': ['14'], 'west': ['5']} if first else {'initial': ['15'], 'removed': ['6']})


def test_resources_and_reference_scores_are_offline():
    cases, plans, manifest = load_suite()
    assert len(cases) == len(plans) == 512
    assert manifest['files']['cases.json'] == '7aa446c23aa360a783b2ba5935e98e9305d67528ea55a7fea9a8f4cd30db529a'
    assert len(reference_responses()['records']) == 6144
    with patch('urllib.request.urlopen', side_effect=AssertionError('Replay made a network request')):
        result = replay_reference()
    values = {(r['model'], r['track']): r['summary']['correct'] for r in result['runs']}
    assert len(values) == 16
    assert values[('Reader-1', 'direct_qa')] == 330
    assert values[('NuExtract3', 'direct_qa')] == 466
    assert values[('NuExtract3', 'extraction')] == 426
    assert values[('NuExtract3', 'calculator')] == 443
    assert values[('GLiNER2 Base', 'extraction')] == 214
    assert values[('NuExtract Tiny', 'calculator')] == 89
    assert values[('SmolLM2 360M', 'direct_qa')] == 1
    assert values[('Qwen 3.5 0.8B', 'direct_qa')] == 316
    assert all(r['summary']['expected'] == 512 for r in result['runs'])


def test_all_tracks_share_extraction_but_keep_distinct_metrics(tmp_path):
    model = Model()
    run = evaluate_documents(model, tracks='all', max_tasks=2, output=tmp_path/'run.json')
    assert len(model.calls) == 4
    assert {t: x['summary']['correct'] for t, x in run['tracks'].items()} == {'direct_qa': 2, 'extraction': 2, 'calculator': 2}
    assert run['tracks']['extraction']['records'][0]['generation_id'] == run['tracks']['calculator']['records'][0]['generation_id']
    assert not any('reference_fields' in prompt or '"expected"' in prompt for prompt, _ in model.calls)
    replay = replay_responses(run)
    assert [r['summary']['correct'] for r in replay['runs']] == [2, 2, 2]
    display = normalize_report(run)
    assert len(display['suites']) == 3 and len(display['rows']) == 6
    assert len({r['suite'] for r in display['rows']}) == 3


def test_resume_preserves_failures_and_refuses_setting_changes(tmp_path):
    path = tmp_path/'run.json'
    with pytest.raises(RuntimeError, match='Stopped after'):
        evaluate_documents(Model(fail_at=3), tracks='all', max_tasks=2, output=path, max_consecutive_errors=1)
    saved = json.loads(path.read_text())
    original = saved['tracks']['direct_qa']['records'][1]
    retry = Model()
    run = evaluate_documents(retry, tracks='all', max_tasks=2, output=path, resume=True, max_consecutive_errors=1)
    assert len(retry.calls) == 1
    assert run['tracks']['direct_qa']['records'][1] == original
    assert run['tracks']['direct_qa']['summary']['correct'] == 1
    with pytest.raises(ValueError, match='fingerprint'):
        evaluate_documents(Model(), tracks='all', max_tasks=2, output=path, resume=True, max_consecutive_errors=1, temperature=0.2)
    with pytest.raises(FileExistsError):
        evaluate_documents(Model(), max_tasks=2, output=path)


def test_replay_ignores_stale_grades_and_parsed_values():
    run = evaluate_documents(Model(), tracks='all', max_tasks=1)
    record = run['tracks']['direct_qa']['records'][0]
    record['raw_completion'] = '{"status":"ANSWER","value":999}'
    record['grade']['answer_correct'] = True
    run['tracks']['extraction']['records'][0]['raw_completion'] = '{"east":["999"],"west":["5"]}'
    result = replay_responses(run)
    scores = {r['track']: r['summary']['correct'] for r in result['runs']}
    assert scores['direct_qa'] == scores['extraction'] == 0
    with pytest.raises(ValueError, match='Duplicate'):
        replay_responses({'records': [{'case_id': 'arithmetic-01', 'response': {}}]*2})
    with pytest.raises(ValueError, match='Unknown task'):
        replay_responses({'records': [{'case_id': 'invented', 'response': {}}]})


def test_currency_is_not_invented_and_extraction_is_not_computed():
    assert normalize_value('money', 'CAD 15') == {'amount': '15', 'currency': 'CAD'}
    assert normalize_value('money', 15)['currency'] is None
    assert normalize_value('pct', '10%') == '10'
    assert normalize_value('date', '14 March 2024') == '2024-03-14'
    cases, plans, _ = load_suite()
    case = cases[0]
    result = execute(case['document'], public_plan(plans[case['id']]), {'east': ['999'], 'west': ['5']})
    assert result['status'] == 'CONTRACT_ERROR'


def test_native_prompt_unsupported_before_generation_and_subset_validation():
    model = Model()
    with pytest.raises(ValueError, match='native-template'):
        evaluate_documents(model, prompt_format='native-template')
    assert model.calls == []
    with pytest.raises(ValueError, match='case_ids'):
        evaluate_documents(model, case_ids=['not-a-case'])
    assert model.calls == []


def test_partial_coverage_is_not_padded_with_failures():
    result = replay_responses({'records': [{'case_id': 'arithmetic-01', 'response': {'status': 'ANSWER', 'value': 19}}]})
    score = result['runs'][0]['summary']
    assert score['attempted'] == score['correct'] == 1
    assert score['expected'] == 512 and not score['complete']


def test_malformed_output_is_kept_and_not_retried():
    model = Model()
    model.generate = lambda prompt, **kw: 'not JSON'
    result = evaluate_documents(model, max_tasks=1)
    record = result['tracks']['direct_qa']['records'][0]
    assert record['raw_completion'] == 'not JSON'
    assert record['grade']['outcome'] == 'malformed_response'


def test_http_attempt_is_observable_and_not_hidden_by_retry():
    from telos.eval.adapters.openai_api import OpenAIAPIAdapter
    model = OpenAIAPIAdapter('test', api_base='http://localhost:9999/v1', api_key='not-a-real-secret')
    def fail(*args, **kwargs):
        raise urllib.error.HTTPError('http://localhost:9999/v1/chat/completions', 429, 'quota', {}, io.BytesIO(b'quota exhausted'))
    with patch('urllib.request.urlopen', side_effect=fail) as request:
        result = evaluate_documents(model, max_tasks=1)
    assert request.call_count == 1
    record = result['tracks']['direct_qa']['records'][0]
    assert record['request_trace'][0]['http_status'] == 429
    assert record['grade']['outcome'] == 'rate_limit_error'


def test_master_entrypoint_routes_multiple_document_models(monkeypatch, tmp_path):
    from telos.eval import runner
    monkeypatch.setattr(runner, 'load_adapter', lambda *a, **k: Model())
    report = runner.evaluate(['a', 'b'], suite='document512', max_tasks=1,
                             output_path=tmp_path/'models.json', track='all')
    assert report['schema'] == 'telos.eval.documents.collection.v1'
    assert len(report['runs']) == 2
    assert all(len(r['tracks']) == 3 for r in report['runs'])


def test_explicit_api_backend_does_not_route_gemma_to_gemini(monkeypatch):
    from telos.eval.adapters import registry
    monkeypatch.setattr(registry, 'OpenAIAPIAdapter', lambda **kw: kw)
    result = registry.load_adapter('gemma-4-local', backend='openai_api', api_base='http://localhost:1234/v1')
    assert result['model_name'] == 'gemma-4-local'
