"""Display contracts: preserve outcomes, avoid duplicate scores, escape data."""
import io
import json

import pytest

from telos.eval.display import EvaluationProgress, display_mode, normalize_report, write_html_report


def native():
    return {'suite': 'humaneval', 'total_tasks': 2, 'pass_at_1_pct': 50,
            'tasks': [{'id': 'one', 'outcome': 'PASSED', 'category': 'Code'},
                      {'id': 'two', 'outcome': 'TIMEOUT', 'category': 'Code'}]}


def test_native_aliases_are_not_double_counted():
    suite = native()
    data = {'model_checkpoint': 'example/model', 'code': {'functional': suite}, 'functional': suite}
    report = normalize_report(data)
    assert len(report['rows']) == 2
    assert len(report['suites']) == 1
    assert [r['correct'] for r in report['rows']] == [True, False]
    assert {r['model'] for r in report['rows']} == {'example/model'}


def test_dual_tracks_do_not_include_aggregate_alias():
    suite = dict(native(), no_hint=native(), hint=native())
    report = normalize_report(suite)
    assert len(report['suites']) == 2
    assert len(report['rows']) == 4
    assert len({r['suite'] for r in report['rows']}) == 2


def test_probe_top1_is_not_misrepresented_as_functional_execution():
    data = {'overall': {'top1_acc_pct': 50, 'total_probes': 2},
            'probe_type': 'causal', 'probes': [{'top1': True, 'target': 'x'}, {'top1': False, 'target': 'y'}]}
    result = normalize_report(data)
    assert len(result['rows']) == 2
    assert [r['correct'] for r in result['rows']] == [True, False]
    assert result['suites'][0]['suite'] == 'causal'


def test_reader_errors_and_missing_grades_remain_unscored():
    data = {'results': [{'case_id': 'x', 'error': {'http_status': 403}},
                        {'case_id': 'y', 'response': {'status': 'ANSWER', 'value': '5'}}]}
    report = normalize_report(data)
    assert all(r['correct'] is None for r in report['rows'])
    assert report['rows'][0]['status'] == 'ERROR'
    assert all(r['confidence'] is None for r in report['rows'])


def test_reader_cases_supply_document_and_expected_without_changing_grade():
    data = {'results': [{'case_id': 'x', 'grade': {'answer_correct': False},
                        'response': {'status': 'ANSWER', 'value': '4'}}]}
    report = normalize_report(data, cases=[{'id': 'x', 'document': '🧪 5', 'expected': '5'}])
    assert report['rows'][0]['document'] == '🧪 5'
    assert report['rows'][0]['expected'] == '5'
    assert report['rows'][0]['correct'] is False


def test_reader_runs_are_separate_suites():
    one = {'results': [{'case_id': 'x', 'response': {'status': 'ANSWER'}}]}
    report = normalize_report({'primary': one, 'diagnostic': one})
    assert {r['suite'] for r in report['rows']} == {'primary', 'diagnostic'}


def test_summary_only_preserves_missing_individual_records():
    report = normalize_report({'suite': 'arc', 'total_tasks': 10, 'pass_at_1_pct': 60})
    assert report['rows'] == []
    assert report['suites'][0]['score'] == 60


def test_audited_archive_preserves_identity_and_score_label():
    data = {'tasks_per_model': 10, 'rows': [{'name': 'Variant unverified', 'suite': 'arc',
            'saved_success_pct': 60, 'passed': 6, 'total': 10}]}
    report = normalize_report(data)
    assert report['suites'][0]['metric'] == 'saved_success_pct'
    assert report['suites'][0]['model'] == 'Variant unverified'
    assert report['rows'] == []


@pytest.mark.parametrize('data', [[], {}, {'pass_at_1_pct': None}])
def test_unsupported_report_has_clear_error(data):
    with pytest.raises(ValueError):
        normalize_report(data)


def test_html_keeps_hostile_text_inside_json_and_escapes_title(tmp_path):
    hostile = '</script><script>alert(1)</script>{{LOGO}}'
    data = {'results': [{'case_id': hostile, 'response': {'question': hostile, 'status': 'ABSENT'}}]}
    out = write_html_report(data, tmp_path / 'report.html', title='<img onerror=bad>')
    content = out.read_text()
    assert '<title>&lt;img onerror=bad&gt;' in content
    start = content.index('<script id="report-data" type="application/json">') + len('<script id="report-data" type="application/json">')
    end = content.index('</script>', start)
    payload = content[start:end]
    assert '</script>' not in payload
    assert json.loads(payload)['rows'][0]['question'] == hostile
    assert '<script src=' not in content
    assert '<link href=' not in content


class TTY(io.StringIO):
    def isatty(self):
        return True


def test_plain_progress_has_no_escapes_and_finishes():
    out = TTY()
    with display_mode('plain'):
        progress = EvaluationProgress('test\x1b[31m\n', 2, stream=out)
        progress.update(0, 0)
        progress.update(1, 1)
        progress.update(2, 1)
    assert '\x1b' not in out.getvalue()
    assert '2/2' in out.getvalue()
    assert '1 passed' in out.getvalue()


def test_disabled_progress_and_context_reset():
    out = TTY()
    with display_mode('off'):
        EvaluationProgress('test', 1, stream=out).update(1, 1)
    assert out.getvalue() == ''
    EvaluationProgress('test', 1, stream=out).update(1, 1)
    assert '1/1' in out.getvalue()


def test_no_color_disables_color_without_disabling_tty_progress(monkeypatch):
    monkeypatch.setenv('NO_COLOR', '')
    out = TTY()
    progress = EvaluationProgress('test', 1, stream=out)
    assert progress.tty and not progress.color
    progress.update(1, 1)
    assert '\x1b[38;' not in out.getvalue()
