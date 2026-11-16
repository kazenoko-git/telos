"""Convert document tracks into the existing cream report format, without pooling."""
from .suite import load_suite


def normalize(data, title, source):
    cases, plans, _ = load_suite()
    case_map = {c['id']: c for c in cases}
    if data['schema'] == 'telos.eval.documents.v1':
        model = data['identity']['model']['model']
        runs = [{'model': model, 'track': t, **run} for t, run in data['tracks'].items()]
    elif data['schema'] == 'telos.eval.documents.collection.v1':
        runs = []
        for report in data['runs']:
            model = report['identity']['model']['model']
            runs.extend({'model': model, 'track': t, **run} for t, run in report['tracks'].items())
    else:
        runs = data['runs']
    rows, suites = [], []
    for run in runs:
        score = run['summary']
        label = run['track']
        suites.append({'model': run['model'], 'suite': label, 'score': score['saved_success_pct'],
                       'metric': score['metric'], 'total': score['attempted'],
                       'outcomes': score['outcomes'], 'expected': score['expected']})
        for record in run['records']:
            case = case_map[record['case_id']]
            response = record.get('response') or {}
            grade = record['grade']
            rows.append({'id': record['case_id'], 'model': run['model'], 'suite': label, 'category': case['category'],
                         'correct': grade.get('exact_fields') if label == 'extraction' else grade.get('answer_correct'),
                         'status': response.get('status', grade['outcome']), 'question': case['question'],
                         'value': record.get('fields') if label == 'extraction' else response.get('value'),
                         'expected': plans[record['case_id']]['reference_fields'] if label == 'extraction' else case.get('expected', case.get('statuses')), 'document': case['document'],
                         'confidence': None, 'latency_ms': record.get('client_ms'),
                         'evidence': record.get('evidence_check', {}).get('spans', []),
                         'details': record.get('error') or response.get('reason'),
                         'program': '', 'replay': None, 'offsets': None, 'input_tokens': None})
    return {'schema': 'telos.evaluation.display.v1', 'title': title, 'source': source, 'rows': rows, 'suites': suites,
            'note': 'Synthetic document diagnostics · tracks scored separately.'}
