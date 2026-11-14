"""Read-only audit of saved comparison artifacts; no inference or source mutation."""
import collections
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import model_adapters as adapters
import run_comparative_suite as runner
import generate_suite_512 as generator


def load(name):
    raw = (ROOT / name).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def main():
    gold, suite_hash = load('suite_512_cases.json')
    gm = {c['id']: c for c in gold}
    rebuilt = generator.build_suite()[0]
    out = {'suite_sha256': suite_hash, 'regeneration_equal': rebuilt == gold,
           'unique_gold_ids': len(gm), 'models': {}, 'files': {}}
    merged, merged_hash = load('comparative_512_v2_run.json')
    out['files']['comparative_512_v2_run.json'] = merged_hash
    sources = ['comparative_512_v2_local.json', 'comparative_512_v2_xerj.json',
               'comparative_512_qwen08b.json', 'comparative_512_qwen2b.json',
               'comparative_512_v2_gemini.json']
    for name in sources:
        data, digest = load(name)
        out['files'][name] = digest
        for model, run in data['model_runs'].items():
            records = run['cases']
            ids = [c['case_id'] for c in records]
            result = {'records': len(records), 'unique_ids': len(set(ids)),
                      'unknown_ids': sorted(set(ids) - gm.keys()),
                      'missing_ids': sorted(gm.keys() - set(ids)),
                      'correct': sum(c['grade']['answer_correct'] for c in records),
                      'outcomes': dict(collections.Counter(c['grade']['outcome_classification'] for c in records)),
                      'correctness_mismatches': [], 'classification_mismatches': [],
                      'evidence_grade_mismatches': [], 'raw_normalization_mismatches': [],
                      'wrapper_value_failures': [], 'single_field_equivalent_candidates': [],
                      'nonanswer_nonnull': [], 'confidence_outside_0_1': [],
                      'finish_reasons': {}, 'http_errors': {}, 'error_kinds': {}, 'correct_without_answer_evidence': [],
                      'correct_with_invalid_offsets': [], 'category_correct': {},
                      'merged_records': len(merged['model_runs'].get(model, {}).get('cases', []))}
            finishes = collections.Counter()
            for c in records:
                case = gm[c['case_id']]; res = c['response']; grade = c['grade']
                classification = runner.classify_outcome(case, res)
                if classification != grade['outcome_classification']:
                    result['classification_mismatches'].append(c['case_id'])
                if (classification in ('correct_answer', 'correct_abstention')) != grade['answer_correct']:
                    result['correctness_mismatches'].append(c['case_id'])
                offsets, has_evidence = runner.check_evidence(case['document'], res.get('status'), res.get('evidence', []))
                if (offsets, has_evidence) != (grade['evidence_offsets_valid'], grade['answer_has_evidence']):
                    result['evidence_grade_mismatches'].append(c['case_id'])
                if grade['answer_correct'] and not has_evidence:
                    result['correct_without_answer_evidence'].append(c['case_id'])
                if grade['answer_correct'] and not offsets:
                    result['correct_with_invalid_offsets'].append(c['case_id'])
                if res.get('status') not in ('ANSWER', 'ERROR', 'MALFORMED') and res.get('value') is not None:
                    result['nonanswer_nonnull'].append(c['case_id'])
                conf = res.get('confidence')
                if isinstance(conf, (float, int)) and not 0 <= conf <= 1:
                    result['confidence_outside_0_1'].append(c['case_id'])
                raw = res.get('raw_response') or {}
                choice = (raw.get('choices') or [{}])[0]
                if choice: finishes[choice.get('finish_reason', 'missing')] += 1
                txt = choice.get('message', {}).get('content') or raw.get('text')
                if not txt: continue
                try:
                    parsed = json.loads(txt)
                except (ValueError, TypeError): continue
                rawval = parsed.get('value') if isinstance(parsed, dict) else parsed
                if res.get('status') == 'ANSWER':
                    normalized = adapters.normalize_value(case.get('type'), rawval)
                    if normalized != res.get('value'):
                        result['raw_normalization_mismatches'].append(c['case_id'])
                    if not grade['answer_correct'] and case['type'] != 'money' and isinstance(rawval, dict):
                        result['wrapper_value_failures'].append(c['case_id'])
                        if len(rawval) == 1 and 'expected' in case:
                            v = next(iter(rawval.values()))
                            if runner.equal(adapters.normalize_value(case['type'], v), case['expected']):
                                result['single_field_equivalent_candidates'].append(c['case_id'])
            result['finish_reasons'] = dict(finishes)
            result['error_kinds'] = dict(collections.Counter((c['response'].get('error') or {}).get('kind') for c in records if c['response'].get('error')))
            result['http_errors'] = dict(collections.Counter(str((c['response'].get('error') or {}).get('http_status')) for c in records if (c['response'].get('error') or {}).get('http_status')))
            for cat in sorted({c['category'] for c in records}):
                subset = [c for c in records if c['category'] == cat]
                result['category_correct'][cat] = [sum(c['grade']['answer_correct'] for c in subset), len(subset)]
            result['recomputed_summary'] = runner.compute_subset_metrics(records)
            out['models'][model] = result
    html = (ROOT / 'comparative_display.html').read_text()
    match = re.search(r'<script id="report-data" type="application/json">(.*?)</script>', html, re.S)
    if match:
        payload = json.loads(match.group(1))
        out['dashboard_payload_keys'] = list(payload)
        out['dashboard_model_summaries'] = [{k: m.get(k) for k in ('id', 'subsets')} for m in payload.get('models', [])]
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
