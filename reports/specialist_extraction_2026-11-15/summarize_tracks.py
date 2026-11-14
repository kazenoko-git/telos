"""Summarize separate evaluation tracks without mixing pipeline and direct QA."""
import collections
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SUITE=ROOT.parent/'xerj_reader_2026-11-14'/'suite_512_cases.json'


def correct(score):
    if isinstance(score,bool):return score
    if isinstance(score,dict):return bool(score.get('answer_correct',score.get('correct',False)))
    return False


def summarize(path,case_map):
    data=json.loads(path.read_text());records=data.get('records',[]);meta=data.get('run_metadata',{})
    output_mode=meta.get('track',meta.get('output_mode',path.stem))
    ids=[r['case_id'] for r in records]
    if len(set(ids))!=len(ids):raise ValueError(f'Duplicate task IDs: {path.name}')
    if set(ids)-case_map.keys():raise ValueError(f'Unknown task IDs: {path.name}')
    result={'artifact':path.name,'model_id':meta.get('model_id'),'track':output_mode,'records':len(records),'complete':len(records)==512,
            'extraction_exact':0,'extraction_grounded':0,'pipeline_correct':0,'direct_correct':0,'executor_contract_errors':0,'categories':{},'status_groups':{},'errors':{}}
    errs=collections.Counter()
    for record in records:
        case=case_map[record['case_id']];cat=case['category'];group='answer_present' if 'expected' in case else 'absence_or_conflict'
        eg=record.get('extraction_grade') or {};pipeline=correct(record.get('executor_score'));direct=correct(record.get('grade',record.get('direct_score')))
        ext=bool(eg.get('exact_fields'));grounded=bool(eg.get('grounded'))
        for key,val in [('extraction_exact',ext),('extraction_grounded',grounded),('pipeline_correct',pipeline),('direct_correct',direct)]:result[key]+=val
        for container,bucket in [(result['categories'],cat),(result['status_groups'],group)]:
            row=container.setdefault(bucket,{'total':0,'extraction_exact':0,'pipeline_correct':0,'direct_correct':0});row['total']+=1;row['extraction_exact']+=ext;row['pipeline_correct']+=pipeline;row['direct_correct']+=direct
        executor=record.get('deterministic_executor',record.get('executor_result',{}))
        result['executor_contract_errors']+=executor.get('status')=='CONTRACT_ERROR'
        attempts=record.get('attempts')
        attempt_error=(attempts[-1].get('error') if isinstance(attempts,list) and attempts else None)
        err=record.get('error') or (record.get('response') or {}).get('error') or (record.get('inference_settings') or {}).get('error') or attempt_error
        if err:errs[str(err.get('kind','error') if isinstance(err,dict) else err)]+=1
    result['errors']=dict(errs)
    return result


def main():
    cases=json.loads(SUITE.read_text());case_map={c['id']:c for c in cases}
    paths=sorted(p for p in ROOT.glob('*.json') if p.name not in ('summary.json',) and 'smoke' not in p.name and 'preflight' not in p.name and 'native_order' not in p.name and 'health' not in p.name)
    rows=[summarize(p,case_map) for p in paths if 'records' in json.loads(p.read_text())]
    (ROOT/'summary.json').write_text(json.dumps({'tracks':rows},indent=2)+'\n')
    lines=['# Tiny extraction models','', 'Direct QA, extraction-only, and calculator results are separate tracks. Pipeline operations and requested source fields are human-specified.','', '| Model / track | Tasks | Exact extraction | Calculator answer | Direct QA |','|---|---:|---:|---:|---:|']
    for r in rows:
        direct='direct' in str(r['track']).lower() or 'direct' in r['artifact'];n=r['records']
        lines.append(f"| {r['model_id']} / {r['track']} | {n}/512 | {'—' if direct else str(r['extraction_exact'])+'/'+str(n)} | {'—' if direct else str(r['pipeline_correct'])+'/'+str(n)} | {str(r['direct_correct'])+'/'+str(n) if direct else '—'} |")
    lines+=['','Extraction means exact source fields; calculator means final status/value correctness after deterministic execution. Source-substring grounding is not semantic entailment. Empty missing fields are evaluated separately in summary.json. Incomplete runs are not full-suite scores.','', 'See ../xerj_reader_2026-11-14/tiny_specialist_protocol.md for the frozen protocol.']
    (ROOT/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))

if __name__=='__main__':main()
