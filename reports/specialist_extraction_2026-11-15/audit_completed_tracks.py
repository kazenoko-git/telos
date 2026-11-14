"""Recompute all saved track grades offline, preserving source inference artifacts."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CONTRACT=ROOT.parent/'xerj_reader_2026-11-14';sys.path.insert(0,str(CONTRACT))
from specialist_contract import public_plan,grade_extraction,execute
from run_comparative_suite import classify_outcome


def main():
    suite_path=CONTRACT/'suite_512_cases.json';plan_path=CONTRACT/'pipeline_task_plans.json'
    cases={c['id']:c for c in json.loads(suite_path.read_text())};plans={p['case_id']:p for p in json.loads(plan_path.read_text())['cases']}
    result={'suite_sha256':hashlib.sha256(suite_path.read_bytes()).hexdigest(),'taskplan_sha256':hashlib.sha256(plan_path.read_bytes()).hexdigest(),'tracks':[]}
    for path in sorted(ROOT.glob('*.json')):
        if any(x in path.stem for x in ['summary','audit','smoke','preflight','native_order','health']):continue
        data=json.loads(path.read_text());records=data.get('records')
        if records is None:continue
        ids=[r['case_id'] for r in records];assert len(ids)==len(set(ids));assert not set(ids)-cases.keys()
        metadata=data['run_metadata'];assert metadata['suite_sha256']==result['suite_sha256']
        ph=metadata.get('taskplan_sha256',metadata.get('task_plan_sha256'));assert ph==result['taskplan_sha256']
        row={'artifact':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'records':len(records),'complete':len(records)==512,'correct':0,'extraction_exact':0,'prompt_reference_leaks':[]}
        for record in records:
            case=cases[record['case_id']];plan=plans[record['case_id']]
            if 'extraction_grade' in record:
                fields=record.get('parsed_fields')
                computed=grade_extraction(case['document'],plan,fields)
                for k in ('exact_fields','grounded'):
                    assert computed[k]==record['extraction_grade'][k],(path.name,record['case_id'],k)
                executor=record.get('deterministic_executor',record.get('executor_result'))
                if fields is not None:assert execute(case['document'],public_plan(plan),fields)==executor,(path.name,record['case_id'],'execute')
                answer_correct=classify_outcome(case,executor) in ('correct_answer','correct_abstention')
                assert answer_correct==record['executor_score']['answer_correct'],(path.name,record['case_id'],'pipeline_grade')
                row['extraction_exact']+=computed['exact_fields'];row['correct']+=answer_correct
            else:
                answer_correct=classify_outcome(case,record['response']) in ('correct_answer','correct_abstention')
                assert answer_correct==record['grade']['answer_correct'];row['correct']+=answer_correct
            prompt=record.get('prompt') or ''
            if any(x in prompt for x in ['"reference_fields":','"expected":','"operation":']):row['prompt_reference_leaks'].append(record['case_id'])
        assert not row['prompt_reference_leaks']
        result['tracks'].append(row)
    (ROOT/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    for row in result['tracks']:print(row['artifact'],row['records'],'records: replay verified')

if __name__=='__main__':main()
