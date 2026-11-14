"""Replay native extraction normalization only; no inference or gold-driven edits."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CONTRACT=ROOT.parent/'xerj_reader_2026-11-14';sys.path.insert(0,str(CONTRACT))
from run_gliner_base import normalize
from specialist_contract import public_plan,execute,grade_extraction
from run_comparative_suite import classify_outcome
source=ROOT/'gliner2_base_native_order.json';data=json.loads(source.read_text())
cases={c['id']:c for c in json.loads((CONTRACT/'suite_512_cases.json').read_text())}
plans={p['case_id']:p for p in json.loads((CONTRACT/'pipeline_task_plans.json').read_text())['cases']}
for record in data['records']:
    case=cases[record['case_id']];plan=plans[record['case_id']]
    fields=normalize(record['raw_completion'],[f['name'] for f in plan['fields']])
    result=execute(case['document'],public_plan(plan),fields)
    record['parsed_fields']=fields;record['extraction_grade']=grade_extraction(case['document'],plan,fields)
    record['deterministic_executor']=result;record['executor_score']={'answer_correct':classify_outcome(case,result) in ('correct_answer','correct_abstention'),'outcome':'contract_error' if result.get('status')=='CONTRACT_ERROR' else classify_outcome(case,result)}
data['run_metadata']['normalization']='Native span text, source-ordered by native character start, deduplicated; no inference rerun.'
data['run_metadata']['postprocess']={'source':source.name,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'reason':'GLiNER native spans are confidence-ordered; contract requires source order for ordered lists.'}
(ROOT/'gliner2_base_extraction.json').write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
print('extraction',sum(r['extraction_grade']['exact_fields'] for r in data['records']),'pipeline',sum(r['executor_score']['answer_correct'] for r in data['records']))
