"""Offline saved-score replay: Python standard library only; no model or API calls."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
CONTRACT=ROOT/'reports/xerj_reader_2026-11-14'
sys.path.insert(0,str(CONTRACT))
from run_comparative_suite import classify_outcome
from specialist_contract import execute,public_plan,grade_extraction


def main():
    source=ROOT/'reports/reader1_consolidated_2026-11-15'
    report=json.loads((source/'results.json').read_text())
    cases={r['id']:r for r in json.loads((CONTRACT/'suite_512_cases.json').read_text())}
    plans={r['case_id']:r for r in json.loads((CONTRACT/'pipeline_task_plans.json').read_text())['cases']}
    scored={};exact={}
    for row in report['cases']:
        case=cases[row['id']]
        if row['track']=='direct QA':response={'status':row['status'],'value':row['observed']}
        else:
            plan=plans[row['id']];response=execute(case['document'],public_plan(plan),row['extracted_fields'])
            grade=grade_extraction(case['document'],plan,row['extracted_fields']);assert grade['exact_fields']==row['exact_extraction']
            exact[row['model']]=exact.get(row['model'],0)+grade['exact_fields']
        ok=classify_outcome(case,response) in ('correct_answer','correct_abstention')
        assert ok==row['correct'],(row['model'],row['id'],response)
        key=(row['model'],row['track']);counts=scored.setdefault(key,[0,0]);counts[0]+=ok;counts[1]+=1
    for (name,track),(passed,total) in sorted(scored.items()):
        assert total==512
        print(f'{name:20} {track:24} {passed}/{total}')
    print('All 6,144 saved case records replayed. No inference, API keys, or telos installation used.')

if __name__=='__main__':main()
