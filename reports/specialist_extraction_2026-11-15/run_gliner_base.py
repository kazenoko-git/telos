"""Native GLiNER2 span extraction with separately graded deterministic execution."""
import argparse,hashlib,importlib.metadata,json,os,platform,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CONTRACT=ROOT.parent/'xerj_reader_2026-11-14'
sys.path.insert(0,str(CONTRACT))
from specialist_contract import public_plan,execute,grade_extraction
from run_comparative_suite import classify_outcome
MODEL='fastino/gliner2-base-v1'
REVISION='f9634218e53580c56edf0de97ca1a7d3f1c2354e'


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def normalize(raw,names):
    if not isinstance(raw,dict):raise ValueError('Native result must be an object')
    rows=raw.get('requested_values',[])
    if not isinstance(rows,list):raise ValueError('Native requested_values must be a list')
    fields={name:[] for name in names}
    for row in rows:
        if not isinstance(row,dict):raise ValueError('Native row must be an object')
        for name in names:
            vals=row.get(name,[])
            if isinstance(vals,str):vals=[vals]
            if isinstance(vals,dict):vals=[vals]
            if not isinstance(vals,list):raise ValueError('Native field must be a list')
            for value in vals:
                text=value.get('text') if isinstance(value,dict) else value
                if not isinstance(text,str):raise ValueError('Native span missing text')
                if text not in fields[name]:fields[name].append(text)
    return fields


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int);parser.add_argument('--output',type=Path,default=ROOT/'gliner2_base_extraction.json');args=parser.parse_args()
    import torch
    from huggingface_hub import snapshot_download
    from gliner2 import GLiNER2
    torch.set_num_threads(4)
    local=Path(snapshot_download(MODEL,revision=REVISION,local_dir='/tmp/telos-tiny-eval-models/gliner2-base-v1',allow_patterns=['*.json','*.safetensors','spm.model']))
    start=time.perf_counter();model=GLiNER2.from_pretrained(str(local)).to('cpu').eval();load_ms=(time.perf_counter()-start)*1000
    cases=json.loads((CONTRACT/'suite_512_cases.json').read_text());plans=json.loads((CONTRACT/'pipeline_task_plans.json').read_text());byid={p['case_id']:p for p in plans['cases']}
    if args.limit:cases=cases[:args.limit]
    def command(cmd):
        try:return subprocess.check_output(cmd,text=True,stderr=subprocess.DEVNULL).strip()
        except Exception:return None
    metadata={'model_id':MODEL,'revision':REVISION,'weight_sha256':sha(local/'model.safetensors'),'suite_sha256':sha(CONTRACT/'suite_512_cases.json'),'taskplan_sha256':sha(CONTRACT/'pipeline_task_plans.json'),'output_mode':'extraction_and_pipeline','runtime':{k:importlib.metadata.version(k) for k in ['torch','transformers','gliner2','tokenizers']},'device':'cpu','dtype':str(next(model.parameters()).dtype),'timing_policy':'Per-case client wall time includes native inference, field parsing and local grading; model load excluded.','runner_sha256':sha(Path(__file__)),'threads':4,'threshold':0.5,'model_load_ms':load_ms,'hardware':{'platform':platform.platform(),'chip':command(['sysctl','-n','machdep.cpu.brand_string']),'ram_bytes':command(['sysctl','-n','hw.memsize']),'power_settings':command(['pmset','-g','custom'])},'started_utc':datetime.now(timezone.utc).isoformat(),'finished_utc':None,'expected_cases':len(cases),'source_implementation':'run_gliner_base.py','normalization':'Unwrap native span text; preserve native order; deduplicate identical span text per field. No gold-based completion.'}
    result={'run_metadata':metadata,'records':[]}
    def save():
        tmp=args.output.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n');tmp.replace(args.output)
    for i,case in enumerate(cases):
        plan=byid[case['id']];pub=public_plan(plan)
        schema={'requested_values':[{'name':f['name'],'dtype':'list','description':f['description']+' Original question: '+case['question']+' Copy literal numbers, dates, or names only; omit units from numeric literals. Do not calculate.'} for f in pub['fields']]}
        record={'case_id':case['id'],'category':case['category'],'subset':case['subset'],'attempts':1,'native_schema':schema,'error':None}
        timer=time.perf_counter();raw=None
        try:
            with torch.inference_mode():raw=model.extract_json(case['document'],schema,threshold=0.5,include_confidence=True,include_spans=True)
            fields=normalize(raw,[f['name'] for f in pub['fields']])
            executor=execute(case['document'],pub,fields)
            record.update(raw_completion=raw,parsed_fields=fields,extraction_grade=grade_extraction(case['document'],plan,fields),deterministic_executor=executor,executor_score={'answer_correct':classify_outcome(case,executor) in ('correct_answer','correct_abstention'),'outcome':classify_outcome(case,executor)})
        except Exception as err:
            record.update(raw_completion=raw,parsed_fields=None,extraction_grade={'exact_fields':False,'grounded':False},deterministic_executor={'status':'RUNTIME_ERROR','value':None},executor_score={'answer_correct':False},error={'kind':type(err).__name__,'message':str(err)})
        record['client_ms']=round((time.perf_counter()-timer)*1000,3);result['records'].append(record);save()
        if args.limit or (i+1)%32==0:print(f"{i+1}/{len(cases)} extraction={sum(r['extraction_grade']['exact_fields'] for r in result['records'])} pipeline={sum(r['executor_score']['answer_correct'] for r in result['records'])}",flush=True)
        if i<2 and record['error']:raise RuntimeError(record['error'])
    metadata['finished_utc']=datetime.now(timezone.utc).isoformat();save()

if __name__=='__main__':main()
