"""Score new generation traces as they arrive; no extra model generation."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from collections import Counter

OUT = Path(__file__).resolve().parent
LABELS = ['100m_r50_modal_h100_seed42_cad1','100m_5b','100m_5b_alpha075','100m_schedstep1','100m_tpu_2']
TASKS = {t['task_id']: t for t in [json.loads(l) for l in gzip.decompress((OUT/'HumanEval.jsonl.gz').read_bytes()).splitlines()]}
HELPER = OUT/'strict_executor.py'
STOPS = ['\ndef ', '\nclass ', '\nif __name__']


def execute(code):
    with tempfile.TemporaryDirectory(prefix='telos_strict_humaneval_') as scratch:
        p = subprocess.Popen([sys.executable,'-I',str(HELPER)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,cwd=scratch,start_new_session=True)
        try:
            stdout, stderr = p.communicate(json.dumps({'code':code}),timeout=5.0)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGKILL)
            p.communicate()
            return {'outcome':'TIMEOUT','details':'Five-second subprocess wall timeout'}
        try:
            return json.loads(stdout)
        except json.JSONDecodeError:
            return {'outcome':'RUNTIME_EXCEPTION','details':f'Worker exited{p.returncode} without a result; stderr:{stderr[-300:]}'}


def save(value):
    temp=OUT/'strict_summary.tmp.json';temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(OUT/'strict_summary.json')


def main():
    ref=TASKS['HumanEval/0'];harness=ref['test']+'\ncheck('+ref['entry_point']+')'
    good=execute(ref['prompt']+ref['canonical_solution']+'\n'+harness)
    bad=execute(ref['prompt']+'    return False\n'+harness)
    assert good['outcome']=='PASSED' and bad['outcome']=='FAILED_ASSERTION',(good,bad)
    summary={'protocol':{'suite':'HumanEval164','processing':'Stop phrases removed; no AST trimming, indentation repair, or function replacement','timeout_seconds':5.0,'memory_limit_mb':512,'network':'disabled','executor_source_sha256':hashlib.sha256(HELPER.read_bytes()).hexdigest(),'dataset_sha256':hashlib.sha256((OUT/'HumanEval.jsonl.gz').read_bytes()).hexdigest(),'sanity':{'canonical':good,'incorrect':bad}},'models':{}}
    seen={label:set() for label in LABELS}
    for label in LABELS:
        summary['models'][label]={'tasks':0,'passed':0,'outcomes':{},'task_results':[]}
    save(summary)
    deadline=time.monotonic()+5*3600
    while time.monotonic()<deadline:
        changed=False
        for label in LABELS:
            path=OUT/(label+'.generations.jsonl')
            if not path.exists():continue
            for line in path.read_text().splitlines():
                try:row=json.loads(line)
                except json.JSONDecodeError:continue
                task_id=row['id']
                if task_id in seen[label]:continue
                task=TASKS[task_id]
                completion=row['raw_completion']
                for stop in STOPS:
                    completion=completion.split(stop,1)[0]
                candidate=task['prompt']+completion
                result=execute(candidate+'\n'+task['test']+'\ncheck('+task['entry_point']+')\n')
                result.update(id=task_id,completion=completion)
                bucket=summary['models'][label]
                bucket['task_results'].append(result);seen[label].add(task_id)
                bucket['tasks']=len(seen[label]);bucket['passed']+=result['outcome']=='PASSED'
                bucket['outcomes'][result['outcome']]=bucket['outcomes'].get(result['outcome'],0)+1
                bucket['pass_at_1_pct']=100*bucket['passed']/bucket['tasks']
                save(summary);changed=True
                if bucket['tasks']%20==0 or bucket['tasks']==164:
                    print(f'[Strict] {label}: {bucket["tasks"]}/164 scored, {bucket["passed"]} passed',flush=True)
        if all(len(seen[label])==164 for label in LABELS):
            summary['status']='complete';save(summary);print('All5 models strictly scored on164 tasks.',flush=True);return
        native_path=OUT/'summary.json'
        if native_path.exists():
            native=json.loads(native_path.read_text())
            if len(native.get('models',{}))+len(native.get('errors',{}))==len(LABELS) and native.get('errors'):
                summary['status']='incomplete';summary['generation_errors']=native['errors'];save(summary);return
        if not changed:time.sleep(2)
    summary['status']='incomplete';save(summary)


if __name__=='__main__':main()
