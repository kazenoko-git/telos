"""Build reference operands from source documents; gold never enters model prompts."""
import hashlib,json,re
from pathlib import Path
from specialist_contract import execute
ROOT=Path(__file__).resolve().parent
N=r'(\d+(?:\.\d+)?)'
D=r'(\d{2} [A-Za-z]+ \d{4})'


def build(case):
    doc=case['document'];q=case['question'];cat=case['category'];i=int(case['id'].split('-')[-1]);fields=[];refs={};op='resolve'
    def add(name,description,pattern=None,values=None,many=False):
        if values is None:
            match=re.search(pattern,doc)
            if not match:raise ValueError((case['id'],pattern))
            values=[match.group(1)]
        fields.append({'name':name,'description':description,'many':many})
        refs[name]=values
    direct_desc='Literal value(s) directly answering: '+q+' Include competing authoritative values; exclude superseded records and distractors. Return no value when absent.'
    if cat=='arithmetic':
        if 'east and west' in q:
            op='sum';add('east','Number of tasks completed by east',r'east completed '+N);add('west','Number of tasks completed by west',r'west completed '+N)
        elif 'cleanup' in q:
            op='subtract';add('initial','Cache entries before cleanup',r'cache held '+N);add('removed','Cache entries removed during cleanup',N+' entries were removed')
        elif 'central' in q:
            op='sum'
            for role in ('east','central','west'):add(role,'Tasks processed by region '+role,'region '+role+' processed '+N)
        elif 'queue' in q:
            op='add_subtract';add('initial','Initial queue balance',r'starting balance was '+N);add('added','Jobs added during intake',N+' jobs were added');add('removed','Jobs finished and removed',N+' jobs were finished')
        else:
            op='multiply';add('instances','Number of deployed instances',r'deployed across '+N);add('per_instance','Worker threads per instance',r'each running '+N)
    elif cat=='percentages':
        if 'increase' in q:
            op='increase_pct';add('before','Old throughput',r'Old throughput: '+N);add('after','New throughput',r'New throughput: '+N)
        elif 'reduction' in q:
            op='decrease_pct';add('before','Memory before tuning',r'peak memory usage was '+N);add('after','Memory after tuning',r'dropped to '+N)
        else:
            op='share_pct';add('total','Total checked or inspected items/records',N+' items were checked' if i<=20 else r'inspected '+N+' records');add('part','Failed items or records with validation warnings',N+' items failed' if i<=20 else r'Exactly '+N+' records')
    elif cat=='dates' and case['type']=='number':
        op='elapsed_days';add('start','Start/initiation date',r'(?:started |initiated on )'+D);add('end','Finish/conclusion date',r'(?:finished |concluded on )'+D)
    elif cat=='units':
        if '1024' in q:
            op='kb_to_mb';add('kilobytes','Storage metadata size in KB',r'holds '+N+' KB')
        else:
            op='rate'
            if i<=20 and i%2==0:
                row=re.search(r'run-\d+ \| '+N+r' \| '+N,doc);assert row
                add('quantity','Tokens processed by the requested run',values=[row.group(1)]);add('duration','Elapsed seconds for the requested run',values=[row.group(2)])
            else:
                add('quantity','Tokens processed or meters covered for the requested run/vehicle',r'(?:processed |covered )'+N)
                add('duration','Elapsed seconds for the requested run/vehicle',r'in '+N+' seconds')
    elif cat=='lists':
        op='list';values=re.findall(r'(?:alpha|beta|gamma)-\d+|task-[ABC]-\d+|service-\d+-(?:core|cache|gateway)',doc);add('values','All names in the requested approved/queue/dependency order; omit rejected or unrelated names',values=values,many=True)
    elif cat=='missing':add('values',direct_desc,values=[],many=True)
    elif cat=='conflicts':
        values=re.findall(r'(?:loss |latency |USD )'+N,doc);add('values',direct_desc,values=values,many=True)
        if case['type']=='money':add('currency','Currency code attached to the relevant amounts',values=['USD'])
    elif cat=='table_lookup':
        lines=doc.splitlines();sep='|' if '|' in doc else ','
        if i<=20:
            target=re.search(r'run (\w+) at step (\d+)',q).groups();row=next([v.strip() for v in line.split(sep)] for line in lines if tuple(v.strip() for v in line.split(sep)[:2])==target);value=row[-1]
        else:
            target=re.search(r'service (\S+) in region (\S+) with tier (\S+)\?',q).groups();row=next([v.strip() for v in line.split(sep)] for line in lines if tuple(v.strip() for v in line.split(sep)[:3])==target);value=row[-1]
        add('values',direct_desc,values=[value],many=True)
    elif cat=='scalar':add('values',direct_desc,r'(?:final loss: |measured loss |measured inference latency: |allocated memory )'+N,many=True)
    elif cat=='corrections':
        if case['type']=='money':add('values',direct_desc,r'(?:Original invoice amount USD |original billing amount USD )'+N,many=True);add('currency','Currency code attached to the relevant amount',values=['USD'])
        else:add('values',direct_desc,r'(?:replace that final loss with |update node allocation to )'+N,many=True)
    elif cat=='dates':add('values',direct_desc,r'(?:Confirmed delivery date: |scheduled audit date established as )'+D,many=True)
    elif cat=='irrelevant_instructions':add('values',direct_desc,r'(?:final loss |<loss>|convergence epoch: |<duration_sec>)'+N,many=True)
    elif cat=='unicode':
        patterns=[r'Café run-\d+: final loss '+N,r'日本語 run-\d+ processed '+N,r'Zürich Büro 🏢\): expenditure CHF '+N,r'محطة-شرق \(East Station\) logged '+N,r'Ω-Δ \(Ελλάδα 🏛️\): detected '+N]
        pattern=next(p for p in patterns if re.search(p,doc));add('values',direct_desc,pattern,many=True)
    else:raise ValueError(cat)
    return {'case_id':case['id'],'operation':op,'output_type':case['type'],'fields':fields,'reference_fields':refs}


def main():
    import run_comparative_suite as scorer
    raw=(ROOT/'suite_512_cases.json').read_bytes();cases=json.loads(raw);plans=[build(c) for c in cases]
    for case,plan in zip(cases,plans):
        result=execute(case['document'],plan,plan['reference_fields'])
        assert scorer.classify_outcome(case,result) in ('correct_answer','correct_abstention'),(case['id'],result)
    data={'schema':'telos.extraction-operand-plans.v1','suite_sha256':hashlib.sha256(raw).hexdigest(),'policy':'Human-authored task plans; private reference operands only for scoring. Models receive public fields, document, question. Not learned-program generation.','cases':plans}
    (ROOT/'pipeline_task_plans.json').write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n');print('Validated 512 reference operand executions against frozen answer gold')
if __name__=='__main__':main()
