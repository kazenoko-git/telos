"""Build an artifact-derived consolidated report; no inference or source overwrite."""
import base64,collections,hashlib,html,json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
R=REPO/'reports/xerj_reader_2026-11-14'
S=REPO/'reports/specialist_extraction_2026-11-15'
N=REPO/'reports/nuextract3_2026-11-15'
SOURCES={}

def read(path):
    raw=path.read_bytes();SOURCES[str(path.relative_to(REPO))]=hashlib.sha256(raw).hexdigest();return json.loads(raw)

def pct(n,d=512):return f'{n}/{d} ({n/d:.1%})'
def med(records):return statistics.median(r.get('client_ms',r.get('latency_ms',0)) for r in records)


def collect():
    cases={c['id']:c for c in read(R/'suite_512_cases.json')}
    sources=[R/'comparative_512_v2_local.json',R/'comparative_512_v2_xerj.json']
    raw={}
    for source in sources:raw.update(read(source)['model_runs'])
    q=read(R/'comparative_512_qwen08b.json')['model_runs']['qwen3.5-0.8b-mlx'];retries=read(R/'qwen08_http500_retry.json')['model_runs']['qwen3.5-0.8b-mlx']['cases'];by={r['case_id']:r for r in retries};q['cases']=[by.get(c['case_id'],c) for c in q['cases']];raw['qwen3.5-0.8b-mlx']=q
    names={'google/gemma-4-e4b':'Gemma 4 E4B','lfm3-omni-2b':'LFM3 Omni 2B','ternary-bonsai-4b-mlx':'Bonsai 4B','xerj-reader-1':'Reader-1','qwen3.5-0.8b-mlx':'Qwen 3.5 0.8B'}
    rows=[];detail=[]
    for key,name in names.items():
        records=raw[key]['cases'];assert len(records)==512 and len({c['case_id'] for c in records})==512
        row={'name':name,'id':key,'correct':sum(c['grade']['answer_correct'] for c in records),'client_ms':med(records),'categories':{},'error_count':sum(c['response'].get('status') not in ('ANSWER','ABSENT','UNKNOWN','CONFLICT','AMBIGUOUS') for c in records)}
        for c in records:
            case=cases[c['case_id']];cat=case['category'];r=row['categories'].setdefault(cat,[0,0]);r[0]+=c['grade']['answer_correct'];r[1]+=1
            detail.append({'model':name,'track':'direct QA','id':c['case_id'],'category':cat,'correct':c['grade']['answer_correct'],'document':case['document'],'question':case['question'],'expected':case.get('expected',case.get('statuses')),'observed':c['response'].get('value'),'status':c['response'].get('status'),'source':key,'recovery':key=='qwen3.5-0.8b-mlx' and c['case_id'] in by})
        rows.append(row)
    for name,path in [('NuExtract3',N/'nuextract3_direct_qa.json')]:
        records=read(path)['records'];row={'name':name,'id':name,'correct':sum(r['grade']['answer_correct'] for r in records),'client_ms':med(records),'categories':{},'error_count':sum(r['response'].get('status') not in ('ANSWER','ABSENT','UNKNOWN','CONFLICT','AMBIGUOUS') for r in records)}
        assert len(records)==512 and len({r['case_id'] for r in records})==512
        for c in records:
            case=cases[c['case_id']];cat=case['category'];r=row['categories'].setdefault(cat,[0,0]);r[0]+=c['grade']['answer_correct'];r[1]+=1
            detail.append({'model':name,'track':'direct QA','id':c['case_id'],'category':cat,'correct':c['grade']['answer_correct'],'document':case['document'],'question':case['question'],'expected':case.get('expected',case.get('statuses')),'observed':c['response'].get('value'),'status':c['response'].get('status'),'source':str(path.relative_to(REPO))})
        rows.append(row)
    rows.sort(key=lambda r:(-r['correct'],r['name']))
    specialists=[]
    for name,path in [('NuExtract3',N/'nuextract3_extraction.json'),('NuExtract Tiny',S/'nuextract_1_5_tiny_extraction.json'),('GLiNER2 Base',S/'gliner2_base_extraction.json')]:
        records=read(path)['records'];assert len(records)==512
        specialists.append({'name':name,'extraction':sum(r['extraction_grade']['exact_fields'] for r in records),'pipeline':sum(r['executor_score']['answer_correct'] for r in records),'records':512})
        for c in records:
            case=cases[c['case_id']]
            detail.append({'model':name,'track':'extraction + calculator','id':c['case_id'],'category':case['category'],'correct':c['executor_score']['answer_correct'],'document':case['document'],'question':case['question'],'expected':case.get('expected',case.get('statuses')),'observed':c.get('executor_result',c.get('deterministic_executor')),'extracted_fields':c['parsed_fields'],'exact_extraction':c['extraction_grade']['exact_fields'],'source':str(path.relative_to(REPO))})
    gemini=read(R/'comparative_512_v2_gemini.json')['model_runs']['gemini-3.5-flash-lite']['cases'];served=sum(r['response']['status']!='ERROR' for r in gemini);gem={'attempted':len(gemini),'served':served,'correct':sum(r['grade']['answer_correct'] for r in gemini),'errors':len(gemini)-served,'unattempted':512-len(gemini)}
    reader=raw['xerj-reader-1']['cases'];offsets=sum(r['grade']['evidence_offsets_valid'] for r in reader)
    examples=[]
    for cid in ['percentages-01','conflicts-01',next(r['case_id'] for r in reader if r['category']=='table_lookup' and not r['grade']['answer_correct'])]:
        c=next(r for r in reader if r['case_id']==cid);case=cases[cid];examples.append({'id':cid,'question':case['question'],'document':case['document'],'expected':case.get('expected',case.get('statuses')),'observed':c['response'].get('value'),'status':c['response']['status']})
    data={'date':'15 November 2026','direct':rows,'specialists':specialists,'gemini':gem,'reader_offsets':offsets,'reader_answer_evidence':sum(r['response']['status']=='ANSWER' and bool(r['response'].get('evidence')) and r['grade']['evidence_offsets_valid'] for r in reader),'reader_replay':raw['xerj-reader-1'].get('replay_evaluation'),'reader_examples':examples,'cases':detail,'sources':SOURCES,'suite_sha256':SOURCES[str((R/'suite_512_cases.json').relative_to(REPO))],'package_audit':read(ROOT/'package_audit.json')}
    (ROOT/'results.json').write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n');return data


def viewer_report(data):
    """Adapt study records to the public Télos viewer; no separate page layout."""
    from telos.eval.documents.suite import load_suite
    from telos.eval.documents.contracts import public_plan
    cases, plans, _ = load_suite()
    rows, suites = [], []
    for item in data['cases']:
        base = {'id': item['id'], 'model': item['model'], 'category': item['category'],
                'question': item['question'], 'document': item['document'],
                'expected': item['expected'], 'confidence': None, 'latency_ms': None,
                'evidence': [], 'program': '', 'replay': None, 'offsets': None,
                'input_tokens': None, 'details': None}
        if item['track'] == 'direct QA':
            rows.append(dict(base, suite='direct_qa', correct=item['correct'], status=item['status'], value=item['observed']))
        else:
            public = public_plan(plans[item['id']])
            extraction_expected = plans[item['id']]['reference_fields']
            rows.append(dict(base, suite='extraction', correct=item['exact_extraction'], status='EXTRACTED', value=item['extracted_fields'], expected=extraction_expected,
                             details={'operation': public['operation'], 'fields': public['fields']}))
            result = item['observed'] or {}
            rows.append(dict(base, suite='calculator', correct=item['correct'], status=result.get('status','ERROR'), value=result.get('value'),
                             details={'source_fields':item['extracted_fields'], 'operation':public['operation']}))
    # Preserve native source evidence and recorded per-request client times.
    sources = [R/'comparative_512_v2_local.json', R/'comparative_512_v2_xerj.json', R/'comparative_512_qwen08b.json']
    records = {}
    for path in sources:
        for key, run in read(path)['model_runs'].items():
            records[key] = {r['case_id']:r for r in run['cases']}
    for row in rows:
        original = next((i for i in data['direct'] if i['name']==row['model']),None)
        saved = records.get(original['id'],{}).get(row['id'],{}) if original else {}
        if row['suite']=='direct_qa' and saved and row['model']!='Qwen 3.5 0.8B':
            response=saved.get('response',{})
            row.update(latency_ms=saved.get('client_ms'), evidence=response.get('evidence',[]),
                       program=response.get('symbolic',{}).get('program_compact',''),
                       confidence=response.get('confidence'), details=saved.get('error') or response.get('reason'))
            if response.get('status')=='ANSWER' and response.get('evidence'):
                row['offsets']=saved.get('grade',{}).get('evidence_offsets_valid')
    for item in data['direct']:
        suites.append({'model':item['name'],'suite':'direct_qa','score':100*item['correct']/512,
                       'correct':item['correct'],'total':512,'client_ms':item['client_ms'],'metric':'status_value_correctness'})
    for item in data['specialists']:
        for track,key in [('extraction','extraction'),('calculator','pipeline')]:
            suites.append({'model':item['name'],'suite':track,'score':100*item[key]/512,'correct':item[key],
                           'total':512,'metric':'exact_source_fields' if track=='extraction' else 'status_value_correctness'})
    replay=data['reader_replay']
    return {'schema':'telos.evaluation.display.v1','title':'Reader-1 evaluation','source':'512 synthetic tasks · 15 November 2026',
            'note':'512 synthetic document tasks','rows':rows,'suites':suites,
            'presentation':{'featured_model':'Reader-1','default_suite':'direct_qa',
                'comparison_pages':[
                    {'title':'Document extraction models',
                     'note':'512 synthetic tasks · final answers and source extraction scored separately',
                     'models':['Reader-1','NuExtract3','NuExtract Tiny','GLiNER2 Base'],
                     'columns':[{'label':'Direct answer','track':'direct_qa'}, {'label':'Exact source fields','track':'extraction'}, {'label':'Fields + calculator','track':'calculator'}],
                     'matrices':[
                         {'title':'Direct answers','track':'direct_qa','models':['Reader-1','NuExtract3']},
                         {'title':'Exact source fields','track':'extraction','models':['NuExtract3','NuExtract Tiny','GLiNER2 Base'],'labels':{'NuExtract Tiny':'NuExtract Tiny','GLiNER2 Base':'GLiNER2'}}],
                     'footer':f"Reader-1: {data['reader_answer_evidence']}/454 answer evidence offsets matched; {replay['matched_replays']}/{replay['sampled_replays']} sampled formula replays matched. Calculator operations are human-specified; not Reader-1 program generation."},
                    {'title':'Reader-1 and small language models',
                     'note':'512 synthetic tasks · direct answers only',
                     'models':['Reader-1','LFM3 Omni 2B','Bonsai 4B','Qwen 3.5 0.8B','Gemma 4 E4B'],
                     'columns':[{'label':'Direct answer','track':'direct_qa'}, {'label':'Client median','track':'direct_qa','kind':'latency'}],
                     'matrices':[{'title':'Category results','track':'direct_qa','models':['Reader-1','LFM3 Omni 2B','Bonsai 4B','Qwen 3.5 0.8B','Gemma 4 E4B']}],
                     'footer':'Qwen 0.8B includes 53 recovered requests (initially 282/512). Client timing; runtimes differ.'}
                ],
                'verification':[{'model':'Reader-1','suite':'direct_qa','label':'Answer evidence offsets','value':str(data['reader_answer_evidence'])+'/454'},
                                {'model':'Reader-1','suite':'direct_qa','label':'Sampled formula replays','value':str(replay['matched_replays'])+'/'+str(replay['sampled_replays'])}],
                'selected_cases':[{'model':'Reader-1','suite':'direct_qa','id':e['id']} for e in data['reader_examples']],
                'footnotes':['Qwen 0.8B includes 53 recovered requests (initially 282/512).',
                             'Client timing; runtimes differ.']}}


if __name__ == '__main__':
    from telos.eval.display import write_html_report, write_pdf_report
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--html-only',action='store_true')
    args=parser.parse_args()
    data=collect()
    report=viewer_report(data)
    (ROOT/'viewer.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    write_html_report(report,ROOT/'report.html',title=report['title'],source=report['source'])
    if not args.html_only:
        write_pdf_report(ROOT/'report.html',ROOT/'Reader-1-evaluation.pdf')
    (ROOT/'manifest.json').write_text(json.dumps({'sources':SOURCES,'generated':['Reader-1-evaluation.pdf','report.html','viewer.json','results.json']},indent=2)+'\n')
    print('Built report with telos.eval.display')
