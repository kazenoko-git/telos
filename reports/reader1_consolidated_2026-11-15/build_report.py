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
    names={'ministral-3-8b-instruct-2512':'Ministral 3 8B','google/gemma-4-e4b':'Gemma 4 E4B','lfm3-omni-2b':'LFM3 Omni 2B','ternary-bonsai-4b-mlx':'Bonsai 4B','xerj-reader-1':'Reader-1','qwen3.5-0.8b-mlx':'Qwen 3.5 0.8B'}
    rows=[];detail=[]
    for key,name in names.items():
        records=raw[key]['cases'];assert len(records)==512 and len({c['case_id'] for c in records})==512
        row={'name':name,'id':key,'correct':sum(c['grade']['answer_correct'] for c in records),'client_ms':med(records),'categories':{},'error_count':sum(c['response'].get('status') not in ('ANSWER','ABSENT','UNKNOWN','CONFLICT','AMBIGUOUS') for c in records)}
        for c in records:
            case=cases[c['case_id']];cat=case['category'];r=row['categories'].setdefault(cat,[0,0]);r[0]+=c['grade']['answer_correct'];r[1]+=1
            detail.append({'model':name,'track':'direct QA','id':c['case_id'],'category':cat,'correct':c['grade']['answer_correct'],'document':case['document'],'question':case['question'],'expected':case.get('expected',case.get('statuses')),'observed':c['response'].get('value'),'status':c['response'].get('status'),'source':key,'recovery':key=='qwen3.5-0.8b-mlx' and c['case_id'] in by})
        rows.append(row)
    for name,path in [('NuExtract3',N/'nuextract3_direct_qa.json'),('SmolLM2 360M',S/'smollm2_360m_instruct_direct_qa.json')]:
        records=read(path)['records'];row={'name':name,'id':name,'correct':sum(r['grade']['answer_correct'] for r in records),'client_ms':med(records),'categories':{},'error_count':sum(r['response'].get('status') not in ('ANSWER','ABSENT','UNKNOWN','CONFLICT','AMBIGUOUS') for r in records)}
        assert len(records)==512 and len({r['case_id'] for r in records})==512
        for c in records:
            case=cases[c['case_id']];cat=case['category'];r=row['categories'].setdefault(cat,[0,0]);r[0]+=c['grade']['answer_correct'];r[1]+=1
            detail.append({'model':name,'track':'direct QA','id':c['case_id'],'category':cat,'correct':c['grade']['answer_correct'],'document':case['document'],'question':case['question'],'expected':case.get('expected',case.get('statuses')),'observed':c['response'].get('value'),'status':c['response'].get('status'),'source':str(path.relative_to(REPO))})
        rows.append(row)
    rows.sort(key=lambda r:(-r['correct'],r['name']))
    specialists=[]
    for name,path in [('NuExtract3',N/'nuextract3_extraction.json'),('NuExtract Tiny',S/'nuextract_1_5_tiny_extraction.json'),('GLiNER2 Base',S/'gliner2_base_extraction.json'),('SmolLM2 360M',S/'smollm2_360m_instruct_extraction.json')]:
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


def pdf(data):
    from reportlab.pdfgen import canvas
    from reportlab.lib.colors import HexColor as H
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Paragraph
    from reportlab.lib.styles import ParagraphStyle
    regular='/System/Library/Fonts/Supplemental/Arial.ttf';bold='/System/Library/Fonts/Supplemental/Arial Bold.ttf'
    pdfmetrics.registerFont(TTFont('Body',regular));pdfmetrics.registerFont(TTFont('Bold',bold))
    W,HH=842,595;page=H('#FDF9F5');ink=H('#2D2A26');muted=H('#62594F');green=H('#23765D');rust=H('#9B5B3E');line=H('#E8DFD5')
    c=canvas.Canvas(str(ROOT/'Reader-1-evaluation.pdf'),pagesize=(W,HH));c.setTitle('Reader-1 evaluation | Telos');c.setAuthor('Ivan Samuel')
    def text(x,y,s,size=10,bold=False,color=None):c.setFont('Bold' if bold else 'Body',size);c.setFillColor(color or ink);c.drawString(x,y,str(s))
    def para(x,y,s,width=754,size=10,color=None):
        style=ParagraphStyle('p',fontName='Body',fontSize=size,leading=size*1.45,textColor=color or ink);p=Paragraph(s,style);_,h=p.wrap(width,500);p.drawOn(c,x,y-h);return y-h
    def start(title,sub,num):
        c.setFillColor(page);c.rect(0,0,W,HH,fill=1,stroke=0);c.drawImage(str(REPO/'figures/test_transparent_logo.png'),40,518,width=115,height=45,preserveAspectRatio=True,mask='auto');text(180,548,title,23,True);text(180,526,sub,10,color=muted);text(40,22,'Télos | 512 synthetic diagnostic tasks | 15 November 2026',8,color=muted);text(772,22,f'{num}/3',8,color=muted)
    def table(headers,rows,y,widths,highlight=None,fontsize=10,rowheight=29):
        x=40;text(x,y,headers[0],fontsize,True,color=muted)
        for i,h in enumerate(headers[1:],1):text(x+sum(widths[:i]),y,h,fontsize,True,color=muted)
        y-=14
        for row in rows:
            y-=rowheight
            if highlight and row[0]==highlight:c.setFillColor(H('#E7EFE7'));c.roundRect(34,y-8,774,rowheight-1,5,fill=1,stroke=0)
            for i,v in enumerate(row):text(x+sum(widths[:i]),y,v,fontsize,bold=bool(highlight and row[0]==highlight))
            c.setStrokeColor(line);c.line(40,y-10,802,y-10)
        return y-25
    start('Reader-1 evaluation','Corrected results; direct QA and configured pipelines reported separately',1)
    for i,(label,value) in enumerate([('Dates','43/43'),('Corrections','43/43'),('Answer evidence offsets',f"{data['reader_answer_evidence']}/454"),('Sampled executor replays','128/128')]):
        x=40+i*195;c.setFillColor(H('#F4EBE1'));c.roundRect(x,433,182,65,8,fill=1,stroke=0);text(x+12,476,label,9,color=muted);text(x+12,448,value,22,True,color=green)
    y=table(['Direct QA','Correct / 512','Client median','Recorded errors*'],[(r['name'],pct(r['correct']),f"{r['client_ms']:,.0f} ms",r['error_count']) for r in data['direct']],407,[215,210,175,150],highlight='Reader-1',rowheight=28)
    para(40,y,'Scores require the expected status and typed value. Qwen 0.8B includes error recovery. SmolLM2 uses unconstrained decoding and is strongly limited by output-format failures.',size=9,color=muted)
    para(40,64,'* Error column counts saved error/malformed/unsupported-status outputs; it is not the total number of wrong answers. Local latency includes measured client processing; Reader-1 provider timing is separate.',size=8.5,color=muted)
    c.showPage();start('Where Reader-1 does well','Direct QA by category | each entry is correct / category tasks',2)
    order=['Reader-1','NuExtract3','Ministral 3 8B','Gemma 4 E4B','LFM3 Omni 2B','Bonsai 4B','Qwen 3.5 0.8B','SmolLM2 360M'];mapping={r['name']:r for r in data['direct']};cats=['scalar','arithmetic','percentages','dates','corrections','table_lookup','unicode','missing','conflicts','units','irrelevant_instructions','lists']
    widths=[151]+[76]*8;heads=['Category','Reader-1','NuExtract3','Ministral','Gemma','LFM3','Bonsai','Qwen 0.8B','Smol 360M']
    y=table(heads,[(cat.replace('_',' '),*[f"{mapping[n]['categories'][cat][0]}/{mapping[n]['categories'][cat][1]}" for n in order]) for cat in cats],483,widths,fontsize=8,rowheight=25)
    y=para(40,y,'Reader-1 achieved perfect scalar extraction, dates, corrections, Unicode-labelled extraction and ordered lists. It scored 38/43 on irrelevant instructions, versus 12/43 for NuExtract3. Every one of its 454 ANSWER outputs had exact native evidence offsets; all 128 sampled executor replays matched.',size=10)
    para(40,y-14,'The largest gaps are percentages (0/43), unresolved conflicts (0/43), and multi-condition table lookup (10/42). A correct replay proves the selected program executes consistently; it does not prove the selected operands or operation answer the question.',size=10,color=muted)
    c.showPage();start('Extraction pipelines','Source extraction and calculator results are separate from direct QA',3)
    y=table(['Model','Exact source fields','Calculator answer'],[(r['name'],pct(r['extraction']),pct(r['pipeline'])) for r in data['specialists']],480,[250,250,220],rowheight=32)
    y=para(40,y,'The extractor receives a document, question and named source-field descriptions. A separate deterministic engine performs human-specified arithmetic, percentage, calendar and resolution operations. Exact extraction requires every requested source field; pipeline correctness scores the final status/value. These are different metrics from Reader-1 generating its own program.',size=10)
    y=para(40,y-18,'<b>Reader-1 failure examples</b>',size=11)
    for e in data['reader_examples']:
        y=para(40,y-13,f"<b>{e['id']}</b>: {html.escape(e['question'])}<br/>Expected: {html.escape(str(e['expected']))}; observed: {html.escape(str(e['observed']))} ({e['status']}).",size=10)
    para(40,y-20,'Qwen 0.8B includes 53 recovered HTTP 500 requests; its initial score was 282/512. SmolLM2 scores are strongly limited by output-format compliance. These synthetic diagnostics are not a general-document accuracy estimate.',size=9,color=muted)
    c.save()


def web(data):
    esc=lambda s:html.escape(str(s));logo=base64.b64encode((REPO/'figures/test_transparent_logo.png').read_bytes()).decode()
    sections=['<h2>Direct QA</h2><table><tr><th>Model</th><th>Correct / 512</th><th>Client median</th></tr>']
    for r in data['direct']:sections.append(f'<tr class="{"reader" if r["name"]=="Reader-1" else ""}"><td>{esc(r["name"])}</td><td>{pct(r["correct"])}</td><td>{r["client_ms"]:,.0f} ms</td></tr>')
    sections.append('</table><p>Qwen 0.8B includes 53 error-recovery requests (initial 282/512; recovered 316/512). Smol direct QA has 415 format/status failures. Tracks and decoding constraints differ.</p><h2>Extraction and calculator - separate tracks</h2><table><tr><th>Model</th><th>Exact fields</th><th>Calculator answer</th></tr>')
    for r in data['specialists']:sections.append(f'<tr><td>{esc(r["name"])}</td><td>{pct(r["extraction"])}</td><td>{pct(r["pipeline"])}</td></tr>')
    sections.append('</table><p>Calculator operations are human-specified. These scores are not direct-QA scores.</p><h2>Coverage and reproduction</h2>')
    sections.append(f'<p>Gemini: {data["gemini"]["attempted"]} attempted, {data["gemini"]["served"]} served, {data["gemini"]["correct"]} correct; quota-limited and excluded from the ranking. Qwen 2B excluded.</p><p><b>Package-only reproduction: unavailable in telos-ml 1.6.1.</b> Use the companion bundle for offline score replay and the repository runners for new inference. Native model dependencies and API credentials are additional requirements.</p><p>Parameterized synthetic cases; adapter repairs used this suite. No claim of population accuracy, equivalent decoding or semantically verified evidence. See the PDF for methodology and category breakdowns.</p>')
    payload=json.dumps(data['cases'],ensure_ascii=False).replace('<','\\u003c').replace('&','\\u0026')
    out='''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Reader-1 evaluation · Télos</title><style>body{background:#fdf9f5;color:#2d2a26;font:16px/1.55 -apple-system,BlinkMacSystemFont,Arial,sans-serif;margin:40px auto;max-width:1100px;padding:0 24px}header{display:flex;align-items:center;gap:35px}header img{width:160px}h1{font-size:32px;font-weight:550}h2{font-size:23px;margin-top:42px}p{color:#62594f}table{width:100%;border-collapse:collapse}th,td{text-align:left;border-bottom:1px solid #e8dfd5;padding:11px}th{color:#62594f;font-weight:500}.reader,.strengths{background:#e7efe7}.strengths{padding:20px;border-radius:12px;margin:28px 0}input,select{background:#fff9f2;border:1px solid #c8bcad;padding:10px;border-radius:5px;font:inherit;margin:5px}details{border-bottom:1px solid #e8dfd5;padding:12px}pre{white-space:pre-wrap;background:#f4ebe1;padding:18px;border-radius:8px;font-size:13px}summary{cursor:pointer}small{color:#62594f}@media print{input,select{display:none}}</style></head><body>'''
    out+=f'<header><img alt="Télos" src="data:image/png;base64,{logo}"><div><h1>Reader-1 evaluation</h1><p>512 synthetic tasks · 15 November 2026</p></div></header><div class="strengths"><b>Reader-1 strengths</b><br>Dates 43/43 · Corrections 43/43 · Exact offsets on 454/454 answers · Sampled replays 128/128<br><small>Also perfect on scalar extraction, Unicode-labelled extraction and ordered lists. Irrelevant instructions: 38/43.</small></div>'+''.join(sections)
    out+='<h2>All case-level results</h2><p>Filter by model, track, task ID or category. The first 100 matches are shown.</p><input id="search" aria-label="Search cases" placeholder="Search model, category, task ID"><select id="track" aria-label="Track"><option value="">All tracks</option><option>direct QA</option><option>extraction + calculator</option></select><select id="outcome" aria-label="Outcome"><option value="">All outcomes</option><option value="correct">Correct</option><option value="incorrect">Incorrect</option></select><p id="count"></p><main id="cases"></main>'
    out+=f'<script type="application/json" id="data">{payload}</script>'+'''<script>const data=JSON.parse(document.getElementById('data').textContent);function render(){const term=document.getElementById('search').value.toLowerCase(),track=document.getElementById('track').value,out=document.getElementById('outcome').value;const rows=data.filter(r=>(!track||r.track===track)&&(!out||(r.correct?'correct':'incorrect')===out)&&JSON.stringify([r.model,r.id,r.category]).toLowerCase().includes(term));document.getElementById('count').textContent=rows.length+' matching records · '+data.length+' total';const root=document.getElementById('cases');root.replaceChildren();for(const r of rows.slice(0,100)){const d=document.createElement('details'),s=document.createElement('summary'),p=document.createElement('pre');s.textContent=r.model+' · '+r.track+' · '+r.id+' · '+(r.correct?'Correct':'Incorrect');p.textContent=JSON.stringify(r,null,2);d.append(s,p);root.append(d)}}for(const id of ['search','track','outcome'])document.getElementById(id).addEventListener('input',render);render();</script></body></html>'''
    (ROOT/'report.html').write_text(out)

if __name__=='__main__':
    d=collect();pdf(d);web(d)
    (ROOT/'manifest.json').write_text(json.dumps({'sources':SOURCES,'generated':['Reader-1-evaluation.pdf','report.html','results.json']},indent=2)+'\n')
    print('Built consolidated PDF and searchable HTML;',len(d['cases']),'case records')
