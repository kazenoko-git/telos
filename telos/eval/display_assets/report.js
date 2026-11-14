/* All report strings are data: use textContent, never interpreted markup. */
'use strict';
const report = JSON.parse(document.getElementById('report-data').textContent);
const byId = id => document.getElementById(id);
const make = (tag, text, cls) => { const n = document.createElement(tag); if (text !== undefined) n.textContent = text; if (cls) n.className = cls; return n; };
const format = value => value === null || value === undefined ? 'Not recorded' : typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value);
const percent = (a,b) => b ? (100*a/b).toFixed(1)+'%' : '—';
const state = row => row.correct === true ? 'passed' : row.correct === false ? 'failed' : 'unknown';
const stateLabel = row => row.correct === true ? 'Correct' : row.correct === false ? 'Incorrect' : 'Unscored';
let visible = report.rows;
byId('note').textContent = report.note;
byId('source').textContent = report.source ? 'Source · '+report.source : 'Saved results';
for (const name of ['model','suite','category']) {
  const values = [...new Set(report.rows.map(r=>r[name]).filter(Boolean))].sort();
  if (!values.length) values.push(...new Set(report.suites.map(s=>s[name]).filter(Boolean)));
  for (const value of values) { const option = make('option',value); option.value=value; byId(name).append(option); }
}
function metric(label,value,note) {
  const box=make('div',undefined,'metric'); box.append(make('div',label,'metric-label'),make('div',value,'metric-value'),make('div',note,'metric-note')); return box;
}
function check(label,value) { const box=make('div',undefined,'check');box.append(make('span',label),make('strong',value));return box; }
function summaries(rows) {
  const scored=rows.filter(r=>typeof r.correct==='boolean');
  const correct=scored.filter(r=>r.correct).length;
  const latencies=rows.map(r=>r.latency_ms).filter(v=>typeof v==='number'&&Number.isFinite(v)).sort((a,b)=>a-b);
  const n=latencies.length, median=n ? (latencies[Math.floor((n-1)/2)]+latencies[Math.ceil((n-1)/2)])/2 : null;
  const answered=rows.filter(r=>r.status==='ANSWER');
  const answeredScored=answered.filter(r=>typeof r.correct==='boolean');
  const tokens=rows.filter(r=>typeof r.input_tokens==='number').reduce((a,r)=>a+r.input_tokens,0);
  const hasTokens=rows.some(r=>typeof r.input_tokens==='number');
  const suiteOnly=!report.rows.length;
  const summarySuites=report.suites.filter(s=>(!byId('model').value||s.model===byId('model').value)&&(!byId('suite').value||s.suite===byId('suite').value));
  byId('metrics').replaceChildren(
    metric(suiteOnly?'Recorded suites':'Correctness',suiteOnly?String(summarySuites.length):percent(correct,scored.length),suiteOnly?'Summary-only report':`${correct} correct / ${scored.length} scored`),
    metric('Task records',String(rows.length),suiteOnly?'Individual records not supplied':rows.length-scored.length?`${rows.length-scored.length} unscored / errors`:""),
    metric('Median client time',median===null?'—':(median/1000).toFixed(2)+'s',median===null?'Timing not recorded':`${n} timed model responses`),
    metric('Input tokens',hasTokens?tokens.toLocaleString():'—',hasTokens?'':'Usage not recorded'));
  const replay=rows.filter(r=>typeof r.replay==='boolean');
  const offsets=rows.filter(r=>typeof r.offsets==='boolean');
  const high=rows.filter(r=>r.correct===false&&typeof r.confidence==='number'&&r.confidence>=.9);
  const hasConfidence=rows.some(r=>typeof r.confidence==='number');
  byId('checks').replaceChildren(
    check('Formula replay',replay.length?`${replay.filter(r=>r.replay).length} / ${replay.length} match`:'Not recorded'),
    check('Evidence offset checks',offsets.length?`${offsets.filter(r=>r.offsets).length} / ${offsets.length} valid`:'Not recorded'),
    check('Answered-value precision',answeredScored.length?percent(answeredScored.filter(r=>r.correct).length,answeredScored.length):'Not recorded'),
    check('Incorrect, confidence ≥ 0.90',hasConfidence?String(high.length):'Not recorded'));
  const categories=byId('categories');categories.replaceChildren();
  if (suiteOnly) {
    byId('chart-label').textContent='Reported suite scores';
    for (const suite of summarySuites) bar(suite.model+' · '+suite.suite, suite.score, suite.score.toFixed(1)+'%',categories);
  } else {
    byId('chart-label').textContent='Correct / scored records';
    const groups=new Map();
    for (const row of scored) { const key=row.category||'General';if(!groups.has(key)) groups.set(key,[]);groups.get(key).push(row); }
    for (const [label,group] of groups) {const pass=group.filter(r=>r.correct).length;bar(label.replaceAll('_',' '),100*pass/group.length,`${pass} / ${group.length} · ${percent(pass,group.length)}`,categories);}
    if (!groups.size) categories.append(make('p','No scored cases in this selection.','muted'));
  }
}
function bar(label,value,caption,parent) {
  const box=make('div'),head=make('div',undefined,'bar-label'),track=make('div',undefined,'track'),fill=make('div',undefined,'fill');
  head.append(make('span',label),make('strong',caption));fill.style.width=Math.max(0,Math.min(100,value))+'%';track.append(fill);box.append(head,track);parent.append(box);
}
function render() {
  const query=byId('search').value.toLowerCase();
  visible=report.rows.filter(row=>['model','suite','category'].every(k=>!byId(k).value||row[k]===byId(k).value)&&(!byId('outcome').value||state(row)===byId('outcome').value)&&(!query||JSON.stringify(row).toLowerCase().includes(query)));
  const body=byId('rows');body.replaceChildren();
  for (const row of visible) {
    const tr=make('tr'),cell=make('td'),button=make('button');
    button.append(make('span',row.model+' · '+row.id,'case-id'),make('span',row.question||row.suite));button.addEventListener('click',()=>details(row));cell.append(button);tr.append(cell);
    tr.append(make('td',row.category.replaceAll('_',' ')));
    const result=make('td');result.append(make('span',stateLabel(row),'result '+state(row)),make('div',row.status==='ANSWER'?format(row.value):row.status,'actual'));tr.append(result);
    tr.append(make('td',row.confidence===null?'—':row.confidence.toFixed(4),'nowrap'),make('td',row.latency_ms===null?'—':(row.latency_ms/1000).toFixed(2)+'s','nowrap'));
    body.append(tr);
  }
  byId('count').textContent=`${visible.length} of ${report.rows.length} cases`;
  document.querySelector('.pill').textContent=visible.length===report.rows.length?'Saved evaluation':'Filtered selection';
  byId('empty').hidden=visible.length>0||report.rows.length===0;
  byId('summary-only').hidden=report.rows.length>0;
  summaries(visible);
}
function block(label,text,parent) {const box=make('section',undefined,'detail-block');box.append(make('h3',label),make('pre',text));parent.append(box);return box;}
function details(row) {
  byId('detail-id').textContent=row.model+' / '+row.id;
  byId('detail-title').textContent=row.question||row.suite;
  const content=byId('detail-content');content.replaceChildren();
  const grid=make('div',undefined,'detail-grid');block('Expected',format(row.expected),grid);block('Observed · '+row.status,format(row.value),grid);content.append(grid);
  if(row.program) block('Generated formula',row.program,content);
  if(row.document) {
    const box=make('section',undefined,'detail-block'),pre=make('pre');box.append(make('h3','Source document · quoted evidence highlighted'),pre);
    let at=0;
    const spans=[...(Array.isArray(row.evidence)?row.evidence:[])].sort((a,b)=>a.start-b.start);
    const chars=Array.from(row.document);
    for(const span of spans) {
      if(!Number.isInteger(span.start)||!Number.isInteger(span.end)||span.start<at||span.end<=span.start||span.end>chars.length) continue;
      if(chars.slice(span.start,span.end).join('')!==span.quote) continue;
      pre.append(document.createTextNode(chars.slice(at,span.start).join('')),make('mark',span.quote));at=span.end;
    }
    pre.append(document.createTextNode(chars.slice(at).join('')));content.append(box);
  }
  if(row.evidence&&row.evidence.length) block('Evidence · Unicode code-point offsets',JSON.stringify(row.evidence,null,2),content);
  if(row.details) block('Details',format(row.details),content);
  block('Verification',`Replay: ${format(row.replay)}\nExact offsets: ${format(row.offsets)}\nConfidence: ${format(row.confidence)}`,content);
  byId('detail').showModal();
}
function download(name,content,type) {const url=URL.createObjectURL(new Blob([content],{type}));const link=make('a');link.href=url;link.download=name;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
byId('download').addEventListener('click',()=>download('telos-evaluation.json',JSON.stringify(report,null,2),'application/json'));
byId('csv').addEventListener('click',()=>{
  const keys=['id','model','suite','category','correct','status','value','confidence','latency_ms'];
  // Prefix spreadsheet formulas; JSON remains available without transformation.
  const cell=x=>{let s=x===null||x===undefined?'':typeof x==='object'?JSON.stringify(x):String(x);if(/^[=+@\-\t\r]/.test(s))s="'"+s;return '"'+s.replaceAll('"','""')+'"';};
  download('telos-visible-cases.csv',[keys.join(','),...visible.map(r=>keys.map(k=>cell(r[k])).join(','))].join('\r\n'),'text/csv');
});
byId('print').addEventListener('click',()=>window.print());
byId('close').addEventListener('click',()=>byId('detail').close());
byId('reset').addEventListener('click',()=>{for(const k of ['search','model','suite','category','outcome'])byId(k).value='';render();});
for(const k of ['search','model','suite','category','outcome'])byId(k).addEventListener(k==='search'?'input':'change',render);
render();
