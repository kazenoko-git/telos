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
let pageSize = 60;
const presentation = report.presentation || {};
const models = [...new Set([...report.rows.map(r=>r.model), ...report.suites.map(r=>r.model)])];
const comparison = models.length > 1 || new Set(report.rows.map(r=>r.suite)).size > 1;
const labels = {direct_qa:'Direct QA', extraction:'Source extraction', calculator:'Calculator'};
const trackLabel = t => labels[t] || t;
const notes = {direct_qa:'Correct status + value',extraction:'All requested source fields match',calculator:'Extracted fields + fixed operations'};
const medianOf = rows => {const a=rows.map(r=>r.latency_ms).filter(v=>typeof v==='number'&&Number.isFinite(v)).sort((a,b)=>a-b);return a.length?(a[Math.floor((a.length-1)/2)]+a[Math.ceil((a.length-1)/2)])/2:null;};
byId('note').textContent = report.note;
byId('source').textContent = comparison ? '' : report.source ? 'Source · '+report.source : '';
for (const name of ['model','suite','category']) {
  const values = [...new Set(report.rows.map(r=>r[name]).filter(Boolean))].sort();
  if (!values.length) values.push(...new Set(report.suites.map(s=>s[name]).filter(Boolean)));
  for (const value of values) { const option = make('option',name==='suite'?trackLabel(value):value); option.value=value; byId(name).append(option); }
}
if (comparison) {
  document.body.classList.add('comparison-report');
  byId('comparison').hidden=false;
  byId('model').value=presentation.featured_model || models[0];
  byId('suite').value=presentation.default_suite || report.suites[0]?.suite || report.rows[0]?.suite || '';
  if(!presentation.comparison_pages)for(const id of ['model','suite']) byId('analysis-selectors').append(byId(id).parentElement);
  byId('suite').parentElement.firstChild.textContent='Track';
  byId('comparison-note').textContent=(presentation.footnotes||[]).join(' · ');
}
function scoresFor(track) {
  return report.suites.filter(s=>s.suite===track).map(s=>{
    const rows=report.rows.filter(r=>r.model===s.model&&r.suite===track);
    const correct=rows.length?rows.filter(r=>r.correct===true).length:s.correct;
    return {...s,correct,latency:medianOf(rows) ?? s.client_ms ?? null};
  }).sort((a,b)=>(b.score??-1)-(a.score??-1)||a.model.localeCompare(b.model));
}
function ranking() {
  if(!comparison)return;
  const track=byId('suite').value;
  const tabs=byId('track-tabs');tabs.replaceChildren();
  for(const t of [...new Set(report.suites.map(s=>s.suite))]){
    const button=make('button',trackLabel(t));button.setAttribute('aria-pressed',String(t===track));
    button.addEventListener('click',()=>{byId('suite').value=t;pageSize=60;render();});tabs.append(button);
  }
  byId('track-note').textContent=notes[track]||'';
  const board=byId('leaderboard');board.replaceChildren();
  const head=make('div',undefined,'rank-head');['Model','Correct','Accuracy','Client median'].forEach(x=>head.append(make('span',x)));board.append(head);
  for(const score of scoresFor(track)){
    const row=make('div',undefined,'rank-row'+(score.model===presentation.featured_model?' featured':''));
    row.append(make('strong',score.model,'rank-model'),make('span',`${score.correct ?? '—'} / ${score.total ?? '—'}`,'rank-count'),make('strong',score.score==null?'—':score.score.toFixed(1)+'%','rank-score'),make('span',score.latency==null?'—':(score.latency/1000).toFixed(2)+' s','rank-time'));board.append(row);
  }
  const paired=byId('print-pipelines');paired.replaceChildren();
  const extraction=scoresFor('extraction'),calculator=scoresFor('calculator');
  if(extraction.length){paired.append(make('h2','Extraction pipelines'));const table=make('div',undefined,'pipeline-grid');
    for(const t of ['Model','Source fields','Calculator'])table.append(make('strong',t));
    for(const e of extraction){const c=calculator.find(c=>c.model===e.model);table.append(make('span',e.model),make('span',`${e.correct} / ${e.total}`),make('span',c?`${c.correct} / ${c.total}`:'—'));}paired.append(table,make('p','Human-specified operations; separate from direct QA.','fine'));
  }
}
function examples(){
  const cards=byId('example-cards');cards.replaceChildren();
  for(const key of presentation.selected_cases||[]){const row=report.rows.find(r=>r.id===key.id&&r.model===key.model&&r.suite===key.suite);if(!row)continue;
    const box=make('article',undefined,'example-card');box.append(make('h3',row.question));
    box.append(make('p',`Expected: ${format(row.expected)} · Observed: ${row.status==='ANSWER'?format(row.value):row.status}`));
    const button=make('button','View source');button.addEventListener('click',()=>details(row));box.append(button);cards.append(box);
  }
  byId('examples').hidden=!cards.children.length;
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
  byId('metrics').hidden=comparison;
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
  if(comparison) {
    byId('category-heading').textContent=byId('model').value || 'Categories';
    const saved=(presentation.verification||[]).filter(v=>v.model===byId('model').value&&(!v.suite||v.suite===byId('suite').value));
    if(saved.length)byId('checks').replaceChildren(...saved.map(v=>check(v.label,v.value)));
    else {
      const checks=[];
      if(replay.length)checks.push(check('Formula replay',`${replay.filter(r=>r.replay).length} / ${replay.length}`));
      if(offsets.length)checks.push(check('Evidence offsets',`${offsets.filter(r=>r.offsets).length} / ${offsets.length}`));
      byId('checks').replaceChildren(...checks);
    }
    document.querySelector('.assurance').hidden=!byId('checks').children.length;
  }
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
  if(comparison){
    const track=byId('suite').value;
    const available=report.suites.filter(s=>s.suite===track).map(s=>s.model);
    if(!available.includes(byId('model').value))byId('model').value=available.includes(presentation.featured_model)?presentation.featured_model:available[0]||'';
    for(const option of byId('model').options)option.disabled=Boolean(option.value&&!available.includes(option.value));
  }
  const query=byId('search').value.toLowerCase();
  visible=report.rows.filter(row=>['model','suite','category'].every(k=>!byId(k).value||row[k]===byId(k).value)&&(!byId('outcome').value||state(row)===byId('outcome').value)&&(!query||JSON.stringify(row).toLowerCase().includes(query)));
  const body=byId('rows');body.replaceChildren();
  for (const row of visible.slice(0,pageSize)) {
    const tr=make('tr'),cell=make('td'),button=make('button');
    button.append(make('span',row.model+' · '+row.id,'case-id'),make('span',row.question||row.suite));button.addEventListener('click',()=>details(row));cell.append(button);tr.append(cell);
    tr.append(make('td',row.category.replaceAll('_',' ')));
    const result=make('td');result.append(make('span',stateLabel(row),'result '+state(row)),make('div',row.status==='ANSWER'?format(row.value):row.status,'actual'));tr.append(result);
    tr.append(make('td',row.confidence===null?'—':row.confidence.toFixed(4),'nowrap'),make('td',row.latency_ms===null?'—':(row.latency_ms/1000).toFixed(2)+'s','nowrap'));
    body.append(tr);
  }
  byId('count').textContent=`${Math.min(pageSize,visible.length)} of ${visible.length} matching cases`;
  byId('more').hidden=visible.length<=pageSize;
  ranking();
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
byId('more').addEventListener('click',()=>{pageSize+=60;render();});
byId('close').addEventListener('click',()=>byId('detail').close());
byId('reset').addEventListener('click',()=>{for(const k of ['search','model','suite','category','outcome'])byId(k).value='';if(comparison){byId('model').value=presentation.featured_model||models[0];byId('suite').value=presentation.default_suite||report.suites[0].suite;}pageSize=60;render();});
for(const k of ['search','model','suite','category','outcome'])byId(k).addEventListener(k==='search'?'input':'change',()=>{pageSize=60;render();});
function comparisonPages(){
  const definitions=presentation.comparison_pages;
  if(!Array.isArray(definitions)||!definitions.length)return;
  document.body.classList.add('curated-pages');
  const root=byId('comparison-pages');root.hidden=false;
  function score(model,track,category){
    const rows=report.rows.filter(r=>r.model===model&&r.suite===track&&(!category||r.category===category));
    return rows.length?{correct:rows.filter(r=>r.correct===true).length,total:rows.length}:null;
  }
  function matrix(group){
    const box=make('section',undefined,'category-matrix');box.append(make('h3',group.title));
    const wrap=make('div',undefined,'matrix-wrap'),table=make('table'),head=make('thead'),hr=make('tr');
    hr.append(make('th','Category'));for(const name of group.models)hr.append(make('th',group.labels?.[name]||name));head.append(hr);table.append(head);
    const body=make('tbody');
    const categories=[...new Set(report.rows.filter(r=>r.suite===group.track&&group.models.includes(r.model)).map(r=>r.category))].sort();
    for(const category of categories){
      const tr=make('tr');tr.append(make('th',category.replaceAll('_',' ')));
      const scores=group.models.map(m=>score(m,group.track,category));
      const best=Math.max(...scores.map(s=>s?s.correct/s.total:-1));
      const leaders = scores.filter(s => s && best > 0 && s.correct / s.total === best).length;
      for (const s of scores) {
        const bestScore = s && best > 0 && s.correct / s.total === best;
        const className = bestScore ? (leaders === 1 ? 'category-lead' : 'category-tie') : '';
        const td = make('td', undefined, className);
        if (bestScore) td.title = leaders === 1 ? 'Outright category win' : 'Tied category lead';
        td.append(make('span', s ? `${s.correct}/${s.total}` : '—', 'matrix-value'));
        tr.append(td);
      }
      body.append(tr);
    }
    table.append(body);wrap.append(table);box.append(wrap);return box;
  }
  for(const [index,definition] of definitions.entries()){
    const page=make('article',undefined,'comparison-page');
    const heading=make('div',undefined,'comparison-page-heading');heading.append(document.querySelector('.brand img').cloneNode(),make('h1',definition.title),make('span',`${index+1} / ${definitions.length}`,'page-number'));page.append(heading);
    page.append(make('p',definition.note||'512 synthetic document tasks','comparison-page-note'));
    const summary=make('div',undefined,'score-cards');summary.style.setProperty('--model-count',definition.models.length);
    const directOnly=definition.columns.filter(c=>c.kind!=='latency').length===1;
    if(directOnly)summary.classList.add('direct-only');
    for(const model of definition.models){
      const card=make('section',undefined,'score-card'+(model===presentation.featured_model?' featured-model':''));card.append(make('h3',model));
      for(const col of definition.columns){
        const s=score(model,col.track),metric=make('div',undefined,'score-card-metric');
        if(col.kind==='latency'){
          const time=scoresFor(col.track).find(s=>s.model===model)?.latency;
          metric.append(make('span',col.label),make('strong',time==null?'—':(time/1000).toFixed(2)+' s'));
        }else if(directOnly){
          card.append(make('div',s?percent(s.correct,s.total):'—','score-card-value'),make('div',s?`${s.correct} / ${s.total} correct`:'Not evaluated','score-card-count'));
          continue;
        }else{
          metric.append(make('span',col.label),make('strong',s?percent(s.correct,s.total):'—'));
        }
        card.append(metric);
      }
      if(directOnly){
        const group=definition.matrices[0];
        const categories=[...new Set(report.rows.filter(r=>r.suite===group.track&&group.models.includes(r.model)).map(r=>r.category))];
        const sole=[],tied=[];
        for(const category of categories){const counts=group.models.map(m=>score(m,group.track,category)),best=Math.max(...counts.map(c=>c?c.correct/c.total:-1));const own=score(model,group.track,category);if(own&&best>0&&own.correct/own.total===best){const leaders=counts.filter(c=>c&&c.correct/c.total===best).length;(leaders===1?sole:tied).push(category.replaceAll('_',' '));}}
        card.append(make('p',sole.length?'Leads · '+sole.join(', '):tied.length?`${tied.length} shared category leads`:'No category leads','model-strength'));
      }
      summary.append(card);
    }
    page.append(summary);
    const matrices=make('div',undefined,'comparison-matrices');if(definition.matrices.length===1)matrices.classList.add('single');for(const group of definition.matrices)matrices.append(matrix(group));page.append(matrices);
    page.append(make('p','Green = outright category win. Dot = tied best. Correct / tasks.','comparison-legend'));
    if(definition.footer)page.append(make('p',definition.footer,'comparison-footer'));
    root.append(page);
  }
}
comparisonPages();

examples();
render();
window.telosReportReady=true;
