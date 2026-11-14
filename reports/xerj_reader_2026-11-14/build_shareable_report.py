"""Build a report with explicit recovery provenance; never overwrite source runs."""
import base64
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
import generate_comparative_display as display
import run_comparative_suite as runner

NAMES = {'ministral-3-8b-instruct-2512': 'Ministral 3 8B', 'google/gemma-4-e4b': 'Gemma 4 E4B',
         'lfm3-omni-2b': 'LFM3 Omni 2B', 'ternary-bonsai-4b-mlx': 'Bonsai 4B',
         'xerj-reader-1': 'Reader-1', 'qwen3.5-0.8b-mlx': 'Qwen 3.5 0.8B'}


def main():
    sources = ['comparative_512_v2_local.json', 'comparative_512_v2_xerj.json', 'comparative_512_qwen08b.json']
    data = {'model_runs': {}, 'benchmark_meta': {'suite_cases_file': 'suite_512_cases.json',
            'suite_sha256': hashlib.sha256((ROOT/'suite_512_cases.json').read_bytes()).hexdigest(),
            'case_count': 512, 'publication_policy': 'Status/value correctness; Qwen 2B excluded; Gemini incomplete; no silent retry selection.'}}
    for filename in sources:
        src = json.loads((ROOT / filename).read_text())
        data['model_runs'].update({k: copy.deepcopy(v) for k,v in src['model_runs'].items() if k in NAMES})
    # Retry recovery is applied only when its distinct, completed artifact is supplied.
    retry_path = ROOT / 'qwen08_http500_retry.json'
    if retry_path.exists():
        retry = json.loads(retry_path.read_text())
        retry_meta = retry.get('benchmark_meta', {})
        original_hash = hashlib.sha256((ROOT/'comparative_512_qwen08b.json').read_bytes()).hexdigest()
        if retry_meta.get('source_sha256') != original_hash:
            raise ValueError('Recovery source hash does not match preserved original run')
        if not retry_meta.get('finished_utc'):
            raise ValueError('Recovery run is still in progress')
        records = retry.get('cases', [])
        if not records and retry.get('model_runs'):
            records = next(iter(retry['model_runs'].values())).get('cases', [])
        originals = data['model_runs']['qwen3.5-0.8b-mlx']['cases']
        by_id = {c['case_id']: c for c in originals}
        expected = {c['case_id'] for c in originals if (c['response'].get('error') or {}).get('http_status') == 500}
        if len(records) == len(expected) and {c['case_id'] for c in records} == expected:
            for c in records:
                old = by_id[c['case_id']]
                replacement = copy.deepcopy(c)
                replacement['original_record'] = old
                replacement['recovery_retry'] = True
                by_id[c['case_id']] = replacement
            data['model_runs']['qwen3.5-0.8b-mlx']['cases'] = [by_id[c['case_id']] for c in originals]
            data['benchmark_meta']['recovery'] = {'source': retry_path.name, 'sha256': hashlib.sha256(retry_path.read_bytes()).hexdigest(), 'retried_http500_cases': len(records)}
        else:
            raise ValueError('Recovery artifact must cover exactly the original 53 HTTP500 task IDs')
    for run in data['model_runs'].values():
        for c in run['cases']:
            if c['response'].get('status') == 'MALFORMED':
                c['grade']['outcome_classification'] = 'malformed_response'
        run['summary'] = {name: runner.compute_subset_metrics([c for c in run['cases'] if name == 'combined_512' or c['subset'] == name])
                          for name in ['combined_512','inspected_240','new_272']}
    (ROOT / 'shareable_comparison.json').write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n')
    payload = display.build_normalized_payload(data, display.load_suite_cases(REPO))
    recovered = data['benchmark_meta'].get('recovery')
    qnote = 'Qwen 0.8B includes a separate recovery run of 53 original HTTP500 requests; original first-pass score: 282/512.' if recovered else 'Qwen 0.8B: 53 HTTP500 requests await recovery; first-pass score includes these failures.'
    note = '512 synthetic tasks; status/value correctness. Evidence checked separately. ' + qnote + ' Gemini is incomplete and excluded. Qwen 2B excluded.'
    payload['note'] = note
    html = display.generate_html(payload, base64.b64encode((REPO/'telos/eval/display_assets/logo.png').read_bytes()).decode())
    # Compare the same timing layer; retain provider duration in case details.
    html = html.replace('s.latencies_ms.median', 's.latencies_ms.client_median')
    html = html.replace('Lowest Response Latency', 'Lowest Client Latency').replace('Median Latency', 'Median Client Latency')
    (ROOT/'shareable_report.html').write_text(html)
    render(data, note)


def render(data, note):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['Avenir Next','Helvetica Neue','Arial'], 'text.color':'#2D2A26'})
    fig=plt.figure(figsize=(15,11), dpi=180, facecolor='#FDF9F5')
    logo=fig.add_axes([.055,.9,.14,.08]);logo.imshow(plt.imread(REPO/'figures/test_transparent_logo.png'));logo.axis('off')
    fig.text(.23,.945,'Document extraction', fontsize=25, weight='medium')
    fig.text(.23,.912,'512 synthetic tasks · Status/value correctness · 14 November 2026',fontsize=11,color='#5C554D')
    ordered=sorted(data['model_runs'],key=lambda m:data['model_runs'][m]['summary']['combined_512']['correct_count'], reverse=True)
    ax=fig.add_axes([.25,.58,.50,.27],facecolor='#FDF9F5')
    vals=[data['model_runs'][m]['summary']['combined_512']['accuracy_pct'] for m in ordered]
    ax.barh(range(len(ordered)),vals,color=['#2563AB' if m=='xerj-reader-1' else '#9B5B3E' for m in ordered],height=.56)
    ax.set_yticks(range(len(ordered)),[NAMES[m] for m in ordered],fontsize=12);ax.invert_yaxis();ax.set_xlim(0,110)
    ax.set_xticks([0,25,50,75,100],['0','25','50','75','100%']);ax.tick_params(length=0,pad=10);ax.set_axisbelow(True);ax.grid(axis='x',color='#E8DFD5');ax.spines[['top','right','left','bottom']].set_visible(False)
    for i,m in enumerate(ordered):
        s=data['model_runs'][m]['summary']['combined_512'];ax.text(vals[i]+1,i,f"{s['correct_count']}/512",va='center',fontsize=10)
        fig.text(.83,.58+.27*(1-(i+.5)/len(ordered)),f"{s['latencies_ms']['client_median']:,.0f} ms",va='center',fontsize=12)
    fig.text(.83,.865,'Client median',fontsize=10,color='#5C554D')
    cats=['scalar','arithmetic','percentages','dates','corrections','table_lookup','unicode','missing','conflicts','units','irrelevant_instructions','lists']
    ax2=fig.add_axes([.25,.17,.66,.34]);matrix=np.array([[data['model_runs'][m]['summary']['combined_512']['categories'][c]['accuracy_pct'] for m in ordered] for c in cats])
    ax2.imshow(matrix,cmap=LinearSegmentedColormap.from_list('creamgreen',['#F4EBE1','#D2E5DC','#80B39F']),vmin=0,vmax=100,aspect='auto')
    ax2.set_yticks(range(len(cats)),[c.replace('_',' ').capitalize() for c in cats],fontsize=10);ax2.set_xticks(range(len(ordered)),[NAMES[m].replace('Ministral 3','Ministral').replace('Qwen 3.5','Qwen') for m in ordered],fontsize=9);ax2.xaxis.tick_top();ax2.tick_params(length=0,pad=8)
    for y,c in enumerate(cats):
        for x,m in enumerate(ordered):
            s=data['model_runs'][m]['summary']['combined_512']['categories'][c];ax2.text(x,y,f"{s['correct']}/{s['total']}",ha='center',va='center',fontsize=9)
    for sp in ax2.spines.values():sp.set_visible(False)
    import textwrap
    footer='Status/value scores do not require valid evidence. Provider timing for Reader-1 is separate from client latency.\n'+note+'\nParameterized diagnostic suite; not a general-document benchmark.'
    fig.text(.06,.11,'\n'.join(textwrap.fill(line,145) for line in footer.splitlines()),fontsize=9,color='#5C554D',va='top',linespacing=1.6)
    for ext in ['png','pdf']:fig.savefig(ROOT/f'shareable_overview.{ext}',facecolor=fig.get_facecolor())
    plt.close(fig)

if __name__=='__main__': main()
