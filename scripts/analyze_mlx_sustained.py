"""Evaluate predeclared sustained asynchronous-accumulation acceptance criteria."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):value.update(block)
    return value.hexdigest()


def rate(rows):
    return len(rows)*16384/sum((r['end_monotonic_ns']-r['start_monotonic_ns'])/1e9 for r in rows)


def analyze(root):
    plan=json.loads((root/'acceptance_plan.json').read_text())
    runs=[]
    for index,variant in enumerate(plan['order']):
        directory=root/f'{index:02d}_{variant=="async"}'
        summary=directory/'training/summary.json'
        if not summary.exists():continue
        capture=json.loads((directory/'capture.json').read_text())
        host=[json.loads(line) for line in (directory/'host_steps.jsonl').read_text().splitlines()]
        memory=[json.loads(line) for line in (directory/'training/steps.jsonl').read_text().splitlines()]
        hardware=[json.loads(line) for line in (directory/'hardware.jsonl').read_text().splitlines()]
        warm=host[plan['warmup']:]
        mem=memory[plan['warmup']:]
        third=len(mem)//3
        first,last=mem[:third],mem[-third:]
        windows=[dict(first_update=i+1,last_update=i+64,tokens_per_second=rate(host[i:i+64]),
            peak_mlx_mib=max(r['mlx_peak_bytes'] for r in memory[i:i+64])/2**20)
            for i in range(plan['warmup'],len(host)-63,64)]
        hw=[r for r in hardware if plan['warmup']<=r['completed_steps']<plan['updates']]
        freqs=[r['gpu_active_frequency_mhz'] for r in hw if r['gpu_active_frequency_mhz'] is not None]
        state=directory/'final_training_state.safetensors'
        runs.append(dict(index=index,variant=variant,
            complete=capture.get('worker_returncode')==0 and capture.get('training_summary',{}).get('status')=='complete'
                and len(host)==plan['updates'] and state.exists(),
            updates=len(host),tokens_per_second=rate(warm),
            warmed_seconds=sum((r['end_monotonic_ns']-r['start_monotonic_ns'])/1e9 for r in warm),
            peak_mlx_mib=max(r['mlx_peak_bytes'] for r in memory)/2**20,
            active_growth_mib=(statistics.mean(r['mlx_active_bytes'] for r in last)-statistics.mean(r['mlx_active_bytes'] for r in first))/2**20,
            cache_growth_mib=(statistics.mean(r['mlx_cache_bytes'] for r in last)-statistics.mean(r['mlx_cache_bytes'] for r in first))/2**20,
            final_state_sha256=digest(state) if state.exists() else None,
            final_model_sha256=digest(directory/'training/checkpoints/model.safetensors'),
            gpu_mhz_mean=statistics.mean(freqs) if freqs else None,
            thermal_states=sorted({r['thermal_label'] for r in hw}),windows=windows))
    result=dict(runs=runs,complete=len(runs)==4 and all(r['complete'] for r in runs))
    if result['complete']:
        pairs=[]
        for a,b in ((runs[0],runs[1]),(runs[3],runs[2])):
            gains=[100*(y['tokens_per_second']/x['tokens_per_second']-1) for x,y in zip(a['windows'],b['windows'])]
            pairs.append(dict(sync_index=a['index'],async_index=b['index'],gain_pct=100*(b['tokens_per_second']/a['tokens_per_second']-1),window_gains_pct=gains))
        groups=[]
        for variant in ('sync','async'):
            selected=[r for r in runs if r['variant']==variant]
            groups.append(dict(variant=variant,tokens_per_second=len(selected)*(plan['updates']-plan['warmup'])*16384/sum(r['warmed_seconds'] for r in selected),mean_peak_mlx_mib=statistics.mean(r['peak_mlx_mib'] for r in selected)))
        gain=100*(groups[1]['tokens_per_second']/groups[0]['tokens_per_second']-1)
        overhead=groups[1]['mean_peak_mlx_mib']-groups[0]['mean_peak_mlx_mib']
        checks=dict(paired_gain=all(p['gain_pct']>=2 for p in pairs),pooled_gain=gain>=2,
            every_window_positive=all(g>0 for p in pairs for g in p['window_gains_pct']),
            active_memory_stable=all(r['active_growth_mib']<=32 for r in runs),
            retained_cache_stable=all(r['cache_growth_mib']<=32 for r in runs),
            memory_overhead_bounded=overhead<=384,
            model_equal=len({r['final_model_sha256'] for r in runs})==1,
            complete_training_state_equal=len({r['final_state_sha256'] for r in runs})==1)
        result.update(pairs=pairs,groups=groups,pooled_gain_pct=gain,peak_memory_overhead_mib=overhead,
            checks=checks,accepted=all(checks.values()))
    (root/'sustained_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    result=analyze(parser.parse_args().directory)
    print(json.dumps({k:v for k,v in result.items() if k!='runs'},indent=2))
    for r in result['runs']:print(json.dumps({k:v for k,v in r.items() if k!='windows'}))
