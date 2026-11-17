"""Analyze randomized MLX switchbacks with independent clock/thermal matching.

Report overall net effect separately from the conditional same-observed-clock
comparison. Matching a treatment-dependent clock is not a hardware clock lock.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def paired_summary(pairs,seed=20261117):
    if not pairs:return dict(pairs=0)
    logs=[math.log(p['async_rate']/p['sync_rate']) for p in pairs]
    if len(logs)==1:
        return dict(pairs=1,geometric_gain_pct=100*math.expm1(logs[0]),
            bootstrap_95_pct=None,positive_pairs=int(logs[0]>0),
            caveat='One pair cannot estimate a confidence interval.')
    rng=random.Random(seed)
    bootstrap=sorted(statistics.mean(rng.choices(logs,k=len(logs))) for _ in range(10000))
    return dict(pairs=len(pairs),geometric_gain_pct=100*math.expm1(statistics.mean(logs)),
        bootstrap_95_pct=[100*math.expm1(bootstrap[249]),100*math.expm1(bootstrap[9749])],
        positive_pairs=sum(x>0 for x in logs),
        caveat='Paired block bootstrap, not independent-machine or independent-long-run confidence.')


def hardware_block(rows,start,end):
    contained=[r for r in rows if r.get('window_start_epoch_ns',-1)>=start
        and r.get('window_end_epoch_ns',end+1)<=end]
    valid=[r for r in contained if r['gpu_active_frequency_mhz'] is not None
        and r.get('gpu_frequency_mapping_coverage')==1 and r.get('gpu_active_residency') is not None]
    result=dict(full_intervals=len(contained),mapped_intervals=len(valid),
        thermal_states=sorted({r['thermal_label'] for r in contained}),
        all_ac=bool(contained) and all("Now drawing from 'AC Power'" in r.get('power_source','') for r in contained))
    other_cpu=[]
    for record in contained:
        values=[]
        for line in record.get('processes','').splitlines():
            parts=line.strip().split(None,3)
            if len(parts)==4 and not Path(parts[3]).name.startswith('python') and Path(parts[3]).name!='ps':
                values.append(float(parts[1]))
        other_cpu.append(max(values,default=0))
    result['median_top_other_cpu_pct']=statistics.median(other_cpu) if other_cpu else None
    if not valid:return result
    weights=[r['interval_seconds']*r['gpu_active_residency'] for r in valid]
    total=sum(weights)
    if total<=0:return result
    mean=sum(w*r['gpu_active_frequency_mhz'] for w,r in zip(weights,valid))/total
    variance=sum(w*(r['gpu_active_frequency_mhz']-mean)**2 for w,r in zip(weights,valid))/total
    states={}
    for r in valid:
        for name,value in r['performance_states'].get('GPUPH',[]):
            if name not in ('OFF','DOWN','IDLE'):states[name]=states.get(name,0)+value
    state_total=sum(states.values())
    result.update(frequency_mhz=mean,frequency_cv=math.sqrt(variance)/mean,
        active_state_distribution={k:v/state_total for k,v in states.items()} if state_total else {})
    return result


def matched_pair(sync,asynchronous,policy):
    reasons=[]
    hs,ha=sync['hardware'],asynchronous['hardware']
    for name,h in [('sync',hs),('async',ha)]:
        if h['full_intervals']<policy['min_full_hardware_intervals_per_block']:reasons.append(name+': insufficient intervals')
        if h['mapped_intervals']!=h['full_intervals'] or 'frequency_mhz' not in h:reasons.append(name+': unmapped clock')
        if h.get('frequency_cv',math.inf)>policy['max_block_frequency_cv']:reasons.append(name+': clock variation')
        if len(h['thermal_states'])!=1 or any(x not in ('nominal','fair') for x in h['thermal_states']):reasons.append(name+': thermal transition')
        if not h['all_ac']:reasons.append(name+': AC power not established')
    delta=None
    if 'frequency_mhz' in hs and 'frequency_mhz' in ha:
        delta=abs(hs['frequency_mhz']-ha['frequency_mhz'])/((hs['frequency_mhz']+ha['frequency_mhz'])/2)
        if delta>policy['max_pair_frequency_relative_difference']:reasons.append('frequency mismatch')
    if hs['thermal_states']!=ha['thermal_states']:reasons.append('thermal-state mismatch')
    if hs['median_top_other_cpu_pct'] is None or ha['median_top_other_cpu_pct'] is None:
        reasons.append('background CPU unavailable')
    elif abs(hs['median_top_other_cpu_pct']-ha['median_top_other_cpu_pct'])>policy['max_pair_other_cpu_pct_difference']:
        reasons.append('background CPU mismatch')
    names=set(hs.get('active_state_distribution',{}))|set(ha.get('active_state_distribution',{}))
    tv=sum(abs(hs.get('active_state_distribution',{}).get(k,0)-ha.get('active_state_distribution',{}).get(k,0)) for k in names)/2
    return dict(matched=not reasons,rejections=reasons,relative_frequency_difference=delta,
        active_state_total_variation_distance=tv)


def analyze(root):
    plan=json.loads((root/'plan.json').read_text())
    results=json.loads((root/'results.json').read_text())
    trials=[];all_pairs=[]
    for trial in range(2):
        directory=root/f'trial-{trial}'
        capture=json.loads((directory/'capture.json').read_text())
        rows=[json.loads(line) for line in (directory/'host_steps.jsonl').read_text().splitlines()]
        hardware=[json.loads(line) for line in (directory/'hardware.jsonl').read_text().splitlines()]
        memory=[json.loads(line) for line in (directory/'training/steps.jsonl').read_text().splitlines()]
        schedule=json.loads((root/f'schedule-{trial}.json').read_text())
        assert len(rows)==plan['updates']
        assert [r['async_accumulation'] for r in rows]==[x for mode in schedule['modes'] for x in [mode]*plan['block_updates']]
        assert [r['schedule_block'] for r in rows]==[i//plan['block_updates'] for i in range(len(rows))]
        assert capture['status']=='untraced_complete' and capture['worker_returncode']==0
        blocks=[]
        for start in range(plan['warmup_updates'],len(rows),plan['block_updates']):
            batch=rows[start:start+plan['block_updates']]
            seconds=sum((r['end_monotonic_ns']-r['start_monotonic_ns'])/1e9 for r in batch)
            span=(batch[-1]['end_epoch_ns']-batch[0]['start_epoch_ns'])/1e9
            blocks.append(dict(block=start//plan['block_updates'],first_update=start+1,
                async_accumulation=batch[0]['async_accumulation'],updates=len(batch),seconds=seconds,
                span_seconds=span,rate=len(batch)*16384/seconds,span_rate=len(batch)*16384/span,
                hardware=hardware_block(hardware,batch[0]['start_epoch_ns'],batch[-1]['end_epoch_ns']),
                peak_mlx_mib=max(r['mlx_peak_bytes'] for r in memory[start:start+plan['block_updates']])/2**20))
        pairs=[]
        for left,right in zip(blocks[::2],blocks[1::2]):
            sync,asynchronous=sorted((left,right),key=lambda b:b['async_accumulation'])
            match=matched_pair(sync,asynchronous,plan['matching'])
            p=dict(trial=trial,pair=left['block']//2,sync_block=sync['block'],async_block=asynchronous['block'],
                sync_rate=sync['rate'],async_rate=asynchronous['rate'],gain_pct=100*(asynchronous['rate']/sync['rate']-1),
                sync_span_rate=sync['span_rate'],async_span_rate=asynchronous['span_rate'],
                sync_frequency_mhz=sync['hardware'].get('frequency_mhz'),async_frequency_mhz=asynchronous['hardware'].get('frequency_mhz'),**match)
            pairs.append(p)
        groups=[]
        for mode in (False,True):
            selected=[b for b in blocks if b['async_accumulation']==mode]
            groups.append(dict(async_accumulation=mode,updates=sum(b['updates'] for b in selected),
                seconds=sum(b['seconds'] for b in selected),rate=sum(b['updates'] for b in selected)*16384/sum(b['seconds'] for b in selected),
                span_rate=sum(b['updates'] for b in selected)*16384/sum(b['span_seconds'] for b in selected)))
        measured=memory[plan['warmup_updates']:];third=len(measured)//3
        growth={name:(statistics.mean(r[key] for r in measured[-third:])-statistics.mean(r[key] for r in measured[:third]))/2**20
            for name,key in [('active_growth_mib','mlx_active_bytes'),('cache_growth_mib','mlx_cache_bytes')]}
        trials.append(dict(trial=trial,groups=groups,overall=paired_summary(pairs),
            matched=paired_summary([p for p in pairs if p['matched']]),blocks=blocks,pairs=pairs,
            peak_mlx_mib=max(r['mlx_peak_bytes'] for r in memory)/2**20,
            model_sha256=digest(directory/'training/checkpoints/model.safetensors'),
            array_state_sha256=digest(directory/'final_training_state.safetensors'),**growth))
        all_pairs.extend(pairs)
    matched=[p for p in all_pairs if p['matched']]
    sufficient=(len(matched)>=plan['matching']['minimum_matched_pairs_total'] and
        all(t['matched']['pairs']>=plan['matching']['minimum_matched_pairs_per_trial'] for t in trials))
    aggregate=[]
    for index in (0,1):
        g=[t['groups'][index] for t in trials]
        aggregate.append(dict(async_accumulation=bool(index),rate=sum(x['updates'] for x in g)*16384/sum(x['seconds'] for x in g)))
    output=dict(complete=all(r['capture_status']=='untraced_complete' and r['power_policy_unchanged'] for r in results),
        overall=paired_summary(all_pairs),matched=paired_summary(matched),matched_support_sufficient=sufficient,
        pooled=aggregate,pooled_net_gain_pct=100*(aggregate[1]['rate']/aggregate[0]['rate']-1),
        models_identical=len({t['model_sha256'] for t in trials})==1,
        array_states_identical=len({t['array_state_sha256'] for t in trials})==1,
        trials=trials,caveats=['No hardware clock lock or absolute temperature control.',
            'ThermalState is coarse; CPU clocks and unobserved activity are not fully controlled.',
            'Matched-clock result is a conditional estimate; net switchback result retains treatment-induced clock effects.',
            'Bootstrap intervals describe these adjacent block pairs, not independent long runs or machines.'])
    (root/'switchback-analysis.json').write_text(json.dumps(output,indent=2)+'\n')
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    output=analyze(parser.parse_args().directory)
    print(json.dumps({k:v for k,v in output.items() if k!='trials'},indent=2))
    for trial in output['trials']:print(json.dumps({k:v for k,v in trial.items() if k not in ('blocks','pairs')},indent=2))
