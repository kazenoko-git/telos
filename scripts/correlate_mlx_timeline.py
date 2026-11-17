"""Correlate Metal GPU gaps with buffer preparation and sampled CPU stacks."""
import argparse
from collections import Counter
import csv
from datetime import datetime
import json
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET

from analyze_mlx_timeline import read_rows, merged


def stack_category(frames):
    text='\n'.join(frames)
    if 'ResidencySets::insert' in text or 'ResidencySet commit]' in text:
        return 'buffer residency update'
    if 'IOGPUResourceCreate' in text:
        return 'Metal buffer creation'
    if 'MetalAllocator::malloc' in text:
        return 'Metal allocation / allocator lock'
    if 'Event::wait()' in text or 'waitUntilSignaledValue' in text:
        return 'MLX completion wait'
    if 'performEnqueueKernel' in text or 'CommandEncoder::dispatch' in text:
        return 'kernel command encoding'
    if 'CommandBuffer' in text or 'commit' in text or 'submit' in text:
        return 'command submission / driver'
    if 'compile_replace' in text:
        return 'compiled graph preparation'
    if 'mlx::core::' in text:
        return 'other MLX native work'
    if frames and any('PyEval' in f or '_Py' in f for f in frames[:3]):
        return 'Python execution'
    return 'other / unresolved'


def union_ns(intervals,lo,hi):
    return sum(b-a for a,b in merged(intervals,lo,hi))


def correlate(directory):
    capture=json.loads((directory/'capture.json').read_text())
    pid=str(capture['worker_pid'])
    gpu=read_rows(directory/'gpu_intervals.xml')
    own=[r for r in gpu if r['process'].findtext('pid')==pid and r['state'].get('fmt')=='Active']
    own.sort(key=lambda r:int(r['start'].text))
    submissions={r['cmdbuffer-id'].text:r for r in read_rows(directory/'submissions.xml')
        if r['process'].findtext('pid')==pid}
    encoding=[(int(r['start'].text),int(r['start'].text)+int(r['duration'].text))
        for r in read_rows(directory/'application_intervals.xml') if r['process'].findtext('pid')==pid]
    others=[(int(r['start'].text),int(r['start'].text)+int(r['duration'].text))
        for r in gpu if r['process'].findtext('pid')!=pid and r['state'].get('fmt')=='Active']
    root=ET.parse(directory/'cpu_samples.xml').getroot()
    refs={e.attrib['id']:e for e in root.iter() if 'id' in e.attrib}
    def resolve(e):
        while 'ref' in e.attrib:
            e=refs[e.attrib['ref']]
        return e
    samples=[]
    window_lo=min(int(r['start'].text) for r in own)
    window_hi=max(int(r['start'].text)+int(r['duration'].text) for r in own)
    for r in read_rows(directory/'cpu_samples.xml'):
        if not r['thread'].get('fmt','').startswith('Main Thread') or r['process'].findtext('pid')!=pid:
            continue
        frames=[resolve(e).get('name','') for e in r['stack'].findall('frame')]
        if frames and window_lo<=int(r['time'].text)<window_hi:
            samples.append(dict(time_ns=int(r['time'].text),state=r['thread-state'].get('fmt'),
                category=stack_category(frames),frames=frames))
    epoch=datetime.fromisoformat(ET.parse(directory/'toc.xml').findtext('.//summary/start-date')).timestamp()
    steps=[json.loads(line) for line in (directory/'host_steps.jsonl').read_text().splitlines()]
    step_windows=[(r['step'],round(r['start_epoch_ns']-epoch*1e9),round(r['end_epoch_ns']-epoch*1e9)) for r in steps]
    gaps=[]
    for g in csv.DictReader((directory/'gaps.csv').open()):
        if g['scope']!='training_process':
            continue
        a,b=round(float(g['start_trace_s'])*1e9),round(float(g['end_trace_s'])*1e9)
        ss=[s for s in samples if a<=s['time_ns']<b]
        following=next((r for r in own if int(r['start'].text)>=b),None)
        sub=submissions.get(following['cmdbuffer-id'].text) if following is not None else None
        record=dict(start_ns=a,end_ns=b,gap_ms=(b-a)/1e6,
            encoding_overlap_ms=union_ns(encoding,a,b)/1e6,
            other_gpu_overlap_ms=union_ns(others,a,b)/1e6,
            overlapping_steps=[n for n,x,y in step_windows if x<b and y>a],
            crosses_step_boundary=any(a<=y<=b for _,x,y in step_windows),
            main_thread_stack_samples=len(ss),sample_categories=dict(Counter(s['category'] for s in ss)),
            main_thread_samples=ss)
        if sub is not None:
            start=int(sub['start'].text);end=start+int(sub['duration'].text)
            record.update(next_command_buffer_id=following['cmdbuffer-id'].get('fmt'),
                next_buffer_preparation_start_ns=start,next_buffer_preparation_end_ns=end,
                next_buffer_preparation_overlap_ms=max(0,min(end,b)-max(start,a))/1e6,
                next_buffer_end_to_gpu_ms=(b-end)/1e6,
                next_buffer_prepared_after_gap_start=end>a,
                gpu_reported_start_latency_ms=int(following['start-latency'].text)/1e6
                    if following['start-latency'].text is not None else None)
        gaps.append(record)
    gaps.sort(key=lambda g:g['gap_ms'],reverse=True)
    long=[g for g in gaps if g['gap_ms']>=1]
    big=[g for g in gaps if g['gap_ms']>=5]
    in_gap=lambda t:any(g['start_ns']<=t<g['end_ns'] for g in gaps)
    group_counts={key:dict(Counter(s['category'] for s in samples if in_gap(s['time_ns'])==flag))
        for key,flag in [('gap_samples',True),('outside_gap_samples',False)]}
    result=dict(pid=int(pid),main_thread_stack_samples=len(samples),
        sample_categories=dict(Counter(s['category'] for s in samples)),**group_counts,
        gaps_at_least_1ms=len(long),gaps_at_least_5ms=len(big),
        long_gaps_with_cpu_stacks=sum(g['main_thread_stack_samples']>0 for g in long),
        big_gaps_with_allocation_or_residency_samples=sum(any(
            'residency' in c or 'allocation' in c or 'creation' in c for c in g['sample_categories']) for g in big),
        long_gaps_matched_to_following_buffer=sum('next_command_buffer_id' in g for g in long),
        long_gaps_following_buffer_prepared_after_gap_start=sum(g.get('next_buffer_prepared_after_gap_start',False) for g in long),
        long_gaps_crossing_step_boundary=sum(g['crosses_step_boundary'] for g in long),
        long_gap_total_ms=sum(g['gap_ms'] for g in long),
        following_buffer_preparation_overlap_fraction=sum(g.get('next_buffer_preparation_overlap_ms',0) for g in long)/sum(g['gap_ms'] for g in long),
        following_buffer_end_to_gpu_median_ms=statistics.median(g['next_buffer_end_to_gpu_ms'] for g in long if 'next_buffer_end_to_gpu_ms' in g),
        matched_latency_max_disagreement_ms=max(abs(g['next_buffer_end_to_gpu_ms']-g['gpu_reported_start_latency_ms'])
            for g in long if g.get('gpu_reported_start_latency_ms') is not None),
        top_gaps=gaps[:20],
        caveats=['CPU samples are point observations, not intervals or exact time shares.',
            'Metal submission table spans command-buffer preparation; its endpoint is not a raw driver enqueue timestamp.',
            'End-to-execution deltas are correlated with the GPU table start-latency field as a consistency check.',
            'Same-capture correlation identifies the observed path, not an unprofiled causal effect or a proven speedup.'])
    (directory/'correlation.json').write_text(json.dumps(result,indent=2)+'\n')
    fields=['start_ns','end_ns','gap_ms','encoding_overlap_ms','other_gpu_overlap_ms',
        'next_command_buffer_id','next_buffer_preparation_overlap_ms','next_buffer_end_to_gpu_ms',
        'main_thread_stack_samples','sample_categories','overlapping_steps','crosses_step_boundary']
    with (directory/'correlated_gaps.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        writer.writeheader();writer.writerows(gaps)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # A boundary GPU event can lack a submission row when its buffer was
    # prepared before tracing began. Keep that gap in metrics, but use a
    # matched buffer for the explanatory detail figure.
    top=next((g for g in gaps if 'next_buffer_preparation_start_ns' in g),gaps[0])
    lo=top['start_ns']-5_000_000;hi=top['end_ns']+5_000_000
    fig,ax=plt.subplots(figsize=(12,5),layout='constrained')
    for y,intervals,color in [(3,[(int(r['start'].text),int(r['start'].text)+int(r['duration'].text)) for r in own],'#167c91'),
            (2,others,'#8055a4'),(1,[(top['next_buffer_preparation_start_ns'],top['next_buffer_preparation_end_ns'])]
                if 'next_buffer_preparation_start_ns' in top else [],'#d68a27')]:
        ax.broken_barh([((a-lo)/1e6,(b-a)/1e6) for a,b in merged(intervals,lo,hi)],(y-.25,.5),facecolors=color)
    for s in samples:
        if lo<=s['time_ns']<=hi:
            color='#be3f43' if any(word in s['category'] for word in ('residency','creation','allocation')) else '#217a49'
            x=(s['time_ns']-lo)/1e6
            ax.scatter(x,0,c=color,s=45,zorder=4)
            label={'buffer residency update':'residency update','Metal buffer creation':'buffer creation',
                'kernel command encoding':'kernel encoding'}.get(s['category'],s['category'])
            ax.annotate(label,(x,0),xytext=(0,-20),textcoords='offset points',
                rotation=30,ha='left' if x<5 else 'center',fontsize=8)
    ax.axvspan((top['start_ns']-lo)/1e6,(top['end_ns']-lo)/1e6,color='#d6d6d6',alpha=.25)
    ax.set_yticks([0,1,2,3],['Main-thread stack samples','Next buffer preparation','Other GPU processes','Training GPU execution'])
    ax.set_xlim(0,(hi-lo)/1e6);ax.set_ylim(-1.2,3.6)
    ax.set_xlabel('Milliseconds within the selected detail window')
    ax.set_title('Largest matched training GPU gap: CPU preparation and GPU execution')
    ax.grid(axis='x',alpha=.2)
    fig.savefig(directory/'correlation.png',dpi=160);plt.close(fig)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    result=correlate(parser.parse_args().directory)
    print(json.dumps({key:value for key,value in result.items() if key!='top_gaps'},indent=2))
