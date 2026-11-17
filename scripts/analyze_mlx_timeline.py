"""Summarize recorded Metal execution gaps; never equate them with occupancy."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET


def read_rows(path):
    root = ET.parse(path).getroot()
    refs = {e.attrib['id']: e for e in root.iter() if 'id' in e.attrib}
    columns = [e.findtext('mnemonic') for e in root.find('.//schema').findall('col')]
    def resolve(e):
        while 'ref' in e.attrib:
            e = refs[e.attrib['ref']]
        return e
    return [{key: resolve(e) for key, e in zip(columns, row)} for row in root.iter('row')]


def merged(intervals, lo, hi):
    result = []
    for start, end in sorted((max(lo,a),min(hi,b)) for a,b in intervals if b>lo and a<hi):
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1],end)
        else:
            result.append([start,end])
    return result


def metrics(intervals, lo, hi):
    union = merged(intervals,lo,hi)
    gaps=[]
    cursor=lo
    for start,end in union:
        if start>cursor:
            gaps.append((cursor,start))
        cursor=end
    if cursor<hi:
        gaps.append((cursor,hi))
    values=sorted((b-a)/1e6 for a,b in gaps)
    return dict(recorded_active_fraction=sum(b-a for a,b in union)/(hi-lo),
        recorded_gap_ms=sum(values),gap_count=len(values),
        gap_median_ms=statistics.median(values) if values else 0,
        gap_p95_ms=values[min(len(values)-1,int(.95*(len(values)-1)))] if values else 0,
        gap_max_ms=max(values,default=0),
        gaps_at_least_ms={str(t):sum(v>=t for v in values) for t in (.1,1,5,10)}),gaps


def analyze(directory):
    capture=json.loads((directory/'capture.json').read_text())
    pid=capture['worker_pid']
    rows=read_rows(directory/'gpu_intervals.xml')
    events=[]
    for row in rows:
        if row['state'].get('fmt')!='Active':
            continue
        start=int(row['start'].text)
        proc=row['process']
        events.append(dict(start=start,end=start+int(row['duration'].text),
            process=proc.get('fmt','unattributed'),
            channel=row['channel-name'].get('fmt'),
            pid=proc.findtext('pid')))
    own=[(e['start'],e['end']) for e in events if e['pid']==str(pid)]
    if not own:
        raise ValueError('No process-attributed GPU execution events; idle fraction unavailable')
    lo,hi=min(a for a,b in own),max(b for a,b in own)
    own_stats,own_gaps=metrics(own,lo,hi)
    all_stats,all_gaps=metrics([(e['start'],e['end']) for e in events],lo,hi)
    toc=ET.parse(directory/'toc.xml').getroot()
    epoch=datetime.fromisoformat(toc.findtext('.//summary/start-date')).timestamp()
    steps=[json.loads(line) for line in (directory/'host_steps.jsonl').read_text().splitlines()]
    overlapping=[r['step'] for r in steps if r['start_epoch_ns']<(epoch*1e9+hi)
        and r['end_epoch_ns']>(epoch*1e9+lo)]
    result=dict(pid=pid,window_start_s=lo/1e9,window_end_s=hi/1e9,
        window_seconds=(hi-lo)/1e9,overlapping_steps=overlapping,
        process_gpu_events=len(own),all_gpu_events=len(events),
        process_execution=own_stats,all_recorded_execution=all_stats,
        shader_timeline_enabled=any('Shader Timeline: Enabled' in (e.text or '') for e in toc.iter('value')),
        hardware_counters_enabled=any('Counter Set:' in (e.text or '') and '(null)' not in e.text
            for e in toc.iter('value')),
        caveats=['Window is first-to-last process GPU event, excluding trace startup/tail.',
            'Intervals are recorded GPU execution envelopes, not individual shader occupancy.',
            'Gaps mean absence of recorded execution, not proven total hardware inactivity.',
            'Other apps, capture overhead, and memory pressure affect this short run.',
            'Host steps overlap the window; partially covered steps are included.'])
    (directory/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    with (directory/'gaps.csv').open('w') as stream:
        writer=csv.writer(stream)
        writer.writerow(['scope','start_trace_s','end_trace_s','gap_ms'])
        for scope,gaps in [('training_process',own_gaps),('all_recorded_gpu',all_gaps)]:
            writer.writerows((scope,a/1e9,b/1e9,(b-a)/1e6) for a,b in gaps)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(12,5),layout='constrained')
    # Zoom around the largest process gap; show two independent execution lanes.
    gap=max(own_gaps,key=lambda pair:pair[1]-pair[0],default=(lo,lo))
    center=sum(gap)/2
    zoom_lo=max(lo,center-25e6); zoom_hi=min(hi,center+25e6)
    for ax,left,right,title in [(axes[0],lo,hi,'Captured GPU execution'),
            (axes[1],zoom_lo,zoom_hi,'50 ms detail around the largest training-process gap')]:
        for y,intervals,color in [(1,own,'#167c91'),(0,[(e['start'],e['end']) for e in events],'#8055a4')]:
            segments=merged(intervals,left,right)
            ax.broken_barh([((a-lo)/1e6,(b-a)/1e6) for a,b in segments],(y-.25,.5),facecolors=color)
        ax.set_xlim((left-lo)/1e6,(right-lo)/1e6)
        ax.set_ylim(-.6,1.6); ax.set_yticks([0,1],['All recorded GPU work','Training process'])
        ax.set_xlabel('Milliseconds from first training GPU event');ax.set_title(title)
        ax.grid(axis='x',alpha=.2)
    fig.savefig(directory/'timeline.png',dpi=160)
    plt.close(fig)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    print(json.dumps(analyze(parser.parse_args().directory),indent=2))
