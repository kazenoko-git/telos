"""Run telemetry-monitored, untraced cache or bounded-accumulation comparisons."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['cache','async'],required=True)
    p.add_argument('--updates',type=int,default=64)
    p.add_argument('--warmup-updates',type=int,default=16)
    p.add_argument('--order',help='Comma-separated cache MiB values or sync/async variants')
    p.add_argument('--save-training-state',action='store_true')
    args=p.parse_args()
    if not 0<args.warmup_updates<args.updates:p.error('Warmup must be positive and shorter than run')
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    scripts=Path(__file__).resolve().parent
    manifest=dict(source=str(args.source),data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
        arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        source_sha256={str(f.relative_to(args.source)):hashlib.sha256(f.read_bytes()).hexdigest()
            for f in (args.source/'telos').rglob('*.py')},
        harness_sha256={name:hashlib.sha256((scripts/name).read_bytes()).hexdigest() for name in
            ('capture_mlx_timeline.py','mlx_hardware_monitor.py','run_mlx_followup.py')})
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    variants=[128,256,256,128,256,128,128,256] if args.mode=='cache' else [False,True,True,False,True,False,False,True]
    if args.order:
        tokens=args.order.split(',')
        if args.mode=='async':
            if any(v not in ('sync','async') for v in tokens):p.error('Async order accepts sync/async')
            variants=[v=='async' for v in tokens]
        else:
            variants=[int(v) for v in tokens]
            if any(v not in (128,256,512) for v in variants):p.error('Invalid cache size')
    results=[]
    for index,variant in enumerate(variants):
        cache=variant if args.mode=='cache' else 256
        asynchronous=bool(variant) if args.mode=='async' else False
        trial=root/f'{index:02d}_{variant}'
        command=[sys.executable,str(scripts/'capture_mlx_timeline.py'),'--source',str(args.source),
            '--data',str(args.data),'--output',str(trial),'--updates',str(args.updates),
            '--cache-mib',str(cache),'--no-trace','--hardware-monitor',
            '--warmup',str(args.warmup_updates),'--timeout-seconds',str(max(150,args.updates*1.5+60))]
        if asynchronous:command.append('--async-accumulation')
        if args.save_training_state:command.append('--save-training-state')
        print(json.dumps(dict(event='starting',index=index,cache_mib=cache,async_accumulation=asynchronous)),flush=True)
        with (root/f'{index:02d}.log').open('w') as log:
            proc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=max(240,args.updates*1.5+120))
        capture=json.loads((trial/'capture.json').read_text())
        valid=(proc.returncode==0 and capture.get('worker_returncode')==0
            and capture.get('training_summary',{}).get('status')=='complete')
        rows=[json.loads(x) for x in (trial/'host_steps.jsonl').read_text().splitlines()] if (trial/'host_steps.jsonl').exists() else []
        warm=rows[args.warmup_updates:]
        elapsed=sum((r['end_monotonic_ns']-r['start_monotonic_ns'])/1e9 for r in warm)
        result=dict(index=index,cache_mib=cache,async_accumulation=asynchronous,valid=valid,
            updates=len(rows),warm_updates=len(warm),warmed_seconds=elapsed,
            tokens_per_second=len(warm)*16384/elapsed if elapsed else None,capture_status=capture['status'],directory=str(trial))
        results.append(result);(root/'results.json').write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps(result),flush=True)
        if not valid:raise RuntimeError('Incomplete trial; stopping and retaining partial evidence')
    print('FOLLOWUP_COMPLETE',flush=True)


if __name__=='__main__':main()
