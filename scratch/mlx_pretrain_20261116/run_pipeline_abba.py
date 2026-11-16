"""Fresh-process real-corpus A/B/B/A for bounded async accumulation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--updates',type=int,default=64)
    parser.add_argument('--warmup',type=int,default=12)
    args=parser.parse_args()
    if not 0<=args.warmup<args.updates:parser.error('Invalid warmup/update count')
    root=Path(__file__).resolve().parents[2]
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    env={**os.environ,'PYTHONPATH':str(root),'MLX_MAX_OPS_PER_BUFFER':'16','MLX_MAX_MB_PER_BUFFER':'16'}
    manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
              for pattern in ('telos/models/*.py','telos/training/*.py','scratch/mlx_pretrain_20261116/*.py','scripts/profile_mlx_training.py') for p in root.glob(pattern)}
    (args.output/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
    (args.output/'power.txt').write_text(subprocess.check_output(['pmset','-g','custom'],text=True))
    results=[]
    for index,variant in enumerate(('baseline','pipeline','pipeline','baseline')):
        processes=subprocess.check_output(['ps','-axo','pid=,comm='],text=True).splitlines()
        for line in processes:
            pid,command=line.strip().split(None,1)
            if int(pid)!=os.getpid() and Path(command).name.startswith('python'):
                raise RuntimeError('Another Python process is active; postpone this trial: '+line)
        trial=args.output/f'{index:02d}_{variant}'
        command=[sys.executable,str(root/'scratch/mlx_pretrain_20261116/run_profile.py'),
                 '--output',str(trial),'--tokens',str(args.updates*16384),
                 '--data','/Users/ivansamuel/Documents/Projects/telos/data/english_corpus_5b.bin',
                 '--max-mlx-gib','3.2','--critical-memory-free-percent','25','--critical-swap-free-mib','768',
                 '--experimental-qkv-rope','--experimental-packed-swiglu','--force-fused-attention','--compiled-accumulation']
        if variant=='pipeline':command+=['--pipeline']
        with (args.output/f'{index:02d}_{variant}.log').open('w') as stream:
            process=subprocess.run(command,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=240)
        summary=json.loads((trial/'summary.json').read_text()) if (trial/'summary.json').exists() else {}
        rows=[json.loads(line) for line in (trial/'steps.jsonl').read_text().splitlines()] if (trial/'steps.jsonl').exists() else []
        warmed=rows[args.warmup:]
        valid=process.returncode==0 and summary.get('status')=='complete' and len(rows)==args.updates
        result=dict(variant=variant,valid=valid,returncode=process.returncode,summary=summary,
                    warmup=args.warmup,completed_updates=len(rows),
                    warmed_tokens_per_second=len(warmed)*16384/sum(r['training_step_seconds'] for r in warmed) if valid else None,
                    peak_mlx_bytes=max((r['mlx_peak_bytes'] for r in rows),default=0))
        results.append(result);(args.output/'results.json').write_text(json.dumps(results,indent=2))
        print(json.dumps(result),flush=True)
        if not valid:raise RuntimeError('Trial failed; remaining comparisons cancelled')


if __name__=='__main__':main()
