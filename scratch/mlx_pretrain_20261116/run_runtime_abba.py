"""Compare isolated MLX runtimes at identical long-context AR shapes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--updates',type=int,default=32)
parser.add_argument('--warmup',type=int,default=8)
parser.add_argument('--seq-len',type=int,default=1024)
parser.add_argument('--microbatch',type=int,default=2)
parser.add_argument('--output',type=Path,default=Path('reports/mlx_pretrain_2026-11-16/runtime_long_abba'))
parser.add_argument('--resume',action='store_true',help='Continue completed fresh-process comparison trials')
args=parser.parse_args()
if not 0<=args.warmup<args.updates:parser.error('Invalid warmup')
root=Path(__file__).resolve().parents[2]
output=args.output.resolve()
output.mkdir(parents=True,exist_ok=args.resume)
manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
          for pattern in ('telos/models/*.py','telos/training/*.py','scripts/profile_mlx_training.py') for p in root.glob(pattern)}
if args.resume:
    assert json.loads((output/'source_manifest.json').read_text())==manifest, 'Source changed during comparison'
else:
    (output/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
(output/'power.txt').write_text(subprocess.check_output(['pmset','-g','custom'],text=True))
results=json.loads((output/'results.json').read_text()) if args.resume else []
for old in results:
    assert old['valid'] and old.get('warmup')==args.warmup
    arguments=old['metadata']['arguments']
    assert (arguments['tokens'],arguments['seq_len'],arguments['microbatch'])==(args.updates*args.microbatch*8*args.seq_len,args.seq_len,args.microbatch)
for index,runtime in enumerate(('0.32.2','0.32.4','0.32.4','0.32.2')):
    if index<len(results):
        assert results[index]['runtime']==runtime
        continue
    for line in subprocess.check_output(['ps','-axo','pid=,comm='],text=True).splitlines():
        pid,command=line.strip().split(None,1)
        if '/uv/tools/google-colab-cli/bin/' in command:
            continue  # Verified remote Colab control CLI, not a local MLX worker.
        if int(pid)!=os.getpid() and Path(command).name.startswith('python'):
            raise RuntimeError('Another Python worker is active: '+line)
    trial=output/f'{index:02d}_{runtime}'
    env={**os.environ,'PYTHONPATH':(str(Path('/private/tmp/telos-mlx-0.32.4'))+os.pathsep if runtime=='0.32.4' else '')+str(root),
         'MLX_MAX_OPS_PER_BUFFER':'16','MLX_MAX_MB_PER_BUFFER':'16'}
    command=[sys.executable,str(root/'scripts/profile_mlx_training.py'),'--output',str(trial),
             '--tokens',str(args.updates*args.microbatch*8*args.seq_len),'--seq-len',str(args.seq_len),'--microbatch',str(args.microbatch),'--grad-accum','8',
             '--data','/Users/ivansamuel/Documents/Projects/telos/data/english_corpus_5b.bin',
             '--max-mlx-gib','3.2','--critical-memory-free-percent','25','--critical-swap-free-mib','768',
             '--experimental-qkv-rope','--experimental-packed-swiglu','--force-fused-attention','--compiled-accumulation']
    with (output/f'{index:02d}_{runtime}.log').open('w') as stream:
        process=subprocess.run(command,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=300)
    summary=json.loads((trial/'summary.json').read_text()) if (trial/'summary.json').exists() else {}
    metadata=json.loads((trial/'metadata.json').read_text()) if (trial/'metadata.json').exists() else {}
    rows=[json.loads(x) for x in (trial/'steps.jsonl').read_text().splitlines()] if (trial/'steps.jsonl').exists() else []
    valid=process.returncode==0 and summary.get('status')=='complete' and len(rows)==args.updates and metadata.get('mlx_version')==runtime
    warmed=rows[args.warmup:]
    result=dict(runtime=runtime,valid=valid,returncode=process.returncode,summary=summary,metadata=metadata,warmup=args.warmup,
                warmed_tokens_per_second=len(warmed)*args.microbatch*8*args.seq_len/sum(r['training_step_seconds'] for r in warmed) if valid else None,
                peak_mlx_bytes=max((r['mlx_peak_bytes'] for r in rows),default=0))
    results.append(result);(output/'results.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(result),flush=True)
    if not valid:raise RuntimeError('Runtime trial invalid; remaining runs cancelled')
