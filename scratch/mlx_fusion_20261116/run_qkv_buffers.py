"""Sustained 2x2 QKV/RoPE and command-buffer comparisons on frozen source."""
import hashlib,json,os
from pathlib import Path
import subprocess,sys,time
root=Path.cwd();source=Path('/private/tmp/telos-fusion-frozen-f558146')
output=root/'reports/mlx_fusion_2026-11-16/qkv_buffers_prefix'
output.mkdir(parents=True,exist_ok=True)
manifest={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for directory in ('telos/models','telos/training') for p in (source/directory).glob('*.py')}
manifest['scripts/profile_mlx_training.py']=hashlib.sha256((source/'scripts/profile_mlx_training.py').read_bytes()).hexdigest()
(output/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
(output/'power.txt').write_text(subprocess.check_output(['pmset','-g','custom'],text=True))
variants=[('default_reference',False,False),('buffers16_reference',True,False),('buffers16_qkv',True,True),('default_qkv',False,True)]
results=json.loads((output/'results.json').read_text()) if (output/'results.json').exists() else []
if results and not results[-1]['valid']:
 failed=results.pop()
 with (output/'invalid_attempts.jsonl').open('a') as stream:stream.write(json.dumps(failed)+'\n')
for index,(name,buffers,qkv) in enumerate(variants+list(reversed(variants))):
 if index<len(results):continue
 for line in subprocess.check_output(['ps','-axo','pid=,comm='],text=True).splitlines():
  pid,command=line.strip().split(None,1)
  if int(pid)!=os.getpid() and Path(command).name.startswith('python') and '/uv/tools/google-colab-cli/bin/' not in command:
   raise RuntimeError('Other Python worker active: '+line)
 trial=output/f'{index:02d}_{name}'
 attempt=0
 while trial.exists():
  attempt+=1
  trial=output/f'{index:02d}_{name}_retry{attempt}'
 env={**os.environ,'PYTHONPATH':str(source),'MLX_ENABLE_TF32':'0'}
 for key in ('MLX_MAX_OPS_PER_BUFFER','MLX_MAX_MB_PER_BUFFER','MLX_METAL_FAST_SYNCH'):env.pop(key,None)
 if buffers:env.update(MLX_MAX_OPS_PER_BUFFER='16',MLX_MAX_MB_PER_BUFFER='16')
 command=[sys.executable,str(source/'scripts/profile_mlx_training.py'),'--output',str(trial),
          '--tokens',str(192*16384),'--microbatch','8','--grad-accum','8','--seq-len','256',
          '--data',str(root/'reports/mlx_fusion_2026-11-16/corpus_prefix.bin'),
          '--max-mlx-gib','3.2','--critical-memory-free-percent','25','--critical-swap-free-mib','256','--critical-swap-growth-mib','512',
          '--experimental-packed-swiglu','--force-fused-attention','--compiled-accumulation']
 if qkv:command+=['--experimental-qkv-rope']
 with (output/f'{index:02d}_{name}.log').open('w') as stream:
  process=subprocess.Popen(command,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT)
  started=time.monotonic();competition=[]
  while process.poll() is None:
   if time.monotonic()-started>300:
    process.terminate();process.wait();break
   for line in subprocess.check_output(['ps','-axo','pid=,comm='],text=True).splitlines():
    pid,worker=line.strip().split(None,1)
    if int(pid) not in (os.getpid(),process.pid) and Path(worker).name.startswith('python') and '/uv/tools/google-colab-cli/bin/' not in worker:
     competition.append(line.strip())
   if competition:
    process.terminate();process.wait();break
   time.sleep(1)
  process.wait()
 summary=json.loads((trial/'summary.json').read_text()) if (trial/'summary.json').exists() else {}
 metadata=json.loads((trial/'metadata.json').read_text()) if (trial/'metadata.json').exists() else {}
 rows=[json.loads(x) for x in (trial/'steps.jsonl').read_text().splitlines()] if (trial/'steps.jsonl').exists() else []
 valid=process.returncode==0 and summary.get('status')=='complete' and len(rows)==192 and metadata.get('mlx_version')=='0.32.2'
 assert not metadata or metadata['telos_module'].startswith(str(source))
 warm=rows[16:]
 result=dict(trial_dir=str(trial),competition=competition,name=name,buffers16=buffers,qkv=qkv,valid=valid,metadata=metadata,summary=summary,warmup=16,
             warmed_seconds=sum(r['training_step_seconds'] for r in warm),warmed_updates=len(warm),
             tokens_per_second=len(warm)*16384/sum(r['training_step_seconds'] for r in warm) if valid else None,
             peak_mlx_bytes=max((r['mlx_peak_bytes'] for r in rows),default=0))
 results.append(result);(output/'results.json').write_text(json.dumps(results,indent=2));print(json.dumps(result),flush=True)
 if not valid:raise RuntimeError('Trial invalid; cancelling subsequent runs')
assert len({r['metadata']['initial_weights_sha256'] for r in results})==1,'Initial weights differ'
