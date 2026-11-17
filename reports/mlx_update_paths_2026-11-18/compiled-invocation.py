import sys,json,subprocess,time,os,hashlib
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from scripts.mlx_hardware_monitor import HardwareMonitor
root=Path('logs/mlx_next_20261118'); monitor=HardwareMonitor()
data='/Users/ivansamuel/Documents/Projects/telos-mlx-fusion/reports/mlx_fusion_2026-11-16/corpus_prefix.bin'
plan={'updates':192,'warmup':32,'compiled_order':[False,True,True,False],'async':True,'microbatch':8,'accumulation':8,'cache_mib':256,'clock_lock':False,'data_sha256':hashlib.sha256(Path(data).read_bytes()).hexdigest(),'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
(root/'plan.json').write_text(json.dumps(plan,indent=2))
for i,compiled in enumerate(plan['compiled_order']):
 out=root/f'{i:02d}-compiled-{int(compiled)}'
 command=[sys.executable,'scripts/profile_mlx_training.py','--tokens',str(192*8*8*256),'--data',data,'--output',str(out),'--max-mlx-gib','3.2','--critical-swap-growth-mib','256','--swap-growth-baseline-updates','32','--async-accumulation','--save-initial-weights']
 if compiled: command.append('--compiled-accumulation')
 print('START',i,compiled,flush=True)
 with (root/f'{i:02d}.log').open('w') as log,(root/f'{i:02d}-hardware.jsonl').open('w',buffering=1) as hw:
  proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
  while proc.poll() is None:
   sample=monitor.sample();hw.write(json.dumps(sample)+'\n')
   if sample['thermal_state']>=2:
    proc.terminate();proc.wait();raise RuntimeError('Serious thermal state')
   time.sleep(1)
 summary=json.loads((out/'summary.json').read_text())
 print('FINISH',i,proc.returncode,summary,flush=True)
 if proc.returncode or summary['status']!='complete' or summary['updates']!=192:raise RuntimeError('Invalid run')
print('COMPLETE',flush=True)
