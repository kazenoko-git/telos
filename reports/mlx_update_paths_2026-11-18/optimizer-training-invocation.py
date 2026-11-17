from pathlib import Path
import subprocess,os,json,sys,time
root=Path.cwd();sys.path.insert(0,str(root));from scripts.mlx_hardware_monitor import HardwareMonitor
python=str(root/'.venv/bin/python');out=root/'logs/mlx_next_20261118';monitor=HardwareMonitor()
for i,source in enumerate([root,Path('/tmp/telos-optimizer-candidate-20261118'),Path('/tmp/telos-optimizer-candidate-20261118'),root]):
 target=out/f'optimizer-training-{i}'
 if (target/'summary.json').exists():
  assert json.loads((target/'summary.json').read_text())['status']=='complete'
  print('PRESERVED',i,flush=True);continue
 cmd=[python,str(source/'scripts/profile_mlx_training.py'),'--tokens',str(64*16384),'--data','/Users/ivansamuel/Documents/Projects/telos-mlx-fusion/reports/mlx_fusion_2026-11-16/corpus_prefix.bin','--output',str(target),'--max-mlx-gib','3.2','--critical-swap-growth-mib','256','--swap-growth-baseline-updates','16','--async-accumulation','--compiled-accumulation']
 print('START',i,str(source),flush=True)
 with (out/f'optimizer-training-{i}.log').open('w') as log,(out/f'optimizer-training-{i}-hardware.jsonl').open('w',buffering=1) as hw:
  p=subprocess.Popen(cmd,cwd=root,env={**os.environ,'PYTHONPATH':str(source)},stdout=log,stderr=subprocess.STDOUT)
  while p.poll() is None:
   sample=monitor.sample();hw.write(json.dumps(sample)+'\n')
   if sample['thermal_state']>=2:p.terminate();p.wait();raise RuntimeError('Serious thermal state')
   time.sleep(1)
 s=json.loads((target/'summary.json').read_text());assert p.returncode==0 and s['status']=='complete' and s['updates']==64
 print('DONE',i,flush=True)
print('ALL_DONE',flush=True)
