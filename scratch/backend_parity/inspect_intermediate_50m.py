"""Inspect pinned HF checkpoints locally; evaluate the prior fixed CPU fixture."""
import sys,json,time,gc
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from scripts.compare_saved_50m_gradients import compare_checkpoint,sha256
ROOT=Path(__file__).resolve().parents[2]
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
inventory=json.loads((ROOT/'scratch/backend_parity/hf_50m_inventory.json').read_text())
remote={x['path']:x for x in inventory['files']}
fixture_path=ROOT/'logs/saved_50m_gradient_comparison_20261018.fixture.pt'
f=torch.load(fixture_path,weights_only=True,map_location='cpu')
out=ROOT/'logs/intermediate_50m_comparison_20261018.json'
report={'hf_repo':inventory['repo'],'hf_revision':inventory['revision'],'fixture':str(fixture_path),'fixture_sha256':sha256(fixture_path),'device':'cpu','compute_dtype':'float32','threads':2,'note':'Same common evaluator as endpoint audit; nearby steps are not identical and this does not identify backend causality. No historical objective or optimizer replay.','models':{}}
selected=[('cuda', 'checkpoints/corosred/50m_lightning',s) for s in [1271,3813,7626,11439]]+ [('tpu','checkpoints/corosred/unified/50m_python',s) for s in [4000,8000,12000]]
for label,folder,step in selected:
 relative=f'{folder}/checkpoint_step_{step}.pt'
 path=ROOT/relative
 digest=sha256(path)
 if digest!=remote[relative]['sha256']:
  raise ValueError(f'Local checkpoint differs from pinned HF revision: {relative}')
 print(f'Start {label} step {step}; HF SHA-256 verified',flush=True)
 t=time.monotonic()
 result=compare_checkpoint(path,f['tokens'],f['infill_mask'],.5,.35)
 payload=torch.load(path,map_location='cpu',mmap=True,weights_only=True)
 result['saved_keys']=list(payload)
 result['scheduler_state']=payload.get('scheduler_state_dict')
 result['lrh_acc_ema']=payload.get('lrh_acc_ema')
 result['lrh_auc_ema']=payload.get('lrh_auc_ema')
 result['optimizer_present']='optimizer_state_dict' in payload
 if result['optimizer_present']:
  opt=payload['optimizer_state_dict']
  result['optimizer_param_groups']=[{k:v for k,v in g.items() if k!='params'} for g in opt['param_groups']]
  result['optimizer_steps']=sorted({float(x['step']) for x in opt['state'].values() if 'step' in x})
 result['seconds']=time.monotonic()-t
 report['models'][f'{label}_{step}']=result
 temp=out.with_suffix('.tmp')
 temp.write_text(json.dumps(report,indent=2,default=lambda x: x.item() if isinstance(x,torch.Tensor) and x.numel()==1 else str(x)))
 temp.replace(out)
 g=result['gradient_geometry']['all_backbone']
 print(f'Done {label} {step}: C={result["causal"]["ce"]:.5f}, I={result["infill"]["ce"]:.5f}, cos={g["causal_infill_cosine"]:.4f}, {result["seconds"]:.1f}s',flush=True)
 del payload,result
 gc.collect()
print(out,flush=True)
