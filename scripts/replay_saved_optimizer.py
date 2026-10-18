"""Compare CPU AdamW implementations with identical saved 50M moments/gradients.

This tests CPU arithmetic only, not CUDA/XLA kernels or historical TPU updates.
"""
import copy
from dataclasses import fields
import gc
import hashlib
import json
import importlib
import math
from pathlib import Path
import sys
import torch
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from telos.models import TelosConfig,TelosTransformer
from scripts.compare_saved_50m_gradients import sha256


def build_optimizer(checkpoint,names,fused):
    state=checkpoint['model_state_dict']
    parameters={name:torch.nn.Parameter(state[name].clone()) for group in names for name in group}
    saved=copy.deepcopy(checkpoint['optimizer_state_dict'])
    groups=[]
    for keys,group in zip(names,saved['param_groups']):
        groups.append({'params':[parameters[name] for name in keys]})
        for name,index in zip(keys,group['params']):
            if tuple(saved['state'][index]['exp_avg'].shape)!=tuple(parameters[name].shape):
                raise ValueError('Saved optimizer mapping disagrees with parameter shape')
    optimizer=torch.optim.AdamW(groups,lr=1e-4,fused=fused,foreach=False)
    optimizer.load_state_dict(saved)
    for group in optimizer.param_groups:
        group.update(fused=fused,foreach=False,capturable=False,differentiable=False)
    return parameters,optimizer


@torch.no_grad()
def compare_tensors(a,b,initial=None):
    maximum=0.; squared=0.; reference=0.;updates=0.;changed=0;count=0
    for name,x in a.items():
        y=b[name];delta=x.double()-y.double()
        maximum=max(maximum,float(delta.abs().max()));squared+=float(delta.square().sum())
        reference+=float(x.double().square().sum());changed+=int((x!=y).sum());count+=x.numel()
        if initial is not None:updates+=float((x.double()-initial[name].double()).square().sum())
    return {'max_absolute':maximum,'relative_l2_vs_tensor':math.sqrt(squared/max(reference,1e-300)),
            'relative_l2_vs_plain_cumulative_update':math.sqrt(squared/max(updates,1e-300)) if initial is not None else None,
            'unequal_elements':changed,'elements':count}


def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    checkpoint_path=ROOT/'checkpoints/corosred/50m_lightning/checkpoint_final.pt'
    fixture_path=ROOT/'scratch/backend_parity/fixture.pt'
    capture_path=ROOT/'scratch/backend_parity/cpu_fp32.pt'
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True,mmap=True)
    fixture=torch.load(fixture_path,map_location='cpu',weights_only=True,mmap=True)
    capture=torch.load(capture_path,map_location='cpu',weights_only=True,mmap=True)
    if fixture['checkpoint_sha256']!=sha256(checkpoint_path) or capture['metadata']['fixture_sha256']!=sha256(fixture_path):
        raise ValueError('Capture/checkpoint provenance mismatch')
    keys={f.name for f in fields(TelosConfig)}
    model=TelosTransformer(TelosConfig(**{k:v for k,v in checkpoint['config']['model'].items() if k in keys}))
    names=[[],[]]
    for name,p in model.named_parameters():
        names[1 if p.ndim==1 or 'embed' in name else 0].append(name)
    del model;gc.collect()
    if [len(g) for g in names]!=[len(g['params']) for g in checkpoint['optimizer_state_dict']['param_groups']]:
        raise ValueError('Optimizer parameter group sizes differ')
    gradients={name:capture['tensors']['gradient/'+name] for group in names for name in group}
    ordinary,opt_a=build_optimizer(checkpoint,names,False)
    fused,opt_b=build_optimizer(checkpoint,names,True)
    for name in ordinary:
        if ordinary[name].data_ptr()==fused[name].data_ptr():
            raise ValueError('Replay parameters alias each other')
        for key in ['exp_avg','exp_avg_sq','step']:
            if opt_a.state[ordinary[name]][key].data_ptr()==opt_b.state[fused[name]][key].data_ptr():
                raise ValueError('Replay optimizer states alias each other')
    initial=checkpoint['model_state_dict']
    report={'device':'cpu','torch':str(torch.__version__),'physical_cuda_xla_comparison':False,
            'checkpoint_sha256':sha256(checkpoint_path),'capture_sha256':sha256(capture_path),
            'capture_metadata':capture['metadata'],'source_sha256':sha256(__file__),
            'optimizer_state':'Saved CUDA final moments at step 12716, cloned identically for both CPU modes',
            'mapping_caveat':'Parameter group sizes and every moment shape verified against current named-parameter order; checkpoint has no saved parameter-name manifest',
            'gradient_stream':'Same captured FP32 gradients, repeated with prescribed sign/scale changes; not a continuation of training',
            'steps':[]}
    module=importlib.import_module(torch.optim.AdamW.step.__module__)
    report['dispatch_counts']={'ordinary':0,'fused':0}
    for index,scale in enumerate([1.,.5,1.5,-.25,1.,.75,-.5,.01],1):
        norms=[]
        for parameters,optimizer in [(ordinary,opt_a),(fused,opt_b)]:
            for name,p in parameters.items():p.grad=gradients[name].clone().mul_(scale)
            norm=torch.nn.utils.clip_grad_norm_(list(parameters.values()),1.,foreach=False)
            norms.append(float(norm))
            kind='fused' if optimizer is opt_b else 'ordinary'
            prefix='_fused_' if kind=='fused' else '_single_tensor_'
            function=next(name for name in [prefix+'adamw',prefix+'adam'] if hasattr(module,name))
            with patch.object(module,function,wraps=getattr(module,function)) as call:
                optimizer.step()
                report['dispatch_counts'][kind]+=call.call_count
            optimizer.zero_grad(set_to_none=True)
        if norms[0]!=norms[1]:raise ValueError('Clipping inputs differed')
        row={'step':index,'prescribed_gradient_scale':scale,'preclip_norm':norms[0],
             'clip_coefficient':min(1.,1./(norms[0]+1e-6)),
             'parameters':compare_tensors(ordinary,fused,initial),
             'exp_avg':compare_tensors({n:opt_a.state[p]['exp_avg'] for n,p in ordinary.items()},{n:opt_b.state[p]['exp_avg'] for n,p in fused.items()}),
             'exp_avg_sq':compare_tensors({n:opt_a.state[p]['exp_avg_sq'] for n,p in ordinary.items()},{n:opt_b.state[p]['exp_avg_sq'] for n,p in fused.items()})}
        report['steps'].append(row)
        print(json.dumps(row),flush=True)
    path=ROOT/'logs/saved_optimizer_replay_20261018.json';path.write_text(json.dumps(report,indent=2));print(path)
if __name__=='__main__':main()
