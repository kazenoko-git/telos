"""Real termination/restart on isolated MLX 0.32.4 with strict full-state comparison."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np


def load_cpu(path):
    archive=np.load(path,allow_pickle=False)
    structure=json.loads(bytes(archive['structure']).decode())
    def decode(node):
        if 'array' in node:return archive[node['array']]
        if 'dict' in node:return {key:decode(value) for key,value in node['dict']}
        if 'list' in node:return [decode(value) for value in node['list']]
        return node['value']
    return decode(structure)


def finite(value):
    if value.dtype.kind=='V' and value.dtype.itemsize==2:
        value=(value.view('<u2').astype('<u4')<<16).view('<f4')
    return np.isfinite(value).all()


def compare(a,b,path='$'):
    assert type(a)==type(b),(path,'type')
    if isinstance(a,np.ndarray):
        assert a.dtype==b.dtype and a.shape==b.shape,(path,'dtype/shape')
        assert finite(a) and finite(b),(path,'nonfinite')
        assert np.array_equal(a,b),(path,'values')
        return 1
    if isinstance(a,dict):
        assert a.keys()==b.keys(),(path,'keys')
        return sum(compare(a[k],b[k],path+'.'+k) for k in a)
    if isinstance(a,list):
        assert len(a)==len(b),(path,'length')
        return sum(compare(x,y,path+f'[{i}]') for i,(x,y) in enumerate(zip(a,b)))
    if isinstance(a,float):assert np.isfinite(a) and np.isfinite(b),(path,'nonfinite scalar')
    assert a==b,(path,a,b)
    return 0


def child(config,resume,paradigm):
    import mlx.core as mx
    assert mx.__version__=='0.32.4'
    assert Path(mx.__file__).is_relative_to('/private/tmp/telos-mlx-0.32.4')
    import telos
    print(json.dumps(dict(mlx_version=mx.__version__,mlx_module=mx.__file__,telos=telos.__file__)),flush=True)
    from telos.train.cli import train
    if resume:train(resume_checkpoint=resume,hardware='mlx')
    else:train(paradigm=paradigm,config_path=config,hardware='mlx',autotune=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child',action='store_true')
    parser.add_argument('--paradigm',choices=['ar','mdlm'],default='ar')
    parser.add_argument('--config',type=Path)
    parser.add_argument('--resume',type=Path)
    args=parser.parse_args()
    if args.child:return child(args.config,args.resume,args.paradigm)
    root=Path(__file__).resolve().parents[2]
    output=root/('reports/mlx_pretrain_2026-11-16/runtime_resume_'+args.paradigm)
    output.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((root/'reports/mlx_pretrain_2026-11-16/runtime_long_confirm/01_0.32.4/checkpoints/config.json').read_text())
    cfg['model'].update(d_model=128,n_layers=1,n_heads=2,n_kv_heads=2)
    cfg['training'].update(max_steps=12,batch_size=1,gradient_accumulation=2,warmup_steps=2,force_fused_attention=True)
    cfg['checkpoint'].update(checkpoint_dir=str(output/'run'),save_every_steps=2)
    config=output/'config.json';config.write_text(json.dumps(cfg,indent=2))
    env={**os.environ,'PYTHONPATH':'/private/tmp/telos-mlx-0.32.4'+os.pathsep+str(root),
         'MLX_MAX_OPS_PER_BUFFER':'16','MLX_MAX_MB_PER_BUFFER':'16'}
    command=[sys.executable,str(Path(__file__).resolve()),'--child','--config',str(config),'--paradigm',args.paradigm]
    with (output/'control.log').open('w') as stream:
        subprocess.run(command,env=env,cwd=root,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=90)
    (output/'run').rename(output/'control')
    with (output/'interrupted.log').open('w') as stream:
        process=subprocess.Popen(command,env=env,cwd=root,stdout=stream,stderr=subprocess.STDOUT)
        deadline=time.monotonic()+90
        while not (output/'run/checkpoint_step_2.resume.npz').exists():
            if process.poll() is not None:raise RuntimeError('Child exited before interruption checkpoint')
            if time.monotonic()>deadline:
                process.terminate();process.wait();raise RuntimeError('Checkpoint timeout')
            time.sleep(.005)
        assert process.poll() is None,'Child already finished; not a valid interruption'
        process.terminate();returncode=process.wait(timeout=15)
        assert returncode!=0,'Expected a terminated process'
    with (output/'resumed.log').open('w') as stream:
        subprocess.run(command+['--resume',str(output/'run/checkpoint_step_2.resume.npz')],env=env,cwd=root,
                       stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=90)
    a,b=[load_cpu(output/p/'checkpoint_final.resume.npz') for p in ('control','run')]
    count=compare(a,b)
    assert count>0 and a['global_step']==12
    result=dict(status='PASS',paradigm=a['config']['paradigm'],exact_full_state=True,tensor_leaves=count,terminated_returncode=returncode,
                global_step=12,seq_len=1024,d_model=128,layers=1,effective_batch=2,
                limitations=['Small-model real-corpus interruption check, not a 50M interrupted run or convergence result.'])
    (output/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
