"""Matched actual AR updates, alternating order each pair to reduce desktop drift.

Uses the production model, loss/gradient builder and MLXTrainingStep. It is an
update-kernel diagnostic, excluding setup/export and between-update reporting.
Two independent models coexist; it is not an end-to-end launcher benchmark.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np
import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from telos.models.mlx_transformer import MLXTelosTransformer
from telos.training.trainer_mlx import UnifiedMLXTrainer
from telos.training.mlx_step import MLXTrainingStep,build_mlx_accumulation_step
from telos.training.dataloader import get_global_targets_contiguous
import final_update
import pipeline


def workers():
    for line in subprocess.check_output(['ps','-axo','pid=,comm='],text=True).splitlines():
        pid,command=line.strip().split(None,1)
        if int(pid)!=os.getpid() and Path(command).name.startswith('python'):
            raise RuntimeError('Other Python worker: '+line)


def make(cfg):
    mx.random.seed(42)
    m=cfg['model']
    model=MLXTelosTransformer(vocab_size=m['vocab_size'],d_model=m['d_model'],n_layers=m['n_layers'],
                            n_heads=m['n_heads'],n_kv_heads=m['n_kv_heads'],is_causal=True)
    trainer=UnifiedMLXTrainer('ar',model,copy.deepcopy(cfg),eval_policy='step')
    function,targets=trainer._setup_training_functions();targets.append(mx.random.state)
    optimizer=optim.AdamW(4e-4,weight_decay=.01,betas=[.9,.95],bias_correction=True)
    step=MLXTrainingStep(model,optimizer,mx.compile(function,inputs=targets,outputs=targets),
                         accumulation_step=build_mlx_accumulation_step(function,targets))
    mx.eval(model.parameters())
    return trainer,step,optimizer


def compare(a,b):
    aa=dict(tree_flatten(a));bb=dict(tree_flatten(b));assert aa.keys()==bb.keys()
    maximum=0.;square=denominator=0.;exact=True
    for key in aa:
        x,y=aa[key],bb[key];assert x.dtype==y.dtype and x.shape==y.shape
        xf,yf=x.astype(mx.float32),y.astype(mx.float32)
        assert mx.all(mx.isfinite(xf)).item() and mx.all(mx.isfinite(yf)).item()
        maximum=max(maximum,mx.max(mx.abs(xf-yf)).item())
        square+=mx.sum((xf-yf)**2).item();denominator+=mx.sum(xf**2).item()
        exact &= mx.array_equal(x,y).item()
    return dict(max_abs=maximum,relative_l2=(square/denominator)**.5 if denominator else 0.,exact=exact)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=['final','pipeline','micro16','micro32'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--updates',type=int,default=64)
    parser.add_argument('--warmup',type=int,default=12)
    args=parser.parse_args();workers()
    args.output.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(Path('reports/mlx_pretrain_2026-11-16/pipeline_abba/00_baseline/checkpoints/config.json').read_text())
    cfg['training']['force_fused_attention']=True
    mx.set_memory_limit(int(4.5*1024**3));mx.set_cache_limit(64*1024**2)
    base=make(cfg)
    candidate_cfg=copy.deepcopy(cfg)
    candidate_microbatch=int(args.variant.removeprefix('micro')) if args.variant.startswith('micro') else 8
    candidate_accumulation=64//candidate_microbatch
    candidate_cfg['training']['batch_size']=candidate_microbatch
    candidate_cfg['training']['gradient_accumulation']=candidate_accumulation
    candidate=make(candidate_cfg)
    corpus=Path('/Users/ivansamuel/Documents/Projects/telos/data/english_corpus_5b.bin')
    dtype=np.uint16
    for meta in (Path(str(corpus)+'.json'),corpus.with_suffix('.json')):
        if meta.exists():
            dtype=np.int32 if json.loads(meta.read_text()).get('dtype')=='int32' else np.uint16
            break
    data=np.memmap(corpus,dtype=dtype,mode='r');data=data[:len(data)//256*256].reshape(-1,256)
    assert data[:args.updates*64].max()<8192
    ptr=0;rows=[]
    from scripts.profile_mlx_training import system_sample
    start_system=system_sample()
    for count in range(1,args.updates+1):
        workers()
        batch,ptr=get_global_targets_contiguous(data,ptr,64,256)
        lr=base[0]._get_lr(count,50,args.updates,4e-4,4e-5)
        base[2].learning_rate=candidate[2].learning_rate=lr
        record=dict(step=count,lr=lr)
        for index in ((0,1) if count%2 else (1,0)):
            trainer,step,optimizer=(base,candidate)[index]
            microbatch=candidate_microbatch if index else 8
            accumulation=candidate_accumulation if index else 8
            batches=(mx.array(batch[i*microbatch:(i+1)*microbatch],dtype=mx.int32) for i in range(accumulation))
            if index and args.variant=='pipeline':pipeline.install()
            else:pipeline.uninstall()
            mx.synchronize();started=time.perf_counter()
            try:
                result=final_update.call(step,batches,grad_accum=accumulation) if index and args.variant=='final' else step(batches,grad_accum=accumulation)
                mx.synchronize()
            finally:pipeline.uninstall()
            record['candidate_seconds' if index else 'baseline_seconds']=time.perf_counter()-started
            record['candidate_loss' if index else 'baseline_loss']=result[0].item()/accumulation
        record['peak_mlx_bytes']=mx.get_peak_memory()
        if count%8==0:
            record['system']=system_sample()
            if record['system']['swap_free_mb']<768 or record['system']['memory_free_percent']<25:
                raise RuntimeError('System headroom guard triggered; no valid comparison')
        if record['peak_mlx_bytes']>4.5*1024**3:raise RuntimeError('MLX allocation guard triggered')
        rows.append(record)
        with (args.output/'steps.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
        if count%16==0:print(count,flush=True)
    workers()
    warmed=rows[args.warmup:]
    a=sum(r['baseline_seconds'] for r in warmed);b=sum(r['candidate_seconds'] for r in warmed)
    states={}
    for group,x,y in [('model',base[1].model.parameters(),candidate[1].model.parameters()),
                      ('master',base[2].master_params,candidate[2].master_params),
                      ('optimizer',base[2].state,candidate[2].state)]:
        states[group]=compare(x,y)
    from telos.training.checkpoint import save_mlx_state
    for name,(_,step,optimizer) in zip(('baseline','candidate'),(base,candidate)):
        save_mlx_state(args.output/(name+'.npz'),dict(model=step.model.parameters(),master=optimizer.master_params,optimizer=optimizer.state))
    root=Path.cwd()
    manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for pattern in ('telos/models/*.py','telos/training/*.py','scratch/mlx_pretrain_20261116/*.py') for p in root.glob(pattern)}
    report=dict(variant=args.variant,updates=args.updates,warmup=args.warmup,candidate_microbatch=candidate_microbatch,candidate_accumulation=candidate_accumulation,
                baseline_tokens_per_second=len(warmed)*16384/a,candidate_tokens_per_second=len(warmed)*16384/b,
                pooled_speedup_percent=100*(a/b-1),median_paired_speedup_percent=statistics.median(100*(r['baseline_seconds']/r['candidate_seconds']-1) for r in warmed),
                states=states,initial_system=start_system,final_system=system_sample(),source_manifest=manifest,
                limitations=['Two resident models; setup/export/reporting excluded.','Alternating pair order reduces desktop drift but does not guarantee exclusive GPU access.','AR-specific check; no stochastic-objective or interrupted-process resume result from this run.'])
    (args.output/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='source_manifest'},indent=2))


if __name__=='__main__':main()
