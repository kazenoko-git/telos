"""Bounded single-block Core ML/MLX forward and handoff comparison.

Run each --batch in a fresh process. Requires coremltools9 via PYTHONPATH.
No production model changes or hybrid transformer backward implementation.
"""
import argparse
from collections import Counter
import gc
import inspect
import json
from pathlib import Path
import resource
import statistics
import subprocess
import time

import coremltools as ct
from coremltools.converters.mil import Builder as mb
import mlx.core as mx
from mlx.utils import tree_flatten
import numpy as np

from scripts.profile_mlx_training import system_sample
from telos.models.mlx_components import MLXBlock

D, H, T, DH = 512, 8, 256, 64


def program_for(batch, weights, dynamic):
    def block(x, n1, n2, qkv, proj, gate_up, down):
        def rms(x, weight, prefix):
            z = mb.cast(x=x, dtype='fp32', name=prefix+'_float')
            square = mb.mul(x=z, y=z, name=prefix+'_square')
            mean = mb.reduce_mean(x=square, axes=[-1], keep_dims=True, name=prefix+'_mean')
            variance = mb.add(x=mean, y=np.float32(1e-5), name=prefix+'_epsilon')
            inv = mb.rsqrt(x=variance, name=prefix+'_rsqrt')
            norm = mb.mul(x=z, y=inv, name=prefix+'_normalize')
            weight = mb.cast(x=weight, dtype='fp32', name=prefix+'_weight')
            return mb.cast(x=mb.mul(x=norm, y=weight, name=prefix+'_scale'), dtype='fp16', name=prefix+'_half')

        def linear(x, w):
            if dynamic:
                flat=mb.reshape(x=x,shape=[batch*T,x.shape[-1]])
                out=mb.matmul(x=flat,y=mb.cast(x=w,dtype='fp16'))
                return mb.reshape(x=out,shape=[batch,T,w.shape[1]])
            return mb.linear(x=x, weight=w.astype(np.float16))

        x = mb.cast(x=x, dtype='fp16')
        q, k, v = mb.split(x=linear(rms(x, n1, 'rms1'), qkv), num_splits=3, axis=-1)
        def heads(x):
            return mb.transpose(x=mb.reshape(x=x, shape=[batch,T,H,DH]), perm=[0,2,1,3])
        q, k, v = heads(q), heads(k), heads(v)
        frequencies = np.arange(DH//2, dtype=np.float32) * (2.0 / DH)
        theta = np.arange(T, dtype=np.float32)[:,None] / np.power(np.float32(10000), frequencies)[None,:]
        cosine, sine = [a.reshape(1,1,T,DH//2,1).astype(np.float16) for a in (np.cos(theta), np.sin(theta))]
        def rope(x):
            pairs = mb.reshape(x=x, shape=[batch,H,T,DH//2,2])
            a,b = mb.split(x=pairs, num_splits=2, axis=-1)
            rotated = mb.concat(values=[mb.sub(x=mb.mul(x=a,y=cosine),y=mb.mul(x=b,y=sine)),
                                        mb.add(x=mb.mul(x=a,y=sine),y=mb.mul(x=b,y=cosine))],axis=-1)
            return mb.reshape(x=rotated,shape=[batch,H,T,DH])
        # Match the FP32 accumulation/softmax used by MLX fast attention rather
        # than forcing all attention intermediates into FP16.
        aq=mb.cast(x=rope(q),dtype='fp32',name='attention_q')
        ak=mb.cast(x=rope(k),dtype='fp32',name='attention_k')
        av=mb.cast(x=v,dtype='fp32',name='attention_v')
        scores=mb.matmul(x=aq,y=ak,transpose_y=True,name='attention_scores')
        scores=mb.mul(x=scores,y=np.float32(DH**-.5),name='attention_scale')
        causal=np.where(np.tril(np.ones((T,T),dtype=bool)),np.float32(0),np.float32(-10000))
        scores=mb.add(x=scores,y=causal,name='attention_causal')
        probability=mb.softmax(x=scores,axis=-1,name='attention_probability')
        attention=mb.matmul(x=probability,y=av,name='attention_values')
        attention=mb.cast(x=attention,dtype='fp16',name='attention_half')
        attention = mb.reshape(x=mb.transpose(x=attention,perm=[0,2,1,3]),shape=[batch,T,D])
        residual = mb.add(x=x,y=linear(attention,proj))
        gate, up = mb.split(x=linear(rms(residual,n2,'rms2'),gate_up),num_splits=2,axis=-1)
        mlp = mb.mul(x=mb.mul(x=gate,y=mb.sigmoid(x=gate)),y=up)
        output = mb.add(x=residual,y=linear(mlp,down))
        return mb.cast(x=output,dtype='fp32',name='output')

    shapes = [(batch,T,D)] + [w.shape for w in weights.values()]
    if dynamic:
        @mb.program(input_specs=[mb.TensorSpec(shape=s) for s in shapes],opset_version=ct.target.macOS15)
        def program(x,n1,n2,qkv,proj,gate_up,down):
            return block(x,n1,n2,qkv,proj,gate_up,down)
    else:
        @mb.program(input_specs=[mb.TensorSpec(shape=(batch,T,D))],opset_version=ct.target.macOS15)
        def program(x):
            return block(x,**weights)
    return program


def placement(model):
    from coremltools.models.compute_plan import MLComputePlan
    kw = {}
    if 'compute_units' in inspect.signature(MLComputePlan.load_from_path).parameters:
        kw['compute_units'] = ct.ComputeUnit.CPU_AND_NE
    plan = MLComputePlan.load_from_path(model.get_compiled_model_path(),**kw)
    rows=[]
    def visit(block):
        for op in block.operations:
            usage=plan.get_compute_device_usage_for_mlprogram_operation(op)
            if usage:
                rows.append(dict(op=op.operator_name,outputs=[v.name for v in op.outputs],
                                 preferred=type(usage.preferred_compute_device).__name__,
                                 supported=[type(d).__name__ for d in usage.supported_compute_devices]))
            elif op.operator_name!='const':
                rows.append(dict(op=op.operator_name,outputs=[v.name for v in op.outputs],
                                 preferred='unknown',supported=[]))
            for nested in op.blocks:visit(nested)
    for fn in plan.model_structure.program.functions.values():visit(fn.block)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',type=int,choices=[1,4,8],required=True)
    parser.add_argument('--steps',type=int,default=30)
    parser.add_argument('--warmup',type=int,default=10)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    initial=system_sample()
    mx.set_memory_limit(2*1024**3)
    mx.set_cache_limit(128*1024**2)
    mx.random.seed(42)
    model=MLXBlock(D,H,H,is_causal=True)
    model.set_dtype(mx.float16)
    mx.eval(model.parameters())
    raw=dict(tree_flatten(model.parameters()))
    mapping=dict(n1='norm1.weight',n2='norm2.weight',qkv='qkv_proj.weight',proj='out.weight',
                 gate_up='mlp.gate_up.weight',down='mlp.w3.weight')
    # FP32 master arrays represent the parameters handed over after GPU updates.
    masters={k:raw[v].astype(mx.float32) for k,v in mapping.items()}
    mx.eval(masters)
    weights={k:np.asarray(v) for k,v in masters.items()}
    x=np.random.default_rng(42).normal(0,.2,size=(args.batch,T,D)).astype(np.float32)
    gx=mx.array(x)
    gpu=mx.compile(lambda x:model(x.astype(mx.float16)).astype(mx.float32),inputs=[model.state])
    reference=np.asarray(gpu(gx)).copy()
    checkpoints=[]

    def check():
        sample=system_sample()
        checkpoints.append(sample)
        if (mx.get_peak_memory()>2*1024**3 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>2.5*1024**3
                or sample['memory_free_percent']<=12 or sample['swap_free_mb']<=512):
            raise RuntimeError('Block probe memory guard stopped trial')

    def timed(call):
        samples=[]
        for i in range(args.warmup+args.steps):
            begin=time.perf_counter();value=call();samples.append(time.perf_counter()-begin)
        check()
        return dict(median_ms=1000*statistics.median(samples[args.warmup:]),
                    seconds=samples[args.warmup:]),value

    def gpu_resident():
        y=gpu(gx);mx.eval(y);return y

    def gpu_roundtrip():
        y=gpu(mx.array(x));mx.eval(y);host=np.asarray(y)
        returned=mx.array(host);mx.eval(returned);return returned

    report=dict(batch=args.batch,context=T,d_model=D,heads=H,head_dim=DH,
                hidden=int(raw['mlp.w3.weight'].shape[1]),parameter_count=sum(v.size for v in raw.values()),
                warmup=args.warmup,steps=args.steps,
                precision='FP16 projections/MLP/RoPE; FP32 RMS/attention internals and I/O',
                source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                scope='Full block forward only; no transformer backward or optimizer',
                device=mx.device_info(),mlx=mx.__version__,coremltools=ct.__version__,initial_system=initial,trials=[])
    for dynamic in [False,True]:
        weight_feed={k:(v.T.copy() if v.ndim==2 else v) for k,v in weights.items()} if dynamic else weights
        program=program_for(args.batch,weight_feed,dynamic)
        begin=time.perf_counter()
        core=ct.convert(program,convert_to='mlprogram',minimum_deployment_target=ct.target.macOS15,
                        compute_precision=ct.transform.FP16ComputePrecision(op_selector=lambda op:not op.name.startswith(('rms1_','rms2_','attention_'))),
                        compute_units=ct.ComputeUnit.CPU_AND_NE)
        load=time.perf_counter()-begin
        check()
        plan=placement(core)
        feed=dict(x=x,**(weight_feed if dynamic else {}))
        y=np.asarray(next(iter(core.predict(feed).values())))
        err=y-reference
        entry=dict(dynamic_weights=dynamic,conversion_load_seconds=load,placement=plan,
                   preferred_operation_counts=dict(Counter(p['preferred'] for p in plan)),
                   max_abs_error=float(np.max(abs(err))),relative_l2_error=float(np.linalg.norm(err)/np.linalg.norm(reference)),
                   cached_host_weights_bytes=sum(v.nbytes for v in weights.values()) if dynamic else 0)
        if not np.isfinite(y).all() or entry['relative_l2_error']>0.01:
            raise RuntimeError(f'Block numerical check failed: {entry["relative_l2_error"]}')
        # Same exported graph on CPU distinguishes export differences from
        # differences introduced by device placement/precision.
        cpu=ct.models.MLModel(core.get_spec(),weights_dir=core.weights_dir,compute_units=ct.ComputeUnit.CPU_ONLY)
        cpu_y=np.asarray(next(iter(cpu.predict(feed).values())))
        entry['cpu_only_relative_l2_error']=float(np.linalg.norm(cpu_y-reference)/np.linalg.norm(reference))
        entry['cpu_and_ne_vs_cpu_relative_l2_error']=float(np.linalg.norm(y-cpu_y)/np.linalg.norm(cpu_y))
        entry['numerical_status']='Exploratory only: strict training parity not established'
        entry['cpu_control_passes_1pct_screen']=entry['cpu_only_relative_l2_error']<=0.01
        del cpu
        gc.collect()
        if dynamic:
            changed=dict(feed,proj=weight_feed['proj']*.9)
            modified=np.asarray(next(iter(core.predict(changed).values())))
            entry['changed_projection_output_max_difference']=float(np.max(abs(modified-y)))

        def core_call(cached):
            payload=feed if cached else dict(x=x,**({k:(np.asarray(v).T if v.ndim==2 else np.asarray(v))
                                                   for k,v in masters.items()} if dynamic else {}))
            result=next(iter(core.predict(payload).values()))
            returned=mx.array(result);mx.eval(returned);return returned

        runs=[]
        for mode,fn in [('mlx_resident',gpu_resident),('mlx_roundtrip',gpu_roundtrip),
                        ('coreml_cached',lambda:core_call(True)),('coreml_refresh',lambda:core_call(False)),
                        ('coreml_refresh',lambda:core_call(False)),('coreml_cached',lambda:core_call(True)),
                        ('mlx_roundtrip',gpu_roundtrip),('mlx_resident',gpu_resident)]:
            timing,_=timed(fn);runs.append(dict(mode=mode,**timing))
        entry['runs']=runs
        for mode in sorted(set(r['mode'] for r in runs)):
            values=[v for r in runs if r['mode']==mode for v in r['seconds']]
            entry[mode+'_median_ms']=1000*statistics.median(values)
        report['trials'].append(entry)
        print(json.dumps({k:v for k,v in entry.items() if k not in {'runs','placement'}}),flush=True)
        del core,program
        gc.collect();mx.clear_cache()
        check()
    report.update(final_system=system_sample(),system_checkpoints=checkpoints,mlx_peak_bytes=mx.get_peak_memory(),
                  process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  limitations=['Compute-plan placement is estimated; no direct ANE occupancy measurement.',
                               'The 1% relative-L2 screen only rejects gross export errors; it is not training parity.',
                               'Forward-only probe; no production backward bridge or training quality validation.',
                               'FP16 probe does not establish BF16 parity. Weight values fixed while refreshing API inputs.',
                               'RSS and MLX peaks exclude some driver/ANE allocations; monitor system pressure too.'])
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
