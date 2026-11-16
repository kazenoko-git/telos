"""Same-runtime native/fallback SDPA VJP comparison; research, not a dispatcher.

The environment flag is changed only to construct independent cached graphs in
this single-threaded probe. No production trainer or process-wide default is
modified. Component preferences require full-update confirmation before use.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import time
os.environ.setdefault('MLX_ENABLE_TF32','0')
import mlx.core as mx
from mlx.utils import tree_flatten


def outputs(tree):
    return [value for _,value in tree_flatten(tree)]


def reference(q,k,v):
    q,k,v=(x.astype(mx.float32) for x in (q,k,v))
    if q.shape[1]!=k.shape[1]:
        repeats=q.shape[1]//k.shape[1]
        k,v=mx.repeat(k,repeats,axis=1),mx.repeat(v,repeats,axis=1)
    scores=(q@k.transpose(0,1,3,2))*(q.shape[-1]**-.5)
    mask=mx.arange(q.shape[2])[:,None]>=mx.arange(k.shape[2])[None,:]
    return mx.softmax(mx.where(mask,scores,-float('inf')),axis=-1)@v


def backward_graph(dim, cotangent):
    # Bind each shape's values independently of the benchmark loop and cache.
    def attention(q, k, v):
        return mx.fast.scaled_dot_product_attention(
            q, k, v, scale=dim**-.5, mask='causal', force_fused=True
        )

    def transformed(q, k, v):
        return mx.vjp(attention, (q, k, v), (cotangent,))

    return transformed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pairs',type=int,default=12)
    parser.add_argument('--lengths',type=int,nargs='+',default=[64,128,256,512,1024,2048])
    parser.add_argument('--head-dims',type=int,nargs='+',default=[64,128])
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    assert mx.__version__=='0.32.4','Use the isolated validated runtime'
    mx.set_memory_limit(int(1.5*1024**3));mx.set_cache_limit(32*1024**2)
    results=[];original_flag=os.environ.get('MLX_SDPA_VJP_FALLBACK')
    try:
        for dim in args.head_dims:
            for kv_heads in (8,2):
                for length in args.lengths:
                    batch=max(1,2048//length)
                    mx.random.seed(245)
                    inputs=(mx.random.normal((batch,8,length,dim)).astype(mx.bfloat16),
                            mx.random.normal((batch,kv_heads,length,dim)).astype(mx.bfloat16),
                            mx.random.normal((batch,kv_heads,length,dim)).astype(mx.bfloat16))
                    cotangent=mx.random.normal(inputs[0].shape).astype(mx.bfloat16)
                    mx.eval(inputs,cotangent)
                    reference_values=mx.vjp(reference,inputs,(cotangent.astype(mx.float32),))
                    mx.eval(reference_values)
                    functions={};errors={};case=f'b{batch}_h8_kv{kv_heads}_t{length}_d{dim}'
                    for mode,flag in (('native','0'),('fallback','1')):
                        os.environ['MLX_SDPA_VJP_FALLBACK']=flag
                        transformed=backward_graph(dim,cotangent)
                        graph=transformed(*inputs)
                        graph_path=args.output/f'{case}_{mode}.dot'
                        mx.export_to_dot(str(graph_path),*outputs(graph))
                        dedicated='ScaledDotProductAttentionVJP' in graph_path.read_text()
                        assert dedicated==(mode=='native'),(case,mode,'unexpected backend graph')
                        mx.eval(graph)
                        measurements=[]
                        for name,a,b in zip(('output','dq','dk','dv'),outputs(reference_values),outputs(graph)):
                            a,b=a.astype(mx.float32),b.astype(mx.float32)
                            relative=mx.sqrt(mx.sum((a-b)**2)/mx.maximum(mx.sum(a*a),1e-30)).item()
                            assert mx.all(mx.isfinite(b)).item() and relative<.015,(case,mode,name,relative)
                            measurements.append(dict(tensor=name,relative_l2=relative,max_abs=mx.max(mx.abs(a-b)).item()))
                        errors[mode]=measurements
                        function=mx.compile(transformed)
                        for _ in range(4):mx.eval(function(*inputs))
                        functions[mode]=function
                        del graph
                    samples=[]
                    for pair in range(args.pairs):
                        sample={}
                        for mode in (('native','fallback') if pair%2==0 else ('fallback','native')):
                            os.environ['MLX_SDPA_VJP_FALLBACK']='0' if mode=='native' else '1'
                            mx.synchronize();started=time.perf_counter()
                            for _ in range(4):
                                value=functions[mode](*inputs)
                                mx.async_eval(*outputs(value))
                            mx.synchronize()
                            sample[mode]=(time.perf_counter()-started)/4
                        samples.append(sample)
                    gains=[100*(r['fallback']/r['native']-1) for r in samples]
                    native_wins=sum(g>0 for g in gains)
                    median=statistics.median(gains)
                    preference=('native' if median>=3 and native_wins>=.75*len(gains) else
                                'fallback' if median<=-3 and native_wins<=.25*len(gains) else 'inconclusive')
                    row=dict(shape=case,batch=batch,query_heads=8,kv_heads=kv_heads,length=length,head_dim=dim,
                             errors=errors,samples=samples,median_native_gain_percent=median,native_wins=native_wins,
                             component_preference=preference,peak_mlx_bytes=mx.get_peak_memory())
                    results.append(row);(args.output/'results.json').write_text(json.dumps(results,indent=2))
                    print(case,round(median,2),preference,flush=True)
                    del functions,reference_values,inputs,cotangent,value
                    mx.clear_cache()
    finally:
        if original_flag is None:os.environ.pop('MLX_SDPA_VJP_FALLBACK',None)
        else:os.environ['MLX_SDPA_VJP_FALLBACK']=original_flag
    (args.output/'metadata.json').write_text(json.dumps(dict(mlx_version=mx.__version__,mlx_module=mx.__file__,
        device=mx.device_info(),arguments=vars(args)|{'output':str(args.output)},
        limitations=['Component graph and throughput probe, not whole-model throughput or convergence.',
                     'Preferences are restricted to the measured shapes and require production-loop confirmation.']),indent=2))


if __name__=='__main__':main()
