"""Shape-matched long-context attention VJP against independent FP32 math."""
import os
os.environ.setdefault('MLX_ENABLE_TF32','0')
import argparse
import json
from pathlib import Path
import mlx.core as mx


def reference(q,k,v):
    q,k,v=(x.astype(mx.float32) for x in (q,k,v))
    scores=(q@k.transpose(0,1,3,2))*(q.shape[-1]**-.5)
    causal=mx.arange(q.shape[-2])[:,None]>=mx.arange(k.shape[-2])[None,:]
    return mx.softmax(mx.where(causal,scores,-mx.inf),axis=-1)@v


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    mx.set_memory_limit(1024**3);mx.set_cache_limit(32*1024**2)
    rows=[]
    for dtype in (mx.float32,mx.bfloat16):
        mx.random.seed(66)
        inputs=tuple(mx.random.normal((2,8,1024,64)).astype(dtype) for _ in range(3))
        cotangent=mx.random.normal(inputs[0].shape).astype(dtype)
        ref,ref_grads=mx.vjp(reference,inputs,(cotangent.astype(mx.float32),))
        actual,grads=mx.vjp(lambda q,k,v:mx.fast.scaled_dot_product_attention(q,k,v,scale=.125,mask='causal',force_fused=True),inputs,(cotangent,))
        mx.eval(ref,ref_grads,actual,grads)
        metrics=[]
        for label,a,b in zip(('output','dq','dk','dv'),(*ref,*ref_grads),(*actual,*grads)):
            a,b=a.astype(mx.float32),b.astype(mx.float32)
            assert mx.all(mx.isfinite(a)).item() and mx.all(mx.isfinite(b)).item()
            relative=mx.sqrt(mx.sum((a-b)**2)/mx.maximum(mx.sum(a*a),1e-30)).item()
            maximum=mx.max(mx.abs(a-b)).item()
            assert relative < (1e-4 if dtype==mx.float32 else .015),(label,relative)
            metrics.append(dict(tensor=label,relative_l2=relative,max_abs=maximum))
        rows.append(dict(dtype=str(dtype),metrics=metrics))
    report=dict(mlx_version=mx.__version__,mlx_module=mx.__file__,shape=[2,8,1024,64],
                causal=True,results=rows,peak_mlx_bytes=mx.get_peak_memory(),
                limitations=['First-order attention component check, not whole-model convergence or trained-model quality.'])
    args.output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
