"""Fresh-process head-only loss/gradient benchmark with measured allocation."""
import argparse,json,os,time
from pathlib import Path
os.environ.setdefault('MLX_ENABLE_TF32','0')
import mlx.core as mx
from telos.training.mlx_fused_projection import fused_projection_cross_entropy
from telos.training.mlx_losses import cross_entropy
from telos.training.mlx_vocab_loss import tiled_projection_cross_entropy,token_chunked_projection_cross_entropy

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--variant',choices=['full','vocab','token','metal'],required=True)
parser.add_argument('--chunk-size',type=int,default=256)
parser.add_argument('--vocab',type=int,required=True)
parser.add_argument('--rows',type=int,default=2048)
parser.add_argument('--width',type=int,default=512)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--steps',type=int,default=24)
args=parser.parse_args()
mx.set_memory_limit(int(1.6*1024**3));mx.set_cache_limit(32*1024**2)
mx.random.seed(281)
h=mx.random.normal((args.rows,args.width)).astype(mx.bfloat16)
w=(mx.random.normal((args.vocab,args.width))*(args.width**-.5)).astype(mx.bfloat16)
t=mx.random.randint(0,args.vocab,(args.rows,))
cot=mx.random.uniform(low=-1.,high=1.,shape=(args.rows,))/args.rows
mx.eval(h,w,t,cot)
def full(x,weight):return mx.sum(cross_entropy(x@weight.T,t,reduction='none')*cot)
def losses(x, weight):
    if args.variant == 'metal':
        return fused_projection_cross_entropy(x, weight, t)
    function = token_chunked_projection_cross_entropy if args.variant == 'token' else tiled_projection_cross_entropy
    return function(x, weight, t, chunk_size=args.chunk_size)
def candidate(x,weight):return mx.sum(losses(x, weight)*cot)
reference=mx.compile(mx.value_and_grad(full,argnums=(0,1)))
function=reference if args.variant=='full' else mx.compile(mx.value_and_grad(candidate,argnums=(0,1)))
a,ga=reference(h,w);b,gb=function(h,w);mx.eval(a,b,ga,gb)
expected_losses=cross_entropy(h@w.T,t,reduction='none')
if args.variant=='full':
    actual_losses=expected_losses
else:
    actual_losses=losses(h,w)
loss_max_abs=mx.max(mx.abs(expected_losses-actual_losses)).item()
assert loss_max_abs < (5e-5 if args.variant == 'metal' else 1e-5), loss_max_abs
del expected_losses,actual_losses
error=[]
assert abs((a-b).item())<1e-5,(a.item(),b.item())
for x,y in zip(ga,gb):
    x,y=x.astype(mx.float32),y.astype(mx.float32)
    relative=mx.sqrt(mx.sum((x-y)**2)/mx.maximum(mx.sum(x*x),1e-30)).item()
    assert mx.all(mx.isfinite(y)).item() and relative<.015,relative
    error.append(relative)
loss_difference=(b-a).item()
del a,b,ga,gb,x,y
for _ in range(4):mx.eval(function(h,w))
mx.synchronize();mx.clear_cache();mx.reset_peak_memory()
resident=mx.get_active_memory();latencies=[]
for _ in range(args.steps):
    mx.synchronize();start=time.perf_counter();mx.eval(function(h,w));mx.synchronize()
    latencies.append(time.perf_counter()-start)
result=dict(arguments=vars(args)|{'output':str(args.output)},mlx_version=mx.__version__,mlx_module=mx.__file__,
            latencies=latencies,seconds=sum(latencies),tokens_per_second=args.steps*args.rows/sum(latencies),
            resident_bytes=resident,peak_mlx_bytes=mx.get_peak_memory(),gradient_relative_l2=error,loss_difference=loss_difference,
            loss_max_abs=loss_max_abs,
            limitations=['Head-only BF16 synthetic loss/gradient diagnostic, not full-model training throughput or convergence.',
                         'Actual allocator peak is measured; no O(chunk) allocation guarantee follows from the Python loop.'])
args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
