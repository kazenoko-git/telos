"""Small Core ML placement/latency probes; no training-engine integration.

Requires coremltools (can be installed outside the project and added to PYTHONPATH).
CPU_AND_NE permits CPU fallback; compute-plan placement is an estimate.
"""
import json
from pathlib import Path
import statistics
import time
import inspect
import numpy as np
import mlx.core as mx
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.models.compute_plan import MLComputePlan
from scripts.profile_mlx_training import system_sample


def main():
    rng = np.random.default_rng(42)
    x = (rng.normal(size=(256, 512)) * .1).astype(np.float16)
    w = (rng.normal(size=(512, 512)) * .1).astype(np.float16)
    reports = []
    initial = system_sample()
    mx.set_memory_limit(2 * 1024**3)
    mx.set_cache_limit(64 * 1024**2)
    for dynamic in [False, True]:
        if dynamic:
            @mb.program(input_specs=[mb.TensorSpec(shape=x.shape), mb.TensorSpec(shape=w.shape)], opset_version=ct.target.macOS15)
            def program(x, w):
                return mb.matmul(x=x, y=w, name='output')
            inputs = {'x': x.astype(np.float32), 'w': w.astype(np.float32)}
        else:
            weight = w.T.reshape(512, 512, 1, 1)
            @mb.program(input_specs=[mb.TensorSpec(shape=(1,512,1,256))], opset_version=ct.target.macOS15)
            def program(x):
                return mb.conv(x=x, weight=weight, name='output')
            inputs = {'x': x.T.reshape(1,512,1,256).astype(np.float32)}
        begin = time.perf_counter()
        model = ct.convert(program, convert_to='mlprogram', minimum_deployment_target=ct.target.macOS15,
                           compute_precision=ct.precision.FLOAT16, compute_units=ct.ComputeUnit.CPU_AND_NE)
        conversion_load = time.perf_counter() - begin
        placement = []
        kwargs = {}
        if 'compute_units' in inspect.signature(MLComputePlan.load_from_path).parameters:
            kwargs['compute_units'] = ct.ComputeUnit.CPU_AND_NE
        plan = MLComputePlan.load_from_path(model.get_compiled_model_path(), **kwargs)
        def visit(block):
            for op in block.operations:
                usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
                placement.append({'op': op.operator_name,
                                  'preferred': type(usage.preferred_compute_device).__name__ if usage else None,
                                  'supported': [type(d).__name__ for d in usage.supported_compute_devices] if usage else []})
                for nested in op.blocks:
                    visit(nested)
        for fn in plan.model_structure.program.functions.values():
            visit(fn.block)
        for _ in range(3):
            output = model.predict(inputs)
        timings=[]
        for _ in range(20):
            begin=time.perf_counter();output=model.predict(inputs);timings.append(time.perf_counter()-begin)
        y=next(iter(output.values()))
        if not dynamic:y=y.reshape(512,256).T
        reference=x.astype(np.float32) @ w.astype(np.float32)
        a,b=mx.array(x),mx.array(w)
        gpu=[]
        for i in range(23):
            begin=time.perf_counter();z=a @ b;mx.eval(z)
            if i>=3:gpu.append(time.perf_counter()-begin)
        # Include ordinary numpy/MLX transfer and synchronization around the GPU result.
        transfer=[]
        for _ in range(20):
            begin=time.perf_counter();a2=mx.array(x);b2=mx.array(w);z=a2@b2;mx.eval(z);np.asarray(z);transfer.append(time.perf_counter()-begin)
        reports.append(dict(dynamic_weights=dynamic, shape=list(x.shape), conversion_load_seconds=conversion_load,
                            coreml_predict_median_ms=1000*statistics.median(timings),
                            mlx_resident_median_ms=1000*statistics.median(gpu),
                            mlx_host_roundtrip_median_ms=1000*statistics.median(transfer),
                            max_abs_error=float(np.max(abs(y-reference))), placement=placement))
        if dynamic:
            changed=dict(inputs,w=(w*.5).astype(np.float32))
            y2=next(iter(model.predict(changed).values()))
            reports[-1]['updated_weight_max_abs_error']=float(np.max(abs(y2-reference*.5)))
            # A linear layer has an explicit backward, so no GPU forward replay
            # is needed. This measures real weight refresh and host handoffs.
            x32 = mx.array(x.astype(np.float32))
            @mx.compile
            def backward(master, y):
                loss = mx.mean(y ** 2)
                grad = x32.T @ (2 * y / y.size)
                return master - 1e-3 * grad, loss

            @mx.compile
            def gpu_step(master):
                y = (a @ master.astype(mx.float16)).astype(mx.float32)
                return backward(master, y)

            def train(hybrid):
                master = mx.array(w.astype(np.float32))
                elapsed, losses = [], []
                for i in range(1100):
                    begin = time.perf_counter()
                    if hybrid:
                        feed = dict(inputs, w=np.asarray(master))
                        y = mx.array(next(iter(model.predict(feed).values())))
                        master, loss = backward(master, y)
                    else:
                        master, loss = gpu_step(master)
                    mx.eval(master, loss)
                    if i >= 100:
                        elapsed.append(time.perf_counter() - begin)
                        losses.append(loss.item())
                return master, elapsed, losses
            runs = [train(hybrid) for hybrid in [False, True, True, False]]
            gpu_w, _, gpu_losses = runs[-1]
            hybrid_w, _, hybrid_losses = runs[-2]
            gpu_steps = runs[0][1] + runs[3][1]
            hybrid_steps = runs[1][1] + runs[2][1]
            reports[-1]['linear_training'] = dict(
                measured_updates=1000, warmup_updates=100,
                trial_order=['gpu', 'hybrid', 'hybrid', 'gpu'],
                trial_median_ms=[1000*statistics.median(r[1]) for r in runs],
                mlx_median_ms=1000*statistics.median(gpu_steps),
                hybrid_median_ms=1000*statistics.median(hybrid_steps),
                final_weight_max_abs_difference=float(mx.max(mx.abs(gpu_w-hybrid_w)).item()),
                final_mlx_loss=gpu_losses[-1], final_hybrid_loss=hybrid_losses[-1],
                scope='FP16 linear forward, analytic FP32 GPU backward/update; includes weight refresh and host handoffs.')
        print(json.dumps(reports[-1]),flush=True)
        del plan,model
    report=dict(coremltools=ct.__version__,mlx=mx.__version__,device=mx.device_info(),initial_system=initial,
                final_system=system_sample(),mlx_peak_bytes=mx.get_peak_memory(),trials=reports,
                limitations=['Compute-plan placement is estimated, not measured ANE occupancy.',
                             'FP16 matrix probes are not a transformer training benchmark or BF16 parity proof.',
                             'Core ML predict includes host API overhead; MLX resident timing excludes transfers.'])
    Path('logs/neural_engine_probe_20261111.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
