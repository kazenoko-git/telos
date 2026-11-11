"""Experimental training loop for one linear layer with optional Core ML forward.

Modes:
  gpu:      FP16 MLX forward, MLX backward and AdamW
  ane:      serial Core ML forward, MLX analytic backward and AdamW
  pipeline: one-item Core ML forward prefetch overlapped with MLX backward

Requires coremltools in PYTHONPATH. This is isolated from paradigm trainers.
"""
import argparse
import concurrent.futures
import json
from pathlib import Path
import statistics
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb


class LinearModel(nn.Module):
    def __init__(self, width, weights):
        super().__init__()
        self.weight = mx.array(weights)


def build_coreml_forward(rows, width):
    @mb.program(
        input_specs=[mb.TensorSpec(shape=(rows, width)), mb.TensorSpec(shape=(width, width))],
        opset_version=ct.target.macOS15,
    )
    def program(x, w):
        return mb.matmul(x=x, y=w, name="linear_projection")

    model = ct.convert(
        program,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    from coremltools.models.compute_plan import MLComputePlan
    kwargs = {}
    if "compute_units" in __import__("inspect").signature(MLComputePlan.load_from_path).parameters:
        kwargs["compute_units"] = ct.ComputeUnit.CPU_AND_NE
    plan = MLComputePlan.load_from_path(model.get_compiled_model_path(), **kwargs)
    placements = []

    def visit(block):
        for op in block.operations:
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            placements.append({
                "op": op.operator_name,
                "preferred": type(usage.preferred_compute_device).__name__ if usage else None,
                "supported": [type(d).__name__ for d in usage.supported_compute_devices] if usage else [],
            })
            for nested in op.blocks:
                visit(nested)

    for function in plan.model_structure.program.functions.values():
        visit(function.block)
    if not any("NeuralEngine" in item["preferred"] for item in placements if item["preferred"]):
        raise RuntimeError(f"Core ML did not prefer ANE for this projection shape: {placements}")
    return model, placements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("gpu", "ane", "pipeline"), required=True)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-mlx-gib", type=float, default=1.0)
    args = parser.parse_args()

    mx.set_memory_limit(int(args.max_mlx_gib * 1024**3))
    mx.set_cache_limit(64 * 1024**2)
    rng = np.random.default_rng(42)
    width, rows = args.width, args.rows
    weight = (rng.standard_normal((width, width)) * 0.02).astype(np.float32)
    teacher = (rng.standard_normal((width, width)) * 0.02).astype(np.float32)
    batches = []
    for _ in range(args.updates * args.accumulation):
        x = (rng.standard_normal((rows, width)) * 0.1).astype(np.float32)
        target = (x @ teacher).astype(np.float32)
        batches.append((x, target))
    model = LinearModel(width, weight)
    optimizer = optim.AdamW(learning_rate=1e-3, weight_decay=0.01)
    coreml_model = placements = None
    if args.mode != "gpu":
        coreml_model, placements = build_coreml_forward(rows, width)

    # Fixed weights are copied once per accumulation window and held until its update.
    def forward_ane(batch_x, window_weight):
        output = coreml_model.predict({"x": batch_x, "w": window_weight})
        return next(iter(output.values()))

    @mx.compile
    def backward(batch_x, target, prediction):
        x32 = batch_x.astype(mx.float32)
        y32 = prediction.astype(mx.float32)
        target32 = target.astype(mx.float32)
        dy = (2.0 / y32.size) * (y32 - target32)
        grad_weight = x32.T @ dy
        grad_input = dy @ model.weight.T
        loss = mx.mean((y32 - target32) ** 2)
        return grad_weight, grad_input, loss

    def gpu_forward(batch_x):
        return (batch_x.astype(mx.float16) @ model.weight.astype(mx.float16)).astype(mx.float16)

    timings, losses = [], []
    update = 0
    for start in range(0, len(batches), args.accumulation):
        window = batches[start:start + args.accumulation]
        frozen_weight = np.asarray(model.weight).copy()
        grad_sum = None
        loss_sum = mx.array(0.0, dtype=mx.float32)
        update_start = time.perf_counter()
        if args.mode == "pipeline":
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                pending = pool.submit(forward_ane, *window[0][:1], frozen_weight)
                for index, (batch_x, target) in enumerate(window):
                    prediction_np = pending.result()
                    if index + 1 < len(window):
                        pending = pool.submit(forward_ane, window[index + 1][0], frozen_weight)
                    prediction = mx.array(prediction_np)
                    grad, grad_input, loss = backward(mx.array(batch_x), mx.array(target), prediction)
                    grad_sum = grad if grad_sum is None else grad_sum + grad
                    loss_sum = loss_sum + loss
                    mx.eval(grad_sum, loss_sum, grad_input)
        else:
            for batch_x, target in window:
                x_mx, target_mx = mx.array(batch_x), mx.array(target)
                if args.mode == "ane":
                    prediction = mx.array(forward_ane(batch_x, frozen_weight))
                else:
                    prediction = gpu_forward(x_mx)
                grad, grad_input, loss = backward(x_mx, target_mx, prediction)
                grad_sum = grad if grad_sum is None else grad_sum + grad
                loss_sum = loss_sum + loss
                mx.eval(grad_sum, loss_sum, grad_input)
        grads = {"weight": grad_sum / float(args.accumulation)}
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss_sum)
        mx.synchronize()
        update += 1
        losses.append(float((loss_sum / args.accumulation).item()))
        if update > 2:
            timings.append(time.perf_counter() - update_start)

    result = {
        "mode": args.mode, "shape": [rows, width, width], "accumulation": args.accumulation,
        "updates": update, "warm_updates_excluded": 2,
        "median_update_ms": 1000 * statistics.median(timings),
        "updates_per_second": 1 / statistics.median(timings),
        "final_loss": losses[-1], "loss_history": losses,
        "peak_mlx_gib": mx.get_peak_memory() / 1024**3,
        "compute_plan": placements,
        "limitation": "Single linear model only; no transformer integration or full optimizer-state parity comparison.",
    }
    output = Path(args.output)
    output.write_text(json.dumps(result, indent=2) + "\n")
    np.save(output.with_suffix(".weight.npy"), np.asarray(model.weight))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
