"""Compare accumulated-update boundaries on identical parameter/gradient shapes.

This isolates update work; it is not a whole-training throughput benchmark.
Run without a competing GPU job. Both core files are imported independently.
"""
import argparse
import hashlib
import gc
import importlib.util
import json
import os
from pathlib import Path
import time

os.environ.setdefault("MLX_ENABLE_TF32", "0")


def load_core(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-core", type=Path, required=True)
    parser.add_argument("--candidate-core", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=16)
    args = parser.parse_args()
    if not 0 < args.warmup < args.updates:
        parser.error("Require 0 < warmup < updates")
    args.output.mkdir(parents=True, exist_ok=False)
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import numpy as np
    from mlx.utils import tree_flatten, tree_map
    from scripts.mlx_hardware_monitor import HardwareMonitor

    cores = {name: load_core(path, name) for name, path in
             (("baseline", args.baseline_core), ("candidate", args.candidate_core))}
    initial = mx.load(str(args.weights))
    mx.eval(initial)
    mx.set_cache_limit(256 * 1024**2)
    mx.set_memory_limit(int(3.2 * 1024**3))
    monitor = HardwareMonitor()
    plan = {"arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "mlx_version": mx.__version__, "device": mx.device_info(),
            "hashes": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (args.baseline_core, args.candidate_core, args.weights)},
            "order": ["baseline", "candidate", "candidate", "baseline"],
            "limitations": ["Fixed synthetic gradients on real parameter shapes.",
                            "Measures update boundary, not whole training.",
                            "Observed GPU clocks, not locked clocks."]}
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    results = []
    for index, mode in enumerate(plan["order"]):
        class ParameterModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weights = {str(i): value for i, value in enumerate(initial.values())}
                self.use_master_weights = True
        model = ParameterModel()
        optimizer = optim.AdamW(1e-3, betas=[.9, .95], bias_correction=True)
        grads = tree_map(lambda p: mx.full(p.shape, .001, dtype=p.dtype), model.parameters())
        loss = mx.array(1.)
        mx.eval(grads, loss)
        def micro(_):
            return loss, loss, grads
        mx.reset_peak_memory()
        durations = []
        hardware = []
        for step in range(args.updates):
            optimizer.learning_rate = 1e-3 if step % 3 else 2e-4
            start = time.perf_counter()
            cores[mode].execute_mlx_training_step(
                model, optimizer, micro, iter((None, None)), 2, 1., step == 0,
                async_accumulation=False)
            durations.append(time.perf_counter() - start)
            if step % 8 == 0:
                hardware.append(monitor.sample())
        final = [model.parameters(), optimizer.state, optimizer.master_params]
        mx.eval(final)
        np.savez(args.output / f"state-{index}.npz", **{
            key: np.asarray(value.astype(mx.float32) if value.dtype == mx.bfloat16 else value)
            for key, value in tree_flatten(final)})
        row = {"index": index, "mode": mode, "seconds": durations,
               "warmed_mean_seconds": sum(durations[args.warmup:]) / (args.updates - args.warmup),
               "peak_mlx_bytes": mx.get_peak_memory(), "hardware": hardware}
        results.append(row)
        (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(mode, row["warmed_mean_seconds"], row["peak_mlx_bytes"], flush=True)
        del model, optimizer, grads, final, micro
        gc.collect()
        mx.clear_cache()


if __name__ == "__main__":
    main()
