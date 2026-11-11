"""Profile a bounded MLX AR run through the normal training entry point.

Writes per-update timings and system telemetry without retaining activation
traces. GPU utilization is system-wide, not per-process NAX occupancy.
"""

import argparse
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import threading
import time


def system_sample():
    """Read macOS telemetry without requiring administrator privileges."""
    swap = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True, check=True).stdout
    used = float(re.search(r"used = ([\d.]+)M", swap).group(1))
    result = subprocess.run(["ioreg", "-r", "-c", "AGXAccelerator", "-l"],
                            capture_output=True, text=True, check=True).stdout
    stats = {}
    for key in ("Device Utilization %", "Renderer Utilization %", "Tiler Utilization %", "In use system memory"):
        match = re.search(r'"' + re.escape(key) + r'"=(\d+)', result)
        if match:
            stats[key] = int(match.group(1))
    return {"time": time.time(), "swap_used_mb": used, "gpu_system": stats}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", default="50M")
    parser.add_argument("--tokens", type=int, default=49_166_848)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--microbatch", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--data", default="data/english_corpus_5b.bin")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-mlx-gib", type=float, default=4.0)
    parser.add_argument("--max-swap-growth-mib", type=float, default=512.0)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)

    import mlx.core as mx
    from telos.train.cli import train
    from telos.training.mlx_step import MLXTrainingStep

    mx.set_memory_limit(int(args.max_mlx_gib * 1024**3))
    mx.set_cache_limit(128 * 1024**2)
    initial = system_sample()
    metadata = {
        "arguments": vars(args), "device": mx.device_info(), "mlx_version": mx.__version__,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "initial_system": initial, "pid": os.getpid(),
        "limitations": ["GPU utilization is system-wide, including other applications.",
                        "No dedicated Neural Accelerator occupancy counter exposed in this session.",
                        "MLX memory limit is advisory; live peak and swap guards stop at update boundaries."],
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    stop = threading.Event()
    memory_stop = threading.Event()
    errors = []

    def monitor():
        high_swap_samples = 0
        with (output / "system.jsonl").open("w", buffering=1) as stream:
            while not stop.is_set():
                try:
                    sample = system_sample()
                    stream.write(json.dumps(sample) + "\n")
                    high_swap_samples = high_swap_samples + 1 if (
                        sample["swap_used_mb"] - initial["swap_used_mb"] > args.max_swap_growth_mib
                    ) else 0
                    if high_swap_samples >= 3:
                        memory_stop.set()
                except Exception as error:
                    errors.append(str(error))
                stop.wait(5)

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    original = MLXTrainingStep.__call__
    count = 0
    last_step = None
    status = "running"
    started = time.perf_counter()

    class MemoryGuardStopped(RuntimeError):
        pass

    with (output / "steps.jsonl").open("w", buffering=1) as timings:
        def measured_step(step, *positional, **keywords):
            nonlocal count, last_step
            last_step = step
            begin = time.perf_counter()
            result = original(step, *positional, **keywords)
            elapsed = time.perf_counter() - begin
            count += 1
            record = {"step": count, "time": time.time(), "training_step_seconds": elapsed,
                      "mlx_active_bytes": mx.get_active_memory(), "mlx_peak_bytes": mx.get_peak_memory(),
                      "mlx_cache_bytes": mx.get_cache_memory(),
                      "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
            if count == 1 or count % 50 == 0:
                record["loss"] = result[0].item() / args.grad_accum
            timings.write(json.dumps(record) + "\n")
            if record["mlx_peak_bytes"] > args.max_mlx_gib * 1024**3 or memory_stop.is_set():
                step.synchronize()
                step.model.save_weights(str(output / "memory_stop.safetensors"))
                raise MemoryGuardStopped("Stopped at update boundary after the memory guard triggered")
            return result

        MLXTrainingStep.__call__ = measured_step
        try:
            train(paradigm="ar", params=args.params, tokens=args.tokens, hardware="mlx",
                  batch_size=args.microbatch, grad_accum=args.grad_accum, seq_len=args.seq_len,
                  data_path=args.data, synthetic=False, autotune=False, fused_step=True, async_eval=False,
                  max_lr=4e-4, min_lr=4e-5, weight_decay=0.01,
                  checkpoint_dir=str(output / "checkpoints"), save_every=1000, seed=42)
            status = "complete"
        except MemoryGuardStopped as error:
            status = "memory_guard_stopped"
            errors.append(str(error))
        finally:
            MLXTrainingStep.__call__ = original
            if last_step is not None:
                last_step.synchronize()
            stop.set()
            monitor_thread.join(timeout=10)
            summary = {"status": status, "updates": count,
                       "input_tokens_processed": count * args.microbatch * args.grad_accum * args.seq_len,
                       "wall_seconds": time.perf_counter() - started, "errors": errors,
                       "final_system": system_sample()}
            (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print("PROFILE_SUMMARY=" + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
