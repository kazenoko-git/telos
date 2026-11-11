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
    total = float(re.search(r"total = ([\d.]+)M", swap).group(1))
    used = float(re.search(r"used = ([\d.]+)M", swap).group(1))
    pressure = subprocess.run(["memory_pressure"], capture_output=True, text=True, check=True)
    pressure_text = pressure.stdout + pressure.stderr
    free_percent = int(re.search(r"System-wide memory free percentage: (\d+)%", pressure_text).group(1))
    result = subprocess.run(["ioreg", "-r", "-c", "AGXAccelerator", "-l"],
                            capture_output=True, text=True, check=True).stdout
    stats = {}
    for key in ("Device Utilization %", "Renderer Utilization %", "Tiler Utilization %", "In use system memory"):
        match = re.search(r'"' + re.escape(key) + r'"=(\d+)', result)
        if match:
            stats[key] = int(match.group(1))
    return {"time": time.time(), "swap_used_mb": used, "swap_free_mb": total - used,
            "memory_free_percent": free_percent, "gpu_system": stats}


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
    parser.add_argument("--critical-memory-free-percent", type=int, default=12)
    parser.add_argument("--critical-swap-free-mib", type=float, default=512.0)
    parser.add_argument("--stage-profile", action="store_true",
                        help="Synchronize between stages to measure update work separately")
    parser.add_argument("--fused-step", action=argparse.BooleanOptionalAction, default=True,
                        help="Fuse the whole optimizer update when accumulation is one")
    parser.add_argument("--compile-microbatch", action=argparse.BooleanOptionalAction, default=True,
                        help="Compile the loss and gradient function (diagnostic comparison)")
    parser.add_argument("--async-stage-eval", action="store_true",
                        help="Use async MLX evaluation before the stage synchronization")
    parser.add_argument("--fused-vocab-loss", action="store_true",
                        help="Use tiled tied-embedding output projection and cross-entropy")
    parser.add_argument("--compiled-accumulation", action="store_true",
                        help="Compile each later microbatch backward pass together with gradient accumulation")
    parser.add_argument("--force-fused-attention", action="store_true",
                        help="Force MLX fused SDPA for a diagnostic comparison")
    parser.add_argument("--fused-vocab-chunk-size", type=int, default=256)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)

    import mlx.core as mx
    original_sdpa = mx.fast.scaled_dot_product_attention
    if args.force_fused_attention:
        def forced_sdpa(q, k, v, **keywords):
            keywords["force_fused"] = True
            return original_sdpa(q, k, v, **keywords)
        mx.fast.scaled_dot_product_attention = forced_sdpa
    from telos.train.cli import train
    from telos.training.mlx_step import MLXTrainingStep
    from telos.training import trainer_mlx

    mx.set_memory_limit(int(args.max_mlx_gib * 1024**3))
    mx.set_cache_limit(128 * 1024**2)
    initial = system_sample()
    metadata = {
        "arguments": vars(args), "device": mx.device_info(), "mlx_version": mx.__version__,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "initial_system": initial, "pid": os.getpid(),
        "limitations": ["GPU utilization is system-wide, including other applications.",
                        "No dedicated Neural Accelerator occupancy counter exposed in this session.",
                        "MLX memory limit is advisory; memory-pressure and swap-headroom guards stop at update boundaries."],
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    original_init = MLXTrainingStep.__init__
    original_batch_fetch = trainer_mlx.get_global_targets_contiguous
    active_stage_profile = [None]
    if args.stage_profile:
        def profiled_init(step, *positional, **keywords):
            original_init(step, *positional, **keywords)
            step.stage_profile = {"async_stage_eval": args.async_stage_eval}
            active_stage_profile[0] = step.stage_profile
        MLXTrainingStep.__init__ = profiled_init

        def profiled_batch_fetch(*positional, **keywords):
            begin = time.perf_counter()
            result = original_batch_fetch(*positional, **keywords)
            elapsed = time.perf_counter() - begin
            if active_stage_profile[0] is not None:
                profile = active_stage_profile[0]
                profile["outer_batch_prepare_seconds"] = (
                    profile.get("outer_batch_prepare_seconds", 0.0) + elapsed
                )
            return result

        trainer_mlx.get_global_targets_contiguous = profiled_batch_fetch
    stop = threading.Event()
    memory_stop = threading.Event()
    errors = []

    def monitor():
        critical_memory_samples = 0
        with (output / "system.jsonl").open("w", buffering=1) as stream:
            while not stop.is_set():
                try:
                    sample = system_sample()
                    stream.write(json.dumps(sample) + "\n")
                    critical = (
                        sample["memory_free_percent"] <= args.critical_memory_free_percent
                        or sample["swap_free_mb"] <= args.critical_swap_free_mib
                    )
                    critical_memory_samples = critical_memory_samples + 1 if critical else 0
                    if critical_memory_samples >= 3:
                        memory_stop.set()
                except Exception as error:
                    errors.append(str(error))
                stop.wait(5)

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    original = MLXTrainingStep.__call__
    original_compile = mx.compile
    if not args.compile_microbatch:
        def diagnostic_compile(function, *positional, **keywords):
            if function.__name__ == "microbatch_step_uncompiled":
                return function
            return original_compile(function, *positional, **keywords)
        mx.compile = diagnostic_compile
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
            if args.stage_profile:
                record.update(step.stage_profile)
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
                  data_path=args.data, synthetic=False, autotune=False, fused_step=args.fused_step, async_eval=False,
                  fused_vocab_loss=args.fused_vocab_loss,
                  fused_vocab_chunk_size=args.fused_vocab_chunk_size,
                  compiled_accumulation=args.compiled_accumulation,
                  max_lr=4e-4, min_lr=4e-5, weight_decay=0.01,
                  checkpoint_dir=str(output / "checkpoints"), save_every=1000, seed=42)
            status = "complete"
        except MemoryGuardStopped as error:
            status = "memory_guard_stopped"
            errors.append(str(error))
        finally:
            mx.fast.scaled_dot_product_attention = original_sdpa
            mx.compile = original_compile
            MLXTrainingStep.__call__ = original
            if args.stage_profile:
                MLXTrainingStep.__init__ = original_init
                trainer_mlx.get_global_targets_contiguous = original_batch_fetch
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
