"""
télos Hardware Throughput & Batch Size Benchmark.

Empirically benchmarks different batch sizes (32, 64, 128, 256)
on all TPU cores (or GPU/Apple Silicon) to measure tokens/sec and memory footprint.
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np

# Ensure project root is on PATH
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from mdiff.model.transformer import TelosTransformer, TelosConfig


def resolve_device():
    try:
        import torch_xla.core.xla_model as xm
        return xm.xla_device(), "tpu"
    except Exception:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    if torch.backends.mps.is_available():
        return torch.device("mps"), "mps"
    return torch.device("cpu"), "cpu"


def _worker_benchmark(index: int, batch_size: int, seq_len: int, warmup_steps: int, timed_steps: int, device_type: str, num_cores: int, return_dict):
    if device_type == "tpu":
        import torch_xla.core.xla_model as xm
        import torch_xla
        device = xm.xla_device()
        rank = xm.get_ordinal()
        world_size = xm.xrt_world_size()
    else:
        device = torch.device("cuda" if device_type == "cuda" else ("mps" if device_type == "mps" else "cpu"))
        rank = 0
        world_size = 1

    telos_cfg = TelosConfig(
        vocab_size=8192,
        d_model=384,
        n_layers=13,
        n_heads=6,
        n_kv_heads=6,
        seq_len=seq_len,
        is_causal=False
    )
    
    model = TelosTransformer(telos_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    
    x = torch.randint(0, 8192, (batch_size, seq_len), device=device)
    
    # Warmup
    for _ in range(warmup_steps):
        optimizer.zero_grad()
        logits = model(x)
        loss = logits.sum()
        loss.backward()
        if device_type == "tpu":
            import torch_xla
            import torch_xla.core.xla_model as xm
            xm.optimizer_step(optimizer)
            torch_xla.sync()
        else:
            optimizer.step()
            
    if device_type == "cuda":
        torch.cuda.synchronize()
    elif device_type == "tpu":
        import torch_xla
        torch_xla.sync()
        
    start_time = time.perf_counter()
    for _ in range(timed_steps):
        optimizer.zero_grad()
        logits = model(x)
        loss = logits.sum()
        loss.backward()
        if device_type == "tpu":
            import torch_xla
            import torch_xla.core.xla_model as xm
            xm.optimizer_step(optimizer)
            torch_xla.sync()
        else:
            optimizer.step()
            
    if device_type == "cuda":
        torch.cuda.synchronize()
    elif device_type == "tpu":
        import torch_xla
        torch_xla.sync()
        
    elapsed = time.perf_counter() - start_time
    total_tokens = batch_size * seq_len * timed_steps * world_size
    tokens_per_sec = total_tokens / elapsed
    steps_per_sec = timed_steps / elapsed
    
    if rank == 0:
        return_dict["tokens_per_sec"] = tokens_per_sec
        漂 = tokens_per_sec
        return_dict["steps_per_sec"] = steps_per_sec
        return_dict["elapsed"] = elapsed


def benchmark_batch_size(batch_size: int, device_type: str, num_cores: int = 8, seq_len: int = 512):
    return_dict = {}
    if device_type == "tpu" and num_cores > 1:
        try:
            import torch_xla.distributed.xla_multiprocessing as xmp
            import multiprocessing
            manager = multiprocessing.Manager()
            shared_dict = manager.dict()
            xmp.spawn(
                _worker_benchmark,
                args=(batch_size, seq_len, 5, 15, device_type, num_cores, shared_dict),
                nprocs=num_cores,
                start_method="spawn"
            )
            return dict(shared_dict)
        except Exception as e:
            print(f"    (Multi-core fallback: {e})")
            
    _worker_benchmark(0, batch_size, seq_len, 5, 15, device_type, 1, return_dict)
    return return_dict


def run_benchmark_suite(batch_sizes=[32, 64, 128, 256], num_cores: int = 8):
    device, device_type = resolve_device()
    cores_label = f" ({num_cores} cores)" if device_type == "tpu" else ""
    print("=" * 80)
    print(f"STARTING THROUGHPUT BENCHMARK ON: {device_type.upper()}{cores_label}")
    print(f"Architecture: 25M Parameters (d=384, L=13, seq=512)")
    print("=" * 80)
    
    results = []
    for bs in batch_sizes:
        print(f"\nEvaluating Batch Size: {bs} (per core)...", flush=True)
        try:
            res = benchmark_batch_size(bs, device_type, num_cores)
            toks = res.get("tokens_per_sec", 0)
            steps = res.get("steps_per_sec", 0)
            results.append({"batch_size": bs, "tokens_per_sec": toks, "steps_per_sec": steps})
            print(f"  ✓ {toks:,.0f} tok/s ({steps:.2f} steps/s)")
        except Exception as e:
            print(f"  ✗ Failed / OOM with Batch Size {bs}: {e}")
            break
            
    print("\n" + "=" * 80)
    print(f"{'BATCH SIZE':<12} | {'TOKENS / SEC':<16} | {'STEPS / SEC':<14} | {'STATUS'}")
    print("-" * 80)
    best_bs = None
    best_toks = 0
    for r in results:
        toks = r["tokens_per_sec"]
        status = "Optimal" if toks > best_toks else "Valid"
        if toks > best_toks:
            best_toks = toks
            best_bs = r["batch_size"]
        print(f"{r['batch_size']:<12} | {toks:>14,.0f} | {r['steps_per_sec']:>12.2f} | {status}")
    print("=" * 80)
    print(f"★ HIGHEST THROUGHPUT: Batch Size = {best_bs} ({best_toks:,.0f} tokens/sec across all cores)")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="télos TPU / GPU Throughput Benchmark")
    parser.add_argument("--cores", type=int, default=8, help="Number of TPU cores to utilize (1 or 8)")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[32, 64, 128, 256], help="Batch sizes to evaluate")
    args = parser.parse_args()
    run_benchmark_suite(batch_sizes=args.batch_sizes, num_cores=args.cores)


if __name__ == "__main__":
    main()
