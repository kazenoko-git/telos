"""
télos Hardware Throughput & Batch Size Benchmark.

Empirically benchmarks different batch sizes (32, 64, 128, 256, 512)
on the active hardware accelerator (TPU v5e/v6e, NVIDIA GPU, or Apple Silicon)
to determine the optimal throughput (tokens/sec) and memory footprint.
"""

import os
import sys
import time
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


def benchmark_batch_size(batch_size: int, device, device_type: str, seq_len: int = 512, warmup_steps: int = 5, timed_steps: int = 15):
    """Benchmarks a single batch size for the 25M zero-shock architecture."""
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
    
    # Generate random test batch
    x = torch.randint(0, 8192, (batch_size, seq_len), device=device)
    
    # Warmup
    for _ in range(warmup_steps):
        optimizer.zero_grad()
        logits = model(x)
        loss = logits.sum()
        loss.backward()
        if device_type == "tpu":
            import torch_xla
            torch_xla.sync()
            import torch_xla.core.xla_model as xm
            xm.optimizer_step(optimizer)
            torch_xla.sync()
        else:
            optimizer.step()
            
    # Timed run
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
            torch_xla.sync()
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
    tokens_processed = batch_size * seq_len * timed_steps
    tokens_per_sec = tokens_processed / elapsed
    steps_per_sec = timed_steps / elapsed
    
    # Estimate memory
    mem_str = "N/A"
    if device_type == "cuda":
        mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        mem_str = f"{mem_gb:.2f} GB"
        
    del model, optimizer, x
    if device_type == "cuda":
        torch.cuda.empty_cache()
        
    return {
        "batch_size": batch_size,
        "tokens_per_sec": tokens_per_sec,
        "steps_per_sec": steps_per_sec,
        "elapsed": elapsed,
        "memory": mem_str
    }


def run_benchmark_suite(batch_sizes=[32, 64, 128, 256, 512]):
    device, device_type = resolve_device()
    print("=" * 80)
    print(f"STARTING HARDWARE BENCHMARK ON: {device} ({device_type.upper()})")
    print(f"Testing 25M Architecture (d=384, L=13, seq=512)")
    print("=" * 80)
    
    results = []
    for bs in batch_sizes:
        print(f"\nEvaluating Batch Size: {bs}...", flush=True)
        try:
            res = benchmark_batch_size(bs, device, device_type)
            results.append(res)
            print(f"  ✓ {res['tokens_per_sec']:,.0f} tok/s ({res['steps_per_sec']:.2f} steps/s) | Memory: {res['memory']}")
        except Exception as e:
            print(f"  ✗ Failed / OOM with Batch Size {bs}: {e}")
            results.append({"batch_size": bs, "tokens_per_sec": 0, "steps_per_sec": 0, "memory": "OOM / Error"})
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
    print(f"★ HIGHEST THROUGHPUT: Batch Size = {best_bs} ({best_toks:,.0f} tokens/sec)")
    print("=" * 80)
    return best_bs


if __name__ == "__main__":
    run_benchmark_suite()
