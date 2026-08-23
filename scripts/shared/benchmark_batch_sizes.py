"""
télos Single-Process TPU Throughput & Batch Size Benchmark.

Empirically benchmarks granular batch sizes (32, 48, 64, 80, 96, 112)
on Kaggle TPU v5e to find the absolute maximum throughput before OOM.
"""

import os
if "TPU_PROCESS_ADDRESSES" in os.environ:
    os.environ.pop("TPU_PROCESS_ADDRESSES")
if "CLOUD_TPU_TASK_ID" in os.environ:
    os.environ.pop("CLOUD_TPU_TASK_ID")

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


def resolve_device(device_arg: str | None = None):
    if device_arg:
        if device_arg.lower() in ("tpu", "xla"):
            import torch_xla.core.xla_model as xm
            return xm.xla_device(), "tpu"
        elif "cuda" in device_arg.lower():
            return torch.device("cuda"), "cuda"
        elif "mps" in device_arg.lower():
            return torch.device("mps"), "mps"
        else:
            return torch.device("cpu"), "cpu"

    if os.environ.get("PJRT_DEVICE") == "TPU":
        import torch_xla.core.xla_model as xm
        return xm.xla_device(), "tpu"
        
    if "torch_xla" in sys.modules or os.path.exists("/dev/accel0"):
        import torch_xla.core.xla_model as xm
        return xm.xla_device(), "tpu"
        
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    if torch.backends.mps.is_available():
        return torch.device("mps"), "mps"
    return torch.device("cpu"), "cpu"


def benchmark_single_batch(batch_size: int, device, device_type: str, seq_len: int = 512, warmup_steps: int = 5, timed_steps: int = 15):
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
    total_tokens = batch_size * seq_len * timed_steps
    tokens_per_sec = total_tokens / elapsed
    steps_per_sec = timed_steps / elapsed
    
    del model, optimizer, x
    return {"batch_size": batch_size, "tokens_per_sec": tokens_per_sec, "steps_per_sec": steps_per_sec}


def run_benchmark(batch_sizes=[32, 48, 64, 80, 96]):
    device, device_type = resolve_device()
    print("=" * 80)
    print(f"STARTING TPU HARDWARE BENCHMARK ON: {device} ({device_type.upper()})")
    print(f"Architecture: 25M Parameters (d=384, L=13, seq=512)")
    print("=" * 80)
    
    results = []
    for bs in batch_sizes:
        print(f"\nEvaluating Batch Size: {bs}...", flush=True)
        try:
            res = benchmark_single_batch(bs, device, device_type)
            results.append(res)
            print(f"  ✓ {res['tokens_per_sec']:>12,.0f} tok/s | {res['steps_per_sec']:>6.2f} steps/s")
        except Exception as e:
            print(f"  ✗ Failed / OOM with Batch Size {bs}: {e}")
            break
            
    print("\n" + "=" * 80)
    print(f"{'BATCH SIZE':<14} | {'TOKENS / SEC':<16} | {'STEPS / SEC':<14} | {'STATUS'}")
    print("-" * 80)
    best_bs = None
    best_toks = 0
    for r in results:
        toks = r["tokens_per_sec"]
        status = "Optimal" if toks > best_toks else "Valid"
        if toks > best_toks:
            best_toks = toks
            best_bs = r["batch_size"]
        print(f"{r['batch_size']:<14} | {toks:>14,.0f} | {r['steps_per_sec']:>12.2f} | {status}")
    print("=" * 80)
    print(f"★ ABSOLUTE HIGHEST THROUGHPUT: Batch Size = {best_bs} ({best_toks:,.0f} tokens/sec)")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="télos TPU Granular Throughput Benchmark")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[32, 48, 64, 80, 96], help="Batch sizes to evaluate")
    args = parser.parse_args()
    run_benchmark(batch_sizes=args.batch_sizes)


if __name__ == "__main__":
    main()
