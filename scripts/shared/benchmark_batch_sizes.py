"""
télos 8-Core Hardware Throughput & Batch Size Benchmark.

Measures throughput (tokens/sec) across all 8 TPU cores (or GPU/CPU)
using native PyTorch XLA multiprocessing (xmp.spawn).
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


def _worker_benchmark(index: int, batch_size: int, seq_len: int, warmup_steps: int, timed_steps: int, device_type: str):
    if device_type == "tpu":
        import torch_xla.core.xla_model as xm
        import torch_xla
        device = xm.xla_device()
        rank = xm.get_ordinal()
        world_size = xm.xrt_world_size()
    elif device_type == "cuda":
        device = torch.device("cuda")
        rank = 0
        world_size = 1
    else:
        device = torch.device("cpu")
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
        print(f"  ✓ {tokens_per_sec:>12,.0f} tok/s | {steps_per_sec:>6.2f} steps/s | Aggregate across {world_size} cores", flush=True)


def run_benchmark_for_batch(batch_size: int, device_type: str, seq_len: int = 512):
    print(f"\nEvaluating Batch Size {batch_size} (per core)...", flush=True)
    if device_type == "tpu":
        import torch_xla.distributed.xla_multiprocessing as xmp
        xmp.spawn(
            _worker_benchmark,
            args=(batch_size, seq_len, 5, 15, device_type),
            start_method="spawn"
        )
    else:
        _worker_benchmark(0, batch_size, seq_len, 5, 15, device_type)


def main():
    parser = argparse.ArgumentParser(description="télos 8-Core TPU Throughput Benchmark")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[32, 64], help="Batch sizes to evaluate")
    args = parser.parse_args()
    
    device_type = "tpu" if os.environ.get("PJRT_DEVICE") == "TPU" or "torch_xla" in sys.modules or os.path.exists("/dev/accel0") else ("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 80)
    print(f"STARTING 8-CORE THROUGHPUT BENCHMARK ON: {device_type.upper()}")
    print(f"Architecture: 25M Parameters (d=384, L=13, seq=512)")
    print("=" * 80)
    
    for bs in args.batch_sizes:
        try:
            run_benchmark_for_batch(bs, device_type)
        except Exception as e:
            print(f"  ✗ Failed / OOM with Batch Size {bs}: {e}")
            
    print("\n" + "=" * 80)
    print("Benchmark Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
