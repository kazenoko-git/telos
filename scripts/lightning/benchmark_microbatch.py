"""
Benchmark suite for 25M Telos Transformer on TPU v6e-1 (Lightning AI).
Tests various microbatch sizes and gradient accumulation settings WITHOUT torch_xla.sync()
to find optimal MFU and throughput.
"""

import sys
import os
import gc
import math
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.amp.syncfree as syncfree
from torch.utils.data import Dataset, DataLoader

from mdiff.model.transformer import TelosTransformer, TelosConfig


class SyntheticDataset(Dataset):
    def __init__(self, num_samples: int = 30000, seq_len: int = 512, vocab_size: int = 8192):
        # Deterministic synthetic integers
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len), dtype=torch.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def benchmark_config(batch_size: int, grad_accum: int, seq_len: int = 512, warmup_steps: int = 5, bench_steps: int = 25):
    device = xm.xla_device()
    vocab_size = 8192
    
    cfg = TelosConfig(
        vocab_size=vocab_size,
        d_model=384,
        n_layers=13,
        n_heads=6,
        n_kv_heads=6,
        seq_len=seq_len,
        is_causal=True
    )
    
    model = TelosTransformer(cfg).to(device, dtype=torch.bfloat16)
    model.train()
    
    optimizer = syncfree.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    
    dataset = SyntheticDataset(num_samples=30000, seq_len=seq_len, vocab_size=vocab_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2, prefetch_factor=2)
    device_loader = pl.MpDeviceLoader(loader, device)
    data_iter = iter(device_loader)
    
    tokens_per_step = batch_size * grad_accum * seq_len
    # 25.8M parameters -> 6 * N FLOPs per token
    flops_per_token = 6 * 25.8e6
    
    print(f"\n---> Testing Microbatch: {batch_size} | Grad Accum: {grad_accum} | Effective Batch: {batch_size * grad_accum} ({tokens_per_step:,} tokens/step)")
    
    # Warmup / Compilation
    t0_compile = time.time()
    for s in range(warmup_steps):
        optimizer.zero_grad(set_to_none=True)
        for mb in range(grad_accum):
            try:
                x = next(data_iter)
            except StopIteration:
                data_iter = iter(device_loader)
                x = next(data_iter)
            
            with torch.autocast(device_type="xla", dtype=torch.bfloat16, enabled=True):
                logits = model(x)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = x[:, 1:].contiguous()
                loss = nn.functional.cross_entropy(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
                loss = loss / grad_accum
            loss.backward()
        xm.optimizer_step(optimizer)
    
    compile_time = time.time() - t0_compile
    print(f"     Warmup/Compilation time for {warmup_steps} steps: {compile_time:.2f}s")
    
    # Timed Benchmark
    t0_bench = time.time()
    for s in range(bench_steps):
        optimizer.zero_grad(set_to_none=True)
        for mb in range(grad_accum):
            try:
                x = next(data_iter)
            except StopIteration:
                data_iter = iter(device_loader)
                x = next(data_iter)
            
            with torch.autocast(device_type="xla", dtype=torch.bfloat16, enabled=True):
                logits = model(x)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = x[:, 1:].contiguous()
                loss = nn.functional.cross_entropy(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
                loss = loss / grad_accum
            loss.backward()
        xm.optimizer_step(optimizer)
    
    bench_time = time.time() - t0_bench
    steps_per_sec = bench_steps / bench_time
    toks_per_sec = steps_per_sec * tokens_per_step
    tflops = (toks_per_sec * flops_per_token) / 1e12
    
    print(f"     Result: {toks_per_sec/1e3:.1f}k tok/s | {steps_per_sec:.2f} step/s | {tflops:.2f} TFLOPS | Step Time: {1000/steps_per_sec:.1f}ms")
    
    del model, optimizer, dataset, loader, device_loader
    gc.collect()
    
    return {
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "tokens_per_step": tokens_per_step,
        "toks_per_sec": toks_per_sec,
        "steps_per_sec": steps_per_sec,
        "tflops": tflops,
        "step_ms": 1000 / steps_per_sec,
        "compile_time": compile_time
    }


def main():
    print("=" * 80)
    print("TELOS 25M TPU v6e-1 MICROBATCH & GRAPH COMPILATION BENCHMARK (NO INNER SYNC)")
    print("=" * 80)
    
    configs = [
        (64, 1),
        (64, 2),
        (64, 4),
        (128, 1),
        (128, 2),
        (128, 4),
        (256, 1),
        (256, 2),
        (512, 1),
    ]
    
    results = []
    for bs, ga in configs:
        try:
            res = benchmark_config(batch_size=bs, grad_accum=ga)
            results.append(res)
        except Exception as e:
            print(f"     FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 95)
    print(f"{'Microbatch':<12} | {'GradAccum':<10} | {'Eff. Batch':<12} | {'Throughput':<15} | {'Steps/sec':<12} | {'TFLOPS':<10} | {'Step Time':<10}")
    print("=" * 95)
    for r in results:
        print(f"{r['batch_size']:<12} | {r['grad_accum']:<10} | {r['batch_size']*r['grad_accum']:<12} | {r['toks_per_sec']/1e3:6.1f}k tok/s  | {r['steps_per_sec']:6.2f} st/s   | {r['tflops']:6.2f}     | {r['step_ms']:6.1f} ms")
    print("=" * 95)


if __name__ == "__main__":
    main()
