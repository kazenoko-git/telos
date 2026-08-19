"""
Standalone Runner for 3-Paradigm Throughput & Optimization Benchmark.

Tests:
1. Model Scale Sweep (5M, 12M, 25M, 50M, 100M) across AR, MDLM, UNDLM.
2. Microbatch (1 to 256) & Effective Batch Size (up to 4096) Sweep on ~10M model across AR, MDLM, UNDLM.
"""

import os
import sys
import time
import gc
import yaml
import math
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure working directory is project root
project_root = Path(__file__).resolve().parent.parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

import mlx.core as mx
import mlx.nn as mx_nn
import mlx.optimizers as mx_optim
from mlx.utils import tree_map

from mdiff.model.mlx_components import MLXTelosTransformer
from mdiff.training.trainer import (
    apply_masking_mlx, loss_fn_mlx, build_special_token_lut,
    cast_optimizer_moments_bf16, get_sys_mem_str
)
from undiff.diffusion.forward_process import apply_uniform_noise_mlx
from undiff.diffusion.loss import undlm_loss
from ar.model.mlx_components import MLXCausalTransformer
from ar.training.trainer import ar_loss_fn_mlx


def benchmark_patch(
    paradigm: str,
    model_cfg: dict,
    micro_batch: int,
    grad_accum: int,
    warmup_steps: int = 3,
    bench_steps: int = 10,
    seq_len: int = 512,
    vocab_size: int = 8192
) -> dict:
    mx.clear_cache()
    gc.collect()
    
    if paradigm == "ar":
        model = MLXCausalTransformer(vocab_size=vocab_size, **model_cfg)
    else:
        model = MLXTelosTransformer(vocab_size=vocab_size, **model_cfg)
    model.set_dtype(mx.bfloat16)
    
    n_params = sum(p.size for _, p in mx_nn.utils.tree_flatten(model.parameters()))
    special_lut = build_special_token_lut(vocab_size)
    
    if paradigm == "ar":
        loss_and_grad = mx_nn.value_and_grad(model, ar_loss_fn_mlx)
        def microstep_raw(batch):
            (loss, ce), grads = loss_and_grad(model, batch, vocab_size)
            return loss, ce, grads
    elif paradigm == "mdlm":
        loss_and_grad = mx_nn.value_and_grad(model, loss_fn_mlx)
        def microstep_raw(batch):
            masked, mask_pos, t = apply_masking_mlx(batch, mask_token_id=1, special_token_lut=special_lut)
            (loss, ce), grads = loss_and_grad(model, masked, batch, mask_pos, t, vocab_size)
            return loss, ce, grads
    elif paradigm == "undlm":
        loss_and_grad = mx_nn.value_and_grad(model, undlm_loss)
        def microstep_raw(batch):
            noisy, corrupt_mask, t = apply_uniform_noise_mlx(batch, vocab_size=vocab_size, special_token_lut=special_lut)
            (loss, ce), grads = loss_and_grad(model, noisy, batch, t, vocab_size)
            return loss, ce, grads
    
    dummy_seqs = mx.random.randint(0, vocab_size, (micro_batch, seq_len))
    dl, dc, dg = microstep_raw(dummy_seqs)
    mx.eval(dl, dc, dg)
    del dl, dc, dg
    
    state = [model.state]
    compiled_step = mx.compile(microstep_raw, inputs=state, outputs=state)
    optimizer = mx_optim.AdamW(learning_rate=3e-4, weight_decay=0.1)
    
    for _ in range(warmup_steps):
        accum_grads = None
        for _ in range(grad_accum):
            batch = mx.random.randint(0, vocab_size, (micro_batch, seq_len))
            loss, ce, grads = compiled_step(batch)
            accum_grads = grads if accum_grads is None else tree_map(lambda a, b: a + b, accum_grads, grads)
            mx.eval(accum_grads, loss)
        accum_grads = tree_map(lambda g: g / grad_accum, accum_grads)
        optimizer.update(model, accum_grads)
        mx.eval(model.parameters(), optimizer.state)
    
    optimizer.state = cast_optimizer_moments_bf16(optimizer.state)
    mx.eval(optimizer.state)
    mx.clear_cache()
    
    start_time = time.perf_counter()
    for step in range(bench_steps):
        accum_grads = None
        for _ in range(grad_accum):
            batch = mx.random.randint(0, vocab_size, (micro_batch, seq_len))
            loss, ce, grads = compiled_step(batch)
            accum_grads = grads if accum_grads is None else tree_map(lambda a, b: a + b, accum_grads, grads)
            mx.eval(accum_grads, loss)
        accum_grads = tree_map(lambda g: g / grad_accum, accum_grads)
        optimizer.update(model, accum_grads)
        mx.eval(model.parameters(), optimizer.state)
    
    elapsed = time.perf_counter() - start_time
    
    steps_per_sec = bench_steps / elapsed
    eff_batch = micro_batch * grad_accum
    tok_per_sec = steps_per_sec * eff_batch * seq_len
    ms_per_step = (elapsed / bench_steps) * 1000.0
    peak_mem_gb = mx.get_peak_memory() / 1e9
    active_mem_gb = mx.get_active_memory() / 1e9
    
    del model, optimizer, compiled_step
    gc.collect()
    mx.clear_cache()
    
    return {
        "paradigm": paradigm.upper(),
        "params_m": round(n_params / 1e6, 2),
        "micro_batch": micro_batch,
        "grad_accum": grad_accum,
        "eff_batch": eff_batch,
        "steps_per_sec": round(steps_per_sec, 2),
        "tok_per_sec": int(tok_per_sec),
        "ms_per_step": round(ms_per_step, 1),
        "peak_mem_gb": round(peak_mem_gb, 2),
        "active_mem_gb": round(active_mem_gb, 2)
    }


def main():
    print("=" * 85)
    print("  RUNNING 3-PARADIGM OPTIMIZATION TEST SUITE (AR vs MDLM vs UNDLM)")
    print("=" * 85)
    
    # 1. Model Scale Sweep
    scales = {
        "5M":   {"d_model": 256, "n_layers": 4,  "n_heads": 4,  "n_kv_heads": 2},
        "12M":  {"d_model": 256, "n_layers": 12, "n_heads": 8,  "n_kv_heads": 4},
        "25M":  {"d_model": 512, "n_layers": 8,  "n_heads": 8,  "n_kv_heads": 4},
        "50M":  {"d_model": 768, "n_layers": 8,  "n_heads": 12, "n_kv_heads": 4},
        "100M": {"d_model": 768, "n_layers": 16, "n_heads": 12, "n_kv_heads": 4}
    }
    
    print("\n--- TEST 1: Model Scale Sweep (MicroBatch=4, GradAccum=8) ---")
    scale_results = []
    for s_name, arch in scales.items():
        for p in ["ar", "mdlm", "undlm"]:
            try:
                res = benchmark_patch(p, arch, micro_batch=4, grad_accum=8, bench_steps=10)
                res["scale_tag"] = s_name
                scale_results.append(res)
                print(f"  [{s_name:<4} | {p.upper():<5}] {res['steps_per_sec']:>5.2f} st/s | {res['tok_per_sec']:>8,} tok/s | {res['ms_per_step']:>6.1f} ms/st | Peak RAM: {res['peak_mem_gb']:.2f} GB")
            except Exception as e:
                print(f"  [{s_name:<4} | {p.upper():<5}] OOM / Error: {e}")
                
    df_scale = pd.DataFrame(scale_results)
    print("\n" + df_scale.to_string(index=False))
    
    # 2. Batch Size Sweep on ~10M model
    print("\n--- TEST 2: Microbatch & Effective Batch Size Sweep (~10M model) ---")
    arch_10m = {"d_model": 256, "n_layers": 12, "n_heads": 8, "n_kv_heads": 4}
    mb_list = [1, 2, 4, 8, 16, 32, 64, 128]
    eff_list = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    
    batch_results = []
    for mb in mb_list:
        for eff in eff_list:
            if eff < mb or (eff % mb != 0): continue
            ga = eff // mb
            if ga > 128 and mb > 16: continue
            
            for p in ["ar", "mdlm", "undlm"]:
                try:
                    res = benchmark_patch(p, arch_10m, micro_batch=mb, grad_accum=ga, bench_steps=5)
                    batch_results.append(res)
                    print(f"  [{res['paradigm']:<5} | MB={mb:>3d} | GA={ga:>3d} | EffB={eff:>4d}] {res['steps_per_sec']:>5.2f} st/s | {res['tok_per_sec']:>9,} tok/s | Peak RAM: {res['peak_mem_gb']:.2f} GB")
                except Exception as e:
                    print(f"  [{p.upper():<5} | MB={mb:>3d} | GA={ga:>3d} | EffB={eff:>4d}] OOM: {e}")
                    break

    if batch_results:
        df_batch = pd.DataFrame(batch_results)
        print("\n" + df_batch.to_string(index=False))
        
        print("\n" + "=" * 85)
        print("  SWEET SPOT HIGHLIGHTS BY PARADIGM")
        print("=" * 85)
        for p in ["AR", "MDLM", "UNDLM"]:
            sub = df_batch[df_batch["paradigm"] == p]
            if not sub.empty:
                best = sub.loc[sub["tok_per_sec"].idxmax()]
                print(f"★ {p} Max Throughput: {best['tok_per_sec']:,} tok/s ({best['steps_per_sec']} st/s) at MicroBatch={best['micro_batch']}, GradAccum={best['grad_accum']} (EffB={best['eff_batch']})")
                print(f"   Peak RAM: {best['peak_mem_gb']} GB | Step Latency: {best['ms_per_step']} ms\n")


if __name__ == "__main__":
    main()
