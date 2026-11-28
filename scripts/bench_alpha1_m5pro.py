#!/usr/bin/env python3
"""Live Apple M5 Pro compiled Metal 2-bit decode & MTP speculative benchmark for Telos Alpha 1.

Measures real unified-memory single-stream decode speed (`T=1`, `mx.compile`) and
MTP `K=3` speculative verify throughput (`T=4` verify + 3-head draft, `mx.compile`)
for `BitMoE-2.25B-A377M` using MLX's native Metal `bits=2` `mx.quantized_matmul`
and `mx.gather_qmm` with fused QKVG/QKV projections and 4-bit LM head.

Usage:
  .venv/bin/python -u scripts/bench_alpha1_m5pro.py
  .venv/bin/python -u scripts/bench_alpha1_m5pro.py --lm-head-bits 16
  .venv/bin/python -u scripts/bench_alpha1_m5pro.py --checkpoint /path/to/checkpoint.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mlx.core as mx
import mlx.utils
import yaml

from telos.models.bitmoe_mlx import BitMoETransformerMLX


def _total_param_bytes(model: BitMoETransformerMLX) -> int:
    total = 0
    for _, arr in mlx.utils.tree_flatten(model.parameters()):
        if isinstance(arr, mx.array):
            total += arr.nbytes
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Telos Alpha 1 (2.25B-A377M) compiled Metal 2-bit decode.")
    parser.add_argument("--config", default="configs/ar/bitmoe_2.25b_a377m_v5e8.yaml")
    parser.add_argument("--checkpoint", default=None, help="Optional PyTorch .pt checkpoint to losslessly convert & benchmark.")
    parser.add_argument("--lm-head-bits", type=int, default=4, choices=[4, 8, 16], help="LM head quantization bits (default: 4).")
    parser.add_argument("--no-compile", action="store_true", help="Disable mx.compile (for A/B comparison).")
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--mtp-steps", type=int, default=60)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg["model"]

    print("=" * 76, flush=True)
    print(f"  Telos Alpha 1 — Apple Silicon MLX Metal 2-bit Benchmark", flush=True)
    print(f"  MLX: {mx.__version__} | Device: {mx.default_device()} | compile={not args.no_compile} | lm_head_bits={args.lm_head_bits}", flush=True)
    print("=" * 76, flush=True)

    t0 = time.time()
    if args.checkpoint:
        import torch

        ckpt = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=True)
        sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model = BitMoETransformerMLX.from_pytorch_state_dict(model_cfg, sd, lm_head_bits=args.lm_head_bits)
        src_desc = f"checkpoint ({args.checkpoint})"
    else:
        model = BitMoETransformerMLX.from_config_random(model_cfg, lm_head_bits=args.lm_head_bits)
        src_desc = "random packed 2-bit ternary"
    mx.eval(model.parameters())
    load_s = time.time() - t0

    total_mb = _total_param_bytes(model) / (1024 ** 2)
    print(f"  Loaded {src_desc} in {load_s:.2f}s | Packed Metal footprint: {total_mb:.1f} MiB", flush=True)

    # 1. Prefill to populate GDN recurrent state + static rolling SWA KV cache
    caches = model.init_caches(batch=1, dtype=mx.float16)
    prompt = mx.array([[10 + (i % 100) for i in range(args.prompt_len)]], dtype=mx.int32)
    mx.eval(caches, prompt)

    t_pre = time.perf_counter()
    logits, hidden, caches = model.forward_with_cache(prompt, caches)
    next_tok = mx.argmax(logits[:, -1:, :], axis=-1)
    mx.eval(next_tok, hidden, caches)
    pre_s = time.perf_counter() - t_pre
    print(f"  Prefill ({args.prompt_len} tok): {pre_s * 1000:.1f} ms ({args.prompt_len / pre_s:,.0f} tok/s)", flush=True)

    # 2. Compiled single-stream T=1 decode benchmark (pipelined via mx.async_eval like mlx-lm)
    step_t1 = model.decode_step_t1 if args.no_compile else mx.compile(model.decode_step_t1)
    for _ in range(args.warmup):
        next_tok, caches = step_t1(next_tok, caches)
        mx.eval(next_tok, caches)

    t_dec0 = time.perf_counter()
    prev_tok = next_tok
    for _ in range(args.steps):
        next_tok, caches = step_t1(prev_tok, caches)
        mx.async_eval(next_tok, caches)
        _ = prev_tok.item()
        prev_tok = next_tok
    mx.eval(next_tok, caches)
    dec_elapsed = time.perf_counter() - t_dec0
    tps_t1 = args.steps / dec_elapsed

    print("-" * 76, flush=True)
    print(f"  [T=1 Single-Stream Decode]  {tps_t1:,.1f} tok/s  |  mean={1000.0/tps_t1:.2f} ms/tok (async pipelined)", flush=True)

    # 3. Compiled MTP K=3 speculative cycle (T=4 parallel verify + 3-head draft, async pipelined)
    if model.mtp_heads:
        step_mtp = model.mtp_verify_and_draft_step if args.no_compile else mx.compile(model.mtp_verify_and_draft_step)
        draft_in = mx.concatenate((next_tok, model.draft_mtp(hidden[:, -1:, :], next_tok)), axis=-1)  # (1, 4)
        mx.eval(draft_in)
        for _ in range(10):
            draft_in, caches = step_mtp(draft_in, caches)
            mx.eval(draft_in, caches)

        t_m0 = time.perf_counter()
        prev_win = draft_in
        for _ in range(args.mtp_steps):
            draft_in, caches = step_mtp(prev_win, caches)
            mx.async_eval(draft_in, caches)
            _ = prev_win[0, 0].item()
            prev_win = draft_in
        mx.eval(draft_in, caches)
        mtp_elapsed = time.perf_counter() - t_m0
        cycles_per_s = args.mtp_steps / mtp_elapsed
        mean_mtp_ms = 1000.0 / cycles_per_s
        print(f"  [MTP K=3 Verify(T=4)+Draft] {mean_mtp_ms:.2f} ms/cycle ({cycles_per_s:.1f} verify-cycles/s)", flush=True)
        for acc in (1.8, 2.2, 2.6, 3.0):
            print(f"    -> Projected sustained decode @ {acc:.1f} tok/cycle: {cycles_per_s * acc:,.1f} tok/s", flush=True)

    peak_mem_mb = mx.get_peak_memory() / (1024 ** 2)
    print("-" * 76, flush=True)
    print(f"  Peak Metal Memory: {peak_mem_mb:.1f} MiB", flush=True)

    # 4. Component breakdown (T=1) so we see exactly where every 0.01 ms goes
    x1 = mx.zeros((1, 1, model.d_model), dtype=mx.float16)
    mx.eval(x1)
    gdn_layer = next(l.mixer for l in model.layers if hasattr(l.mixer, "qkvg_proj"))
    swa_layer = next(l.mixer for l in model.layers if hasattr(l.mixer, "qkv_proj"))
    moe_layer = model.layers[0].mlp
    gdn_c = gdn_layer.init_state(1, mx.float16)
    swa_c = swa_layer.init_state(1, mx.float16)
    mx.eval(gdn_c, swa_c)

    def _bench_ms(fn, iters=200):
        c_fn = fn if args.no_compile else mx.compile(fn)
        for _ in range(20):
            mx.eval(c_fn())
        t0 = time.perf_counter()
        for _ in range(iters):
            mx.eval(c_fn())
        return (time.perf_counter() - t0) * 1000.0 / iters

    ms_gdn = _bench_ms(lambda: gdn_layer(x1, gdn_c)[0])
    ms_swa = _bench_ms(lambda: swa_layer(x1, swa_c)[0])
    ms_moe = _bench_ms(lambda: moe_layer(x1))
    ms_lm = _bench_ms(lambda: model.project_logits(x1))
    print("-" * 76, flush=True)
    print(
        f"  [T=1 Component Breakdown] 15x GDN={15*ms_gdn:.2f}ms ({ms_gdn:.3f}/L) | "
        f"5x SWA={5*ms_swa:.2f}ms ({ms_swa:.3f}/L) | "
        f"20x MoE={20*ms_moe:.2f}ms ({ms_moe:.3f}/L) | LM={ms_lm:.2f}ms",
        flush=True,
    )
    print("=" * 76, flush=True)


if __name__ == "__main__":
    main()
