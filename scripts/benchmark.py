"""
télos — Minimal Throughput Benchmark (< 1 minute, < 2 GB RAM)
==============================================================
Tests a SINGLE tiny 5M param model with batch_size=8 for 5 steps.
Measures tok/sec, then EXTRAPOLATES to larger models.

Copy-paste friendly for Mac / Kaggle T4 / Colab TPU / Kaggle TPU.
"""

import os
import sys
import time

# Import torch_xla module before torch for C++ ABI alignment, but defer xm.xla_device() call
HAS_XLA_MODULE = False
XLA_DEBUG_ERR = None
try:
    # pyrefly: ignore
    import torch_xla
    # pyrefly: ignore
    import torch_xla.runtime as xr
    try:
        xr.initialize_cache()
    except Exception:
        pass
    HAS_XLA_MODULE = True
except Exception as e:
    XLA_DEBUG_ERR = str(e)

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Device Detection ───────────────────────────────────────────────

def detect_device():
    if HAS_XLA_MODULE:
        try:
            # pyrefly: ignore
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
            return device, f"TPU ({os.environ.get('TPU_NAME', 'v6e-1')})", "xla"
        except Exception as e:
            print(f"DEBUG: xm.xla_device() error: {e}")
    if XLA_DEBUG_ERR:
        print(f"DEBUG: torch_xla import error: {XLA_DEBUG_ERR}")
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0)
        return torch.device("cuda:0"), f"{n}x {name}", "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), "Apple Silicon MPS", "mps"
    return torch.device("cpu"), "CPU", "cpu"


# ─── Tiny Inline Model (same arch as télos, just small) ─────────────

class TinyBlock(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.w1 = nn.Linear(d, d * 2, bias=False)
        self.w2 = nn.Linear(d * 2, d, bias=False)
        self.heads = heads
        self.hd = d // heads

    def forward(self, x):
        B, T, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).view(B, T, 3, self.heads, self.hd)
        q, k, v = qkv[:,:,0], qkv[:,:,1], qkv[:,:,2]
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v)
        x = x + self.out(a.transpose(1, 2).contiguous().view(B, T, D))
        x = x + self.w2(F.silu(self.w1(self.norm2(x))))
        return x

class TinyModel(nn.Module):
    def __init__(self, V=4096, d=256, layers=4, heads=4):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.blocks = nn.ModuleList([TinyBlock(d, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, V, bias=False)
        self.head.weight = self.emb.weight  # tied

    def forward(self, x):
        x = self.emb(x)
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x))


# ─── Benchmark ──────────────────────────────────────────────────────

def run():
    device, dev_name, dev_type = detect_device()
    use_amp = dev_type in ("cuda", "mps")
    amp_dtype = torch.float16

    V, d, layers, heads = 4096, 256, 4, 4
    bs, seq = 8, 512
    grad_accum = 2
    warmup, measure = 2, 5

    model = TinyModel(V, d, layers, heads).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    tokens = torch.randint(0, V, (bs, seq), device=device)
    targets = torch.randint(0, V, (bs, seq), device=device)

    print("=" * 60)
    print("  télos Minimal Throughput Benchmark")
    print("=" * 60)
    print(f"  Device:  {dev_name}")
    print(f"  Model:   {n_params:,} params (d={d}, {layers}L, {heads}H)")
    print(f"  Batch:   {bs} × {grad_accum} accum = {bs*grad_accum} eff")
    print(f"  Seq:     {seq}")
    print(f"  Prec:    {'bfloat16 (XLA)' if dev_type == 'xla' else ('AMP fp16' if use_amp else 'fp32')}")
    print(f"  Steps:   {warmup} warmup + {measure} measured")
    print("=" * 60)

    model.train()

    # Warmup
    for _ in range(warmup):
        opt.zero_grad()
        for _ in range(grad_accum):
            if use_amp:
                with torch.amp.autocast(device_type=dev_type, dtype=amp_dtype):
                    loss = F.cross_entropy(model(tokens).view(-1, V), targets.view(-1)) / grad_accum
            else:
                loss = F.cross_entropy(model(tokens).view(-1, V), targets.view(-1)) / grad_accum
            loss.backward()
        if dev_type == "xla":
            # pyrefly: ignore
            import torch_xla.core.xla_model as xm
            xm.optimizer_step(opt); xm.mark_step()
        else:
            opt.step()
        if dev_type == "cuda": torch.cuda.synchronize()

    # Measure
    if dev_type == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()

    for _ in range(measure):
        opt.zero_grad()
        for _ in range(grad_accum):
            if use_amp:
                with torch.amp.autocast(device_type=dev_type, dtype=amp_dtype):
                    loss = F.cross_entropy(model(tokens).view(-1, V), targets.view(-1)) / grad_accum
            else:
                loss = F.cross_entropy(model(tokens).view(-1, V), targets.view(-1)) / grad_accum
            loss.backward()
        if dev_type == "xla":
            # pyrefly: ignore
            import torch_xla.core.xla_model as xm
            xm.optimizer_step(opt); xm.mark_step()
        else:
            opt.step()
        if dev_type == "cuda": torch.cuda.synchronize()

    t1 = time.perf_counter()
    elapsed = t1 - t0
    sps = measure / elapsed
    tps = sps * bs * seq * grad_accum

    # ─── FLOP-based scaling ─────────────────────────────────────
    # Transformer FLOPs ≈ 6 × N × T per token (forward+backward)
    # So tok/sec scales as: tok_sec_big ≈ tok_sec_small × (N_small / N_big)
    # This is approximate but grounded in compute reality.

    print(f"\n  MEASURED ({n_params:,} params):")
    print(f"    {sps:.2f} steps/sec | {tps:,.0f} tok/sec | {elapsed:.2f}s total")

    print(f"\n  EXTRAPOLATED (linear FLOP scaling from measured):")
    print(f"  {'Model':<8} {'Params':>12} {'Est tok/s':>12} {'Chinchilla 20×':>16} {'Overtrain 50×':>16}")
    print(f"  {'─'*8} {'─'*12} {'─'*12} {'─'*16} {'─'*16}")

    for name, params in [("5M", n_params), ("25M", 25_000_000),
                          ("85M", 85_000_000), ("232M", 232_000_000)]:
        scale = n_params / params  # smaller model = faster
        est_tps = tps * scale
        chin_hrs = (params * 20) / est_tps / 3600
        over_hrs = (params * 50) / est_tps / 3600
        marker = " ← measured" if params == n_params else ""
        print(f"  {name:<8} {params:>12,} {est_tps:>12,.0f} {chin_hrs:>14.1f}h {over_hrs:>14.1f}h{marker}")

    print(f"\n{'=' * 60}")
    print(f"  Copy these numbers. No more guessing.")
    print(f"{'=' * 60}")

    del model, opt, tokens, targets

if __name__ == "__main__":
    run()
