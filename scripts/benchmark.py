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

# Force PJRT runtime for TPU detection before importing PyTorch/XLA
if "PJRT_DEVICE" not in os.environ:
    os.environ["PJRT_DEVICE"] = "TPU"

HAS_XLA_MODULE = False
XLA_DEBUG_ERR = None
try:
    # pyrefly: ignore
    import torch_xla
    # pyrefly: ignore
    import torch_xla.core.xla_model as xm
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
    if XLA_DEBUG_ERR:
        print(f"DEBUG: torch_xla import failed with error: {XLA_DEBUG_ERR}")

    if HAS_XLA_MODULE:
        # Try xm.xla_device() first
        try:
            # pyrefly: ignore
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
            return device, f"TPU ({os.environ.get('TPU_NAME', 'v6e-1')})", "xla"
        except Exception as e:
            print(f"DEBUG: xm.xla_device() failed: {e}")

        # Try torch_xla.device() second
        try:
            # pyrefly: ignore
            import torch_xla
            device = torch_xla.device()
            return device, f"TPU ({os.environ.get('TPU_NAME', 'v6e-1')})", "xla"
        except Exception as e:
            print(f"DEBUG: torch_xla.device() failed: {e}")

        # Try torch.device('xla') third
        try:
            device = torch.device("xla:0")
            return device, f"TPU ({os.environ.get('TPU_NAME', 'v6e-1')})", "xla"
        except Exception as e:
            print(f"DEBUG: torch.device('xla') failed: {e}")

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
    seq = 512
    grad_accum = 2
    warmup, measure = 2, 5
    batch_sizes = [4, 8, 16, 32, 64, 128, 256, 512]

    model = TinyModel(V, d, layers, heads).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    print("=" * 65)
    print("  télos Batch-Size Optimization Benchmark")
    print("=" * 60)
    print(f"  Device:     {dev_name}")
    print(f"  Model:      {n_params:,} params (d={d}, {layers}L, {heads}H)")
    print(f"  Precision:  {'bfloat16 (XLA)' if dev_type == 'xla' else ('AMP fp16' if use_amp else 'fp32')}")
    print(f"  Batch Sizes Tested: {batch_sizes}")
    print("=" * 65)

    results = []

    for bs in batch_sizes:
        try:
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
            tokens = torch.randint(0, V, (bs, seq), device=device)
            targets = torch.randint(0, V, (bs, seq), device=device)

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

            results.append((bs, bs * grad_accum, sps, tps, elapsed))
            print(f"  bs={bs:>3d} | eff_batch={bs*grad_accum:>3d} | {sps:>6.2f} steps/s | {tps:>10,.0f} tok/s | {elapsed:.2f}s")

            del opt, tokens, targets
        except Exception as e:
            print(f"  bs={bs:>3d} | OOM/Error: {e}")
            break

    print("\n" + "=" * 65)
    print("  BATCH SIZE COMPARISON SUMMARY")
    print("=" * 65)
    print(f"  {'Batch':<8} {'Eff Batch':<10} {'Steps/sec':<12} {'Tok/sec':<14} {'Speed vs bs=8'}")
    print(f"  {'─'*7:<8} {'─'*9:<10} {'─'*11:<12} {'─'*13:<14} {'─'*13}")

    bs8_tps = next((r[3] for r in results if r[0] == 8), results[0][3] if results else 1.0)
    best_res = max(results, key=lambda x: x[3]) if results else None

    for bs, eff_b, sps, tps, el in results:
        ratio = tps / bs8_tps
        star = " ★ BEST" if best_res and bs == best_res[0] else ""
        print(f"  {bs:<8} {eff_b:<10} {sps:<12.2f} {tps:<14,.0f} {ratio:>5.2f}x{star}")

    print("=" * 65)

    if best_res:
        best_tps = best_res[3]
        print(f"\n  EXTRAPOLATED FROM BEST (bs={best_res[0]}, {best_tps:,.0f} tok/s):")
        print(f"  {'Model':<8} {'Params':>12} {'Est tok/s':>12} {'Chinchilla 20×':>16} {'Overtrain 50×':>16}")
        print(f"  {'─'*8} {'─'*12} {'─'*12} {'─'*16} {'─'*16}")

        for name, params in [("5M", n_params), ("25M", 25_000_000),
                              ("85M", 85_000_000), ("232M", 232_000_000)]:
            scale = n_params / params
            est_tps = best_tps * scale
            chin_hrs = (params * 20) / est_tps / 3600
            over_hrs = (params * 50) / est_tps / 3600
            marker = " ← measured" if params == n_params else ""
            print(f"  {name:<8} {params:>12,} {est_tps:>12,.0f} {chin_hrs:>14.1f}h {over_hrs:>14.1f}h{marker}")

        print("=" * 65)


if __name__ == "__main__":
    run()
