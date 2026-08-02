"""
télos — 25M Model Benchmark v2 (PyTorch / XLA / MPS / CUDA)
============================================================
Benchmarks the EXACT 25M Model Architecture:
- vocab_size: 8192
- d_model: 512, n_layers: 6, n_heads: 8, n_kv_heads: 2 (GQA 4:1)
- sequence length: 512 tokens
- Batch sizes tested: [4, 8, 16, 32, 64, 128, 256, 512]
- Explicit blocking hardware synchronization (torch_xla.sync(wait=True))
"""

import os
import sys
import time

# Force PJRT runtime for TPU detection before importing PyTorch/XLA
if "PJRT_DEVICE" not in os.environ:
    os.environ["PJRT_DEVICE"] = "TPU"

HAS_XLA_MODULE = False
XLA_DEBUG_ERR = None
XLA_DEVICE_OBJ = None

try:
    # pyrefly: ignore
    import torch_xla
    # pyrefly: ignore
    import torch_xla.runtime as xr
    try:
        xr.initialize_cache()
    except Exception:
        pass
    XLA_DEVICE_OBJ = torch_xla.device()
    HAS_XLA_MODULE = True
except Exception as e:
    XLA_DEBUG_ERR = str(e)

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Device Detection ───────────────────────────────────────────────

def detect_device():
    if HAS_XLA_MODULE and XLA_DEVICE_OBJ is not None:
        return XLA_DEVICE_OBJ, f"TPU ({os.environ.get('TPU_NAME', 'v6e-1')})", "xla"
    elif XLA_DEBUG_ERR:
        print(f"Notice: PyTorch XLA TPU device initialization failed ({XLA_DEBUG_ERR}).")

    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0)
        return torch.device("cuda:0"), f"{n}x {name}", "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), "Apple Silicon MPS", "mps"
    return torch.device("cpu"), "CPU", "cpu"


# ─── 25M Parameter Model Architecture ───────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        hidden = int(d_model * 8 / 3)
        hidden = ((hidden + 63) // 64) * 64
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(d_model, hidden, bias=False)
        self.w3 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class Block25M(nn.Module):
    def __init__(self, d_model: int = 512, n_heads: int = 8, n_kv_heads: int = 2):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.head_dim = d_model // n_heads
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.kv_groups = n_heads // n_kv_heads

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.mlp = SwiGLU(d_model)

    def forward(self, x):
        B, T, D = x.shape
        h = self.norm1(x)
        q = self.q_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.kv_groups > 1:
            k = k.repeat_interleave(self.kv_groups, dim=1)
            v = v.repeat_interleave(self.kv_groups, dim=1)

        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.out_proj(attn_out)
        x = x + self.mlp(self.norm2(x))
        return x


class Model25M(nn.Module):
    def __init__(self, V=8192, d=512, layers=6, heads=8, kv_heads=2):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.blocks = nn.ModuleList([Block25M(d, heads, kv_heads) for _ in range(layers)])
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, V, bias=False)
        self.head.weight = self.emb.weight

    def forward(self, x):
        x = self.emb(x)
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x))


# ─── Benchmark Loop ──────────────────────────────────────────────────

def main():
    device, dev_name, dev_type = detect_device()
    use_amp = dev_type in ("cuda", "mps")
    amp_dtype = torch.float16

    V, d, layers, heads, kv_heads = 8192, 512, 6, 8, 2
    seq = 512
    grad_accum = 2
    warmup, measure = 2, 5

    if dev_type == "xla":
        batch_sizes = [4, 8, 16, 32, 64, 128, 256, 512]
    else:
        batch_sizes = [4, 8, 16, 32, 64]

    model = Model25M(V, d, layers, heads, kv_heads).to(device)
    if dev_type == "xla":
        model = model.to(torch.bfloat16)

    n_params = sum(p.numel() for p in model.parameters())

    print("=" * 70)
    print("  télos 25M Model Benchmark v2 (Real 25M Architecture)")
    print("=" * 70)
    print(f"  Device:     {dev_name}")
    print(f"  Model:      {n_params:,} params (d={d}, {layers}L, {heads}H, {kv_heads}KV)")
    print(f"  Precision:  {'bfloat16 (XLA)' if dev_type == 'xla' else ('AMP fp16' if use_amp else 'fp32')}")
    print(f"  Batch Sizes: {batch_sizes}")
    print("=" * 70)

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
                    loss = F.cross_entropy(model(tokens).view(-1, V), targets.view(-1)) / grad_accum
                    loss.backward()
                if dev_type == "xla":
                    opt.step()
                    # pyrefly: ignore
                    import torch_xla
                    torch_xla.sync()
                else:
                    opt.step()
                if dev_type == "cuda": torch.cuda.synchronize()

            # Measure with explicit hardware sync barrier
            if dev_type == "cuda": torch.cuda.synchronize()
            if dev_type == "xla":
                # pyrefly: ignore
                import torch_xla
                torch_xla.sync(wait=True)

            t0 = time.perf_counter()

            for _ in range(measure):
                opt.zero_grad()
                for _ in range(grad_accum):
                    loss = F.cross_entropy(model(tokens).view(-1, V), targets.view(-1)) / grad_accum
                    loss.backward()
                if dev_type == "xla":
                    opt.step()
                    # pyrefly: ignore
                    import torch_xla
                    torch_xla.sync()
                else:
                    opt.step()
                if dev_type == "cuda": torch.cuda.synchronize()

            if dev_type == "xla":
                # pyrefly: ignore
                import torch_xla
                torch_xla.sync(wait=True)

            t1 = time.perf_counter()
            elapsed = t1 - t0
            sps = measure / elapsed
            tps = sps * bs * seq * grad_accum
            step_latency_ms = (elapsed / measure) * 1000.0

            results.append((bs, bs * grad_accum, sps, tps, step_latency_ms))
            print(f"  bs={bs:>3d} | eff_batch={bs*grad_accum:>3d} | {sps:>6.2f} steps/s | {tps:>10,.0f} tok/s | {step_latency_ms:>6.1f} ms/step")

            del opt, tokens, targets
        except Exception as e:
            print(f"  bs={bs:>3d} | Error: {e}")
            break

    print("\n" + "=" * 70)
    print("  25M MODEL BENCHMARK V2 SUMMARY")
    print("=" * 70)
    print(f"  {'Batch':<7} {'Eff Batch':<10} {'Steps/sec':<12} {'Tok/sec':<14} {'Step Latency':<12}")
    print("  " + "─" * 60)
    for bs, eff, sps, tps, lat in results:
        print(f"  {bs:<7d} {eff:<10d} {sps:>7.2f} st/s   {tps:>10,.0f} tok/s   {lat:>6.1f} ms")
    print("=" * 70)


if __name__ == "__main__":
    main()
