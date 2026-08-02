"""
Apple MLX Throughput Benchmark for télos MDLM
==============================================
Tests a 5M param model natively in Apple MLX (Apple Silicon M-series GPU).
Uses lazy evaluation (mx.eval) and unified memory.

Measures actual tok/sec and compares directly against PyTorch MPS.
"""

import time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


class MLXSwiGLU(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        hidden = int(d_model * 8 / 3)
        hidden = ((hidden + 63) // 64) * 64
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(d_model, hidden, bias=False)
        self.w3 = nn.Linear(hidden, d_model, bias=False)

    def __call__(self, x):
        return self.w3(nn.silu(self.w1(x)) * self.w2(x))


class MLXBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.mlp = MLXSwiGLU(d_model)
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

    def __call__(self, x):
        B, T, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, T, 3, self.n_heads, self.head_dim)
        q = qkv[:, :, 0].transpose(0, 2, 1, 3)  # (B, H, T, HD)
        k = qkv[:, :, 1].transpose(0, 2, 1, 3)
        v = qkv[:, :, 2].transpose(0, 2, 1, 3)

        scale = 1.0 / (self.head_dim ** 0.5)
        scores = (q @ k.transpose(0, 1, 3, 2)) * scale
        attn = mx.softmax(scores, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, D)

        x = x + self.out(out)
        x = x + self.mlp(self.norm2(x))
        return x


class MLXTinyModel(nn.Module):
    def __init__(self, vocab_size: int = 4096, d_model: int = 256, n_layers: int = 4, n_heads: int = 4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.layers = [MLXBlock(d_model, n_heads) for _ in range(n_layers)]
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def __call__(self, x):
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x))


def loss_fn(model, tokens, targets):
    logits = model(tokens)
    B, T, V = logits.shape
    logits_flat = logits.reshape(-1, V)
    targets_flat = targets.reshape(-1)
    return nn.losses.cross_entropy(logits_flat, targets_flat, reduction="mean")


def main():
    print("=" * 65)
    print("  télos Apple MLX Framework Throughput Benchmark")
    print("=" * 65)
    print(f"  Backend:    Apple MLX Metal Framework")
    print(f"  MLX Ver:    {mx.__version__}")

    V, d, layers, heads = 4096, 256, 4, 4
    seq = 512
    grad_accum = 2
    warmup, measure = 2, 5
    batch_sizes = [4, 8, 16, 32, 64]

    model = MLXTinyModel(V, d, layers, heads)
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())

    param_count = sum(p.size for p in tree_flatten(model.parameters()))

    print(f"  Model:      {param_count:,} params (d={d}, {layers}L, {heads}H)")
    print(f"  Precision:  bfloat16 (MLX Metal GPU)")
    print("=" * 65)

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    results = []

    for bs in batch_sizes:
        try:
            optimizer = optim.AdamW(learning_rate=1e-4)
            tokens = mx.random.randint(0, V, (bs, seq))
            targets = mx.random.randint(0, V, (bs, seq))
            mx.eval(tokens, targets)

            def step_fn(m, tok, tgt):
                loss, grads = loss_and_grad(m, tok, tgt)
                optimizer.update(m, grads)
                return loss

            # Warmup
            for _ in range(warmup):
                loss = step_fn(model, tokens, targets)
                mx.eval(model.parameters(), optimizer.state)

            # Measured steps
            t0 = time.perf_counter()
            for _ in range(measure):
                for _ in range(grad_accum):
                    loss = step_fn(model, tokens, targets)
                mx.eval(model.parameters(), optimizer.state)

            t1 = time.perf_counter()
            elapsed = t1 - t0
            sps = measure / elapsed
            tps = sps * bs * seq * grad_accum

            results.append((bs, bs * grad_accum, sps, tps, elapsed))
            print(f"  bs={bs:>3d} | eff_batch={bs*grad_accum:>3d} | {sps:>6.2f} steps/s | {tps:>10,.0f} tok/s | {elapsed:.2f}s")

        except Exception as e:
            print(f"  bs={bs:>3d} | Error: {e}")
            break

    print("\n" + "=" * 65)
    print("  APPLE MLX BATCH SIZE COMPARISON SUMMARY")
    print("=" * 65)
    print(f"  {'Batch':<8} {'Eff Batch':<10} {'Steps/sec':<12} {'Tok/sec':<14} {'Speed vs PyTorch MPS'}")
    print(f"  {'─'*7:<8} {'─'*9:<10} {'─'*11:<12} {'─'*13:<14} {'─'*19}")

    pytorch_mps_best = 149670  # Measured earlier in PyTorch MPS
    best_res = max(results, key=lambda x: x[3]) if results else None

    for bs, eff_b, sps, tps, el in results:
        ratio = tps / pytorch_mps_best
        star = " ★ BEST MLX" if best_res and bs == best_res[0] else ""
        print(f"  {bs:<8} {eff_b:<10} {sps:<12.2f} {tps:<14,.0f} {ratio:>5.2f}x vs PyTorch{star}")

    print("=" * 65)


def tree_flatten(params):
    if isinstance(params, dict):
        for v in params.values():
            yield from tree_flatten(v)
    elif isinstance(params, list):
        for v in params:
            yield from tree_flatten(v)
    elif hasattr(params, "size"):
        yield params


if __name__ == "__main__":
    main()
