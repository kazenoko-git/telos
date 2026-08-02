"""
télos MDLM — Apple MLX Optimization Benchmark
===============================================
Benchmarks:
1. Micro-Batch Sizes (bs=2, 4, 8, 16, 32, 64)
2. Head Slicing (Proposal 3: un-embed only masked tokens) vs Full Un-Embedding
3. Fused Metal FlashAttention (mx.fast.scaled_dot_product_attention)
"""

import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


class MLXBlock(nn.Module):
    def __init__(self, d_model: int = 512, n_heads: int = 8):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.head_dim = d_model // n_heads
        self.n_heads = n_heads

        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

    def __call__(self, x):
        B, T, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, T, 3, self.n_heads, self.head_dim)
        q = qkv[:, :, 0].transpose(0, 2, 1, 3)
        k = qkv[:, :, 1].transpose(0, 2, 1, 3)
        v = qkv[:, :, 2].transpose(0, 2, 1, 3)

        scale = 1.0 / (self.head_dim ** 0.5)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, D)

        x = x + self.out(out)
        x = x + self.mlp(self.norm2(x))
        return x


class MLXModel(nn.Module):
    def __init__(self, vocab_size: int = 8192, d_model: int = 512, n_layers: int = 6, n_heads: int = 8):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.layers = [MLXBlock(d_model, n_heads) for _ in range(n_layers)]
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward_hidden(self, x):
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

    def __call__(self, x):
        h = self.forward_hidden(x)
        return self.head(h)


def full_unembedding_loss(model, tokens, targets, mask_pos):
    logits = model(tokens)
    B, T, V = logits.shape
    ce = nn.losses.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), reduction="none").reshape(B, T)
    return mx.sum(ce * mask_pos) / mx.clip(mx.sum(mask_pos), 1.0, float(B * T))


def head_sliced_loss(model, tokens, targets, mask_pos):
    h = model.forward_hidden(tokens)
    B, T, D = h.shape
    h_flat = h.reshape(-1, D)
    targets_flat = targets.reshape(-1)

    # Convert mask positions for MLX indexing
    mask_np = np.array(mask_pos)
    idx_np = np.where(mask_np.reshape(-1))[0]

    if len(idx_np) == 0:
        idx_np = np.array([0], dtype=np.int32)

    idx_mx = mx.array(idx_np)
    h_masked = mx.take(h_flat, idx_mx, axis=0)
    targets_masked = mx.take(targets_flat, idx_mx, axis=0)

    logits_masked = model.head(h_masked)
    return mx.mean(nn.losses.cross_entropy(logits_masked, targets_masked, reduction="none"))


def main():
    print("=" * 70)
    print("  télos MLX Head Slicing & Batch Size Optimization Benchmark")
    print("=" * 70)
    print(f"  Backend:    Apple MLX Metal ({mx.__version__})")
    print(f"  Precision:  bfloat16 (Metal GPU)")
    print("=" * 70)

    V, d, layers, heads = 8192, 512, 6, 8
    seq = 512
    batch_sizes = [2, 4, 8, 16, 32]
    warmup, measure = 2, 5

    for mode_name, loss_func in [("Full Un-Embedding", full_unembedding_loss), ("Head Slicing (Proposal 3)", head_sliced_loss)]:
        print(f"\n  --- Mode: {mode_name} ---")
        print(f"  {'Batch':<7} {'Steps/sec':<12} {'Tok/sec':<14} {'Step Latency':<12}")
        print("  " + "─" * 45)

        for bs in batch_sizes:
            model = MLXModel(V, d, layers, heads)
            model.set_dtype(mx.bfloat16)
            optimizer = optim.AdamW(learning_rate=1e-4)

            loss_and_grad = nn.value_and_grad(model, loss_func)

            tokens = mx.random.randint(0, V, (bs, seq))
            targets = mx.random.randint(0, V, (bs, seq))
            mask_pos = mx.random.uniform(0, 1, (bs, seq)) > 0.5
            mx.eval(tokens, targets, mask_pos)

            def step_fn(t, y, m):
                l, g = loss_and_grad(model, t, y, m)
                optimizer.update(model, g)
                return l

            for _ in range(warmup):
                l = step_fn(tokens, targets, mask_pos)
                mx.eval(model.parameters(), optimizer.state)

            t0 = time.perf_counter()
            for _ in range(measure):
                l = step_fn(tokens, targets, mask_pos)
                mx.eval(model.parameters(), optimizer.state)
            t1 = time.perf_counter()

            avg_time = (t1 - t0) / measure
            sps = 1.0 / avg_time
            tps = sps * bs * seq

            print(f"  bs={bs:<4}  {sps:>7.2f} st/s   {tps:>10,.0f} tok/s   {avg_time*1000:>6.1f} ms")


if __name__ == "__main__":
    main()
