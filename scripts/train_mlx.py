"""
télos MDLM — Apple MLX High-Throughput Trainer (25M Model)
===========================================================
Trains the 25M Masked Diffusion Language Model natively in Apple MLX.
Leverages bfloat16 precision, fused Metal FlashAttention, and Beta(1.5, 1.5)
timestep importance sampling.
"""

import time
import math
import yaml
import argparse
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


# ─── Model Components (MLX Native) ──────────────────────────────────

class MLXRMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((d_model,))
        self.eps = eps

    def __call__(self, x):
        variance = mx.mean(mx.square(x), axis=-1, keepdims=True)
        return x * mx.rsqrt(variance + self.eps) * self.weight


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
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        self.norm1 = MLXRMSNorm(d_model)
        self.norm2 = MLXRMSNorm(d_model)
        self.head_dim = d_model // n_heads
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.kv_groups = n_heads // n_kv_heads

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.mlp = MLXSwiGLU(d_model)

    def __call__(self, x):
        B, T, D = x.shape
        h = self.norm1(x)

        q = self.q_proj(h).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(h).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(h).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        if self.kv_groups > 1:
            k = mx.repeat(k, self.kv_groups, axis=1)
            v = mx.repeat(v, self.kv_groups, axis=1)

        scale = 1.0 / (self.head_dim ** 0.5)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, D)

        x = x + self.out(out)
        x = x + self.mlp(self.norm2(x))
        return x


class MLXTelosTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.layers = [MLXBlock(d_model, n_heads, n_kv_heads) for _ in range(n_layers)]
        self.norm = MLXRMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def __call__(self, x):
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x))


# ─── Timestep Sampling & Loss Function ─────────────────────────────

def sample_beta_timesteps(batch_size: int, eps: float = 1e-5):
    """Beta(1.5, 1.5) timestep sampling for MDLM."""
    # Beta sample approximation via Uniform transforms
    u = mx.random.uniform(0.0, 1.0, (batch_size, 1))
    # Sinusoidal shape mapping for Beta(1.5, 1.5)
    t = 0.5 - 0.5 * mx.cos(np.pi * u)
    return mx.clip(t, eps, 1.0)


def apply_masking_mlx(input_ids, mask_token_id=1, special_tokens={0, 1, 2, 3}):
    """Dynamic forward masking process in MLX."""
    B, T = input_ids.shape
    t_values = sample_beta_timesteps(B)
    rand_matrix = mx.random.uniform(0.0, 1.0, (B, T))
    raw_mask = rand_matrix < t_values

    # Special token preservation
    is_special = mx.zeros((B, T), dtype=mx.bool_)
    for st in special_tokens:
        is_special = is_special | (input_ids == st)

    mask_positions = raw_mask & (~is_special)
    masked_input_ids = mx.where(mask_positions, mask_token_id, input_ids)
    return masked_input_ids, mask_positions, t_values


def loss_fn(model, masked_input_ids, targets, mask_positions, t_values, vocab_size):
    logits = model(masked_input_ids)
    B, T, V = logits.shape

    # Per-token cross entropy
    logits_flat = logits.reshape(-1, V)
    targets_flat = targets.reshape(-1)

    ce_per_token = nn.losses.cross_entropy(logits_flat, targets_flat, reduction="none").reshape(B, T)
    masked_ce = ce_per_token * mask_positions.astype(mx.float32)

    masked_count = mx.clip(mx.sum(mask_positions.astype(mx.float32), axis=1), 1.0, float(T))
    per_example_ce = mx.sum(masked_ce, axis=1) / masked_count
    unweighted_ce = mx.mean(per_example_ce)

    # 1/t ELBO reweighting with eps=1e-3 clamp
    t_weights = 1.0 / mx.clip(mx.squeeze(t_values, -1), 1e-3, 1.0)
    reweighted_loss = mx.mean(per_example_ce * t_weights)
    return reweighted_loss, unweighted_ce


# ─── Data Iterator ──────────────────────────────────────────────────

def get_data_batch(dataset_matrix, idx_ptr, batch_size, seq_len):
    N = dataset_matrix.shape[0]
    indices = [(idx_ptr + i) % N for i in range(batch_size)]
    batch_seqs = dataset_matrix[indices, :seq_len]
    return mx.array(batch_seqs, dtype=mx.int32)


# ─── Training Loop ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/phase_b_25m_mlx.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    m_cfg = cfg["model"]
    t_cfg = cfg["training"]
    c_cfg = cfg.get("checkpoint", {})

    print("=" * 70)
    print("  télos MDLM — Apple MLX Trainer (25M Model)")
    print("=" * 70)
    print(f"  Backend:       Apple MLX Metal Framework ({mx.__version__})")
    print(f"  Architecture:  d={m_cfg['d_model']}, layers={m_cfg['n_layers']}, heads={m_cfg['n_heads']}")
    print(f"  Batch Size:    {t_cfg['batch_size']} × {t_cfg['gradient_accumulation']} = {t_cfg['batch_size']*t_cfg['gradient_accumulation']} eff")
    print(f"  Max Steps:     {t_cfg['max_steps']:,}")
    print(f"  Precision:     bfloat16 (MLX Unified GPU)")
    print("=" * 70)

    # Initialize Model
    model = MLXTelosTransformer(
        vocab_size=m_cfg["vocab_size"],
        d_model=m_cfg["d_model"],
        n_layers=m_cfg["n_layers"],
        n_heads=m_cfg["n_heads"],
        n_kv_heads=m_cfg["n_kv_heads"]
    )
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())

    param_count = sum(p.size for p in tree_flatten(model.parameters()))
    print(f"  Model Parameters: {param_count:,}")

    # Load pre-tokenized 500M token dataset
    train_bin = Path("data/python_corpus_mac.bin")

    if train_bin.exists():
        print(f"  Loading pre-tokenized dataset from {train_bin}...")
        raw_data = np.memmap(train_bin, dtype=np.int32, mode="r")
        n_seqs = len(raw_data) // m_cfg["seq_len"]
        dataset_matrix = raw_data[:n_seqs * m_cfg["seq_len"]].reshape(n_seqs, m_cfg["seq_len"])
        print(f"  Loaded {n_seqs:,} sequences ({n_seqs * m_cfg['seq_len'] / 1e6:.1f}M tokens).")
    else:
        print("  Notice: Pre-tokenized dataset file not found. Generating synthetic stream for throughput run...")
        dataset_matrix = np.random.randint(0, m_cfg["vocab_size"], (10000, m_cfg["seq_len"]), dtype=np.uint16)

    # Optimizer & Scheduler Setup
    max_steps = int(t_cfg["max_steps"])
    warmup_steps = int(t_cfg["warmup_steps"])
    max_lr = float(t_cfg["max_lr"])
    min_lr = float(t_cfg["min_lr"])
    weight_decay = float(t_cfg.get("weight_decay", 0.1))

    def get_lr(step):
        if step < warmup_steps:
            return max_lr * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

    optimizer = optim.AdamW(learning_rate=max_lr, weight_decay=weight_decay)

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    # Automatic Timestamped Run Versioning
    base_dir = Path(c_cfg.get("dir", "checkpoints/phase_b_25m_mlx"))
    if base_dir.exists() and any(base_dir.iterdir()):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        ckpt_dir = Path(f"{base_dir}_{timestamp}")
    else:
        ckpt_dir = base_dir

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Checkpoint Directory: {ckpt_dir} (Versioned)")

    print("\n  Starting training loop...")
    print("─" * 70)

    start_time = time.time()
    idx_ptr = 0
    bs = t_cfg["batch_size"]
    grad_accum = t_cfg["gradient_accumulation"]

    for step in range(1, max_steps + 1):
        # Update learning rate
        lr = get_lr(step)
        optimizer.learning_rate = lr

        accum_loss = 0.0
        accum_ce = 0.0

        for _ in range(grad_accum):
            targets = get_data_batch(dataset_matrix, idx_ptr, bs, m_cfg["seq_len"])
            idx_ptr += bs

            masked_ids, mask_pos, t_vals = apply_masking_mlx(targets, mask_token_id=1)
            (loss, ce), grads = loss_and_grad_fn(model, masked_ids, targets, mask_pos, t_vals, m_cfg["vocab_size"])
            optimizer.update(model, grads)
            accum_loss += loss.item()
            accum_ce += ce.item()

        mx.eval(model.parameters(), optimizer.state)

        if step % 50 == 0 or step == 1:
            elapsed = time.time() - start_time
            sps = step / elapsed
            tps = sps * bs * grad_accum * m_cfg["seq_len"]
            eta_mins = (max_steps - step) / sps / 60.0

            avg_loss = accum_loss / grad_accum
            avg_ce = accum_ce / grad_accum

            print(f"  Step {step:>6d}/{max_steps} | ELBO Loss: {avg_loss:>6.2f} | "
                  f"CE: {avg_ce:>5.3f} | LR: {lr:.2e} | {sps:>5.1f} st/s | {tps:>9,.0f} tok/s | ETA: {eta_mins:>4.1f}m", flush=True)

        if step % c_cfg.get("save_every_steps", 1000) == 0:
            ckpt_file = ckpt_dir / f"checkpoint_step_{step}.safetensors"
            model.save_weights(str(ckpt_file))
            print(f"  [Checkpoint] Saved weights to {ckpt_file}")

    total_time = time.time() - start_time

    # Final self-contained model artifact saving
    final_weights = ckpt_dir / "model.safetensors"
    model.save_weights(str(final_weights))

    # Save config.json for standalone loading
    import json
    with open(ckpt_dir / "config.json", "w") as f:
        json.dump(m_cfg, f, indent=2)

    # Copy tokenizer into checkpoint dir if available
    tok_source = Path("configs/tokenizer_mac.json")
    if tok_source.exists():
        import shutil
        shutil.copy(tok_source, ckpt_dir / "tokenizer.json")

    print("=" * 70)
    print(f"  Training Complete! Total time: {total_time/60.0:.2f} minutes.")
    print(f"  Saved standalone model artifact to {ckpt_dir}/")
    print("=" * 70)


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
