import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from telos.model.mlx_components import MLXTelosTransformer

def apply_masking_mlx(targets):
    B, T = targets.shape
    u = mx.random.uniform(0.0, 1.0, (B, 1))
    t_values = mx.clip(0.5 - 0.5 * mx.cos(np.pi * u), 1e-5, 1.0)
    rand_matrix = mx.random.uniform(0.0, 1.0, (B, T))
    mask_positions = rand_matrix < t_values
    masked_input_ids = mx.where(mask_positions, 1, targets)
    return masked_input_ids, mask_positions, t_values

def loss_fn(model, masked_input_ids, targets, mask_positions, t_values):
    logits = model(masked_input_ids)
    B, T, V = logits.shape
    logits_flat = logits.reshape(-1, V)
    targets_flat = targets.reshape(-1)
    ce_loss = nn.losses.cross_entropy(logits_flat, targets_flat)
    per_example_ce = mx.mean(ce_loss.reshape(B, T) * mask_positions, axis=-1)
    t_weights = 1.0 / mx.clip(mx.squeeze(t_values, -1), 1e-3, 1.0)
    reweighted_loss = mx.mean(per_example_ce * t_weights)
    return reweighted_loss, per_example_ce

def main():
    m_cfg = {"d_model": 512, "n_layers": 8, "n_heads": 8, "n_kv_heads": 2, "vocab_size": 8192}
    warmup_steps = 3
    num_steps = 15

    print("\n" + "=" * 85)
    print(f" TELOS MDLM — 25M MODEL FAST BENCHMARK SUITE (COMPILED GRAPH)")
    print("=" * 85)

    seq_len = 512
    batch_sizes = [16, 32, 64]
    
    print(f"{'BATCH SIZE':<12} | {'STEPS/SEC':<12} | {'TOKENS/SEC':<15} | {'LATENCY/STEP (ms)':<18}")
    print("-" * 85)

    for bs in batch_sizes:
        model = MLXTelosTransformer(**m_cfg)
        model.set_dtype(mx.bfloat16)
        mx.eval(model.parameters())

        optimizer = optim.AdamW(learning_rate=3e-4)
        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
        
        @mx.compile
        def compiled_step(mdl, tgts):
            masked_ids, mask_pos, t_vals = apply_masking_mlx(tgts)
            (loss, _), grads = loss_and_grad_fn(mdl, masked_ids, tgts, mask_pos, t_vals)
            optimizer.update(mdl, grads)
            return loss

        targets = mx.random.randint(0, 8192, (bs, seq_len))
        mx.eval(targets)

        for _ in range(warmup_steps):
            loss = compiled_step(model, targets)
            mx.eval(model.parameters(), optimizer.state, loss)

        mx.clear_cache()
        start = time.perf_counter()
        
        for _ in range(num_steps):
            loss = compiled_step(model, targets)
            mx.eval(model.parameters(), optimizer.state, loss)

        mx.clear_cache()
        elapsed = time.perf_counter() - start
        
        sps = num_steps / elapsed
        tps = sps * bs * seq_len
        latency_ms = (elapsed / num_steps) * 1000.0

        print(f"{bs:<12} | {sps:<12.2f} | {tps:<15,.0f} | {latency_ms:<18.2f}")

if __name__ == "__main__":
    main()
