"""M5 Pro Metal screening harness for post-v1.8.8 MLX optimizations.

Run locally in a terminal with Metal GPU access:
    .venv/bin/python scratch/bench_v189_m5pro.py
"""

import json
import time
import numpy as np
import mlx.core as mx
import mlx.nn as mx_nn

from telos.models.mlx_transformer import MLXTelosTransformer
from telos.diffusion.sampler import MLXMDLMSampler
from telos.posttrain.mlx_adapters import apply_lora, LoRALinear
from telos.training.mlx_losses import cross_entropy as mlx_ce


def bench_mlx(fn, warmup=2, iters=6):
    for _ in range(warmup):
        out = fn()
        mx.eval(out)
    mx.synchronize()
    mx.reset_peak_memory()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn()
        mx.eval(out)
        mx.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times)), float(mx.get_peak_memory() / (1024 * 1024))


def main():
    results = {"mlx_version": mx.__version__, "device": mx.device_info(mx.gpu)}
    mx.random.seed(42)
    model = MLXTelosTransformer(
        vocab_size=8192, d_model=512, n_layers=14, n_heads=8, n_kv_heads=8,
        tied_embeddings=True, is_causal=True, precision="bfloat16",
    )
    mx.eval(model.parameters())

    # 1. MLX KV-cached GRPO rollout vs uncached
    B, G, prompt_len, max_new = 4, 4, 64, 32
    prompts = np.random.RandomState(42).randint(4, 8192, size=(B, prompt_len), dtype=np.int32)
    prompts_rep = np.repeat(prompts, G, axis=0)

    def rollout_uncached():
        seqs = mx.array(prompts_rep)
        last = None
        for _ in range(max_new):
            h = model.hidden_states(seqs, is_causal=True)
            logits = model.logits_from_hidden(h[:, -1, :]).astype(mx.float32)
            tok = mx.argmax(logits, axis=-1, keepdims=True).astype(mx.int32)
            seqs = mx.concatenate([seqs, tok], axis=1)
            last = logits
            mx.eval(seqs)
        return last

    def rollout_cached():
        h_pre, cache = model.hidden_states_with_cache(mx.array(prompts), is_causal=True)
        logits = mx.repeat(model.logits_from_hidden(h_pre[:, -1, :]).astype(mx.float32), G, axis=0)
        cache = [(mx.repeat(k, G, axis=0), mx.repeat(v, G, axis=0)) for k, v in cache]
        tok = mx.argmax(logits, axis=-1, keepdims=True).astype(mx.int32)
        mx.eval(tok, cache)
        last = logits
        for _ in range(max_new - 1):
            h, cache = model.hidden_states_with_cache(tok, kv_cache=cache, is_causal=True)
            logits = model.logits_from_hidden(h[:, -1, :]).astype(mx.float32)
            tok = mx.argmax(logits, axis=-1, keepdims=True).astype(mx.int32)
            last = logits
            mx.eval(tok, cache)
        return last

    l_unc, l_cac = rollout_uncached(), rollout_cached()
    mx.eval(l_unc, l_cac)
    t_u, m_u = bench_mlx(rollout_uncached, warmup=1, iters=3)
    t_c, m_c = bench_mlx(rollout_cached, warmup=1, iters=5)
    results["mlx_grpo_kv_cache"] = {
        "uncached_ms": round(t_u, 2),
        "cached_ms": round(t_c, 2),
        "speedup": round(t_u / t_c, 2),
        "uncached_peak_mb": round(m_u, 2),
        "cached_peak_mb": round(m_c, 2),
        "final_logits_max_abs_diff": round(float(mx.max(mx.abs(l_unc - l_cac)).item()), 5),
    }

    # 2. MLXMDLMSampler masked-only head
    sampler = MLXMDLMSampler(model, mask_token_id=1, num_steps=32, temperature=0.8)
    t_s, m_s = bench_mlx(lambda: sampler.sample(seq_len=128, prompt_ids=prompts[0, :32]), warmup=1, iters=5)
    results["mlx_mdlm_sampler_masked_head"] = {
        "sample_ms": round(t_s, 2),
        "peak_mb": round(m_s, 2),
    }

    # 3. MLXLoRALinear scale-before-up-projection screening (not yet landed)
    model_q = MLXTelosTransformer(
        vocab_size=8192, d_model=512, n_layers=14, n_heads=8, n_kv_heads=8,
        tied_embeddings=True, is_causal=True, precision="bfloat16",
    )
    mx.eval(model_q.parameters())
    apply_lora(model_q, rank=16, alpha=32.0, dropout=0.0, quantized=True)
    mx.eval(model_q.parameters())
    x_q = mx.array(np.random.RandomState(7).randint(4, 8192, size=(4, 256), dtype=np.int32))
    y_q = mx.array(np.random.RandomState(8).randint(4, 8192, size=(4, 255), dtype=np.int32))
    vg = mx_nn.value_and_grad(model_q, lambda m: mx.mean(mlx_ce(m(x_q, is_causal=True, next_token_only=True), y_q)))
    t_lo, m_lo = bench_mlx(lambda: vg(model_q), warmup=2, iters=6)

    orig_call = LoRALinear.__call__
    def fast_call(self, x):
        out = self.base(x)
        if self.adapter_disabled:
            return out
        h = x if self.dropout.p == 0 else self.dropout(x)
        delta = ((h.astype(mx.float32) @ self.lora_a.T) * self.scale) @ self.lora_b.T
        return out + delta.astype(out.dtype)

    LoRALinear.__call__ = fast_call
    t_lf, m_lf = bench_mlx(lambda: vg(model_q), warmup=2, iters=6)
    LoRALinear.__call__ = orig_call
    results["mlx_lora_scale_before_up_screen"] = {
        "current_ms": round(t_lo, 2),
        "candidate_ms": round(t_lf, 2),
        "speedup": round(t_lo / t_lf, 3),
        "current_peak_mb": round(m_lo, 2),
        "candidate_peak_mb": round(m_lf, 2),
    }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
