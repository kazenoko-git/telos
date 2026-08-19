"""
MLX Causal Transformer Components for Autoregressive (AR) Language Modeling.

Architecture:
- RMSNorm pre-normalization
- SwiGLU MLP activation
- Rotary Position Embeddings (RoPE) via mx.fast.rope
- Grouped Query Attention (GQA) with Additive Causal Masking
"""

import mlx.core as mx
import mlx.nn as nn


class MLXRMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((d_model,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, self.weight, self.eps)


class MLXSwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network."""

    def __init__(self, d_model: int):
        super().__init__()
        hidden = int(d_model * 8 / 3)
        hidden = ((hidden + 63) // 64) * 64
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(d_model, hidden, bias=False)
        self.w3 = nn.Linear(hidden, d_model, bias=False)

    def __call__(self, x):
        return self.w3(nn.silu(self.w1(x)) * self.w2(x))


class MLXCausalBlock(nn.Module):
    """Transformer block with causal (lower-triangular) self-attention."""

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
        self._mask_cache = {}

    def __call__(self, x):
        B, T, D = x.shape
        h = self.norm1(x)

        q = self.q_proj(h).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(h).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(h).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Apply hardware-accelerated RoPE Metal kernel
        q = mx.fast.rope(q, self.head_dim, traditional=False, base=10000.0, scale=1.0, offset=0)
        k = mx.fast.rope(k, self.head_dim, traditional=False, base=10000.0, scale=1.0, offset=0)

        # Causal Attention Mask: cached lower-triangular mask
        cache_key = (T, str(q.dtype))
        if cache_key not in self._mask_cache:
            self._mask_cache[cache_key] = nn.MultiHeadAttention.create_additive_causal_mask(T).astype(q.dtype)
        causal_mask = self._mask_cache[cache_key]

        scale = 1.0 / (self.head_dim ** 0.5)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=causal_mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, D)

        x = x + self.out(out)
        x = x + self.mlp(self.norm2(x))
        return x


class MLXCausalTransformer(nn.Module):
    """Autoregressive Causal Transformer for next-token prediction."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        n_kv_heads: int,
        tied_embeddings: bool = True,
        use_grad_checkpoint: bool = False,
        **kwargs
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.tied_embeddings = tied_embeddings
        self.use_grad_checkpoint = use_grad_checkpoint
        self.emb = nn.Embedding(vocab_size, d_model)
        self.layers = [MLXCausalBlock(d_model, n_heads, n_kv_heads) for _ in range(n_layers)]
        self.norm = MLXRMSNorm(d_model)
        if not tied_embeddings:
            self.head = nn.Linear(d_model, vocab_size, bias=False)

    def hidden_states(self, x):
        x = self.emb(x)
        for layer in self.layers:
            if self.use_grad_checkpoint:
                x = mx.checkpoint(layer)(x)
            else:
                x = layer(x)
        return self.norm(x)

    def logits_from_hidden(self, hidden):
        if self.tied_embeddings:
            return self.emb.as_linear(hidden)
        return self.head(hidden)

    def __call__(self, x):
        return self.logits_from_hidden(self.hidden_states(x))
