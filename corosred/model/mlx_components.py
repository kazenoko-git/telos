"""
MLX Unified Causal + Bidirectional Transformer for COROSred.

Architecture:
- RMSNorm pre-normalization
- SwiGLU Feed-Forward Network
- Rotary Position Embeddings (RoPE) via mx.fast.rope
- Grouped Query Attention (GQA) with dynamic causal draft vs. bidirectional refine masking
- 2-Layer MLP Reliability Head operating on pre-decision hidden states (h_{i-1})
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
        # Standard LLaMA-style hidden dimension scaling (8/3 * d_model aligned to 64 bytes)
        hidden = int(d_model * 8 / 3)
        hidden = ((hidden + 63) // 64) * 64
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(d_model, hidden, bias=False)
        self.w3 = nn.Linear(hidden, d_model, bias=False)

    def __call__(self, x):
        # SwiGLU gating: (silu(W1(x)) * W2(x)) * W3
        return self.w3(nn.silu(self.w1(x)) * self.w2(x))


class COROSredReliabilityHead(nn.Module):
    """
    2-Layer MLP predicting reliability score r_i from pre-decision hidden state h_{i-1}.
    """

    def __init__(self, d_model: int, hidden_dim: int = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = d_model
        # First layer projects hidden state to intermediate MLP dimension
        self.w1 = nn.Linear(d_model, hidden_dim, bias=True)
        # Second layer projects to a scalar reliability logit
        self.w2 = nn.Linear(hidden_dim, 1, bias=True)

    def __call__(self, h):
        # SiLU activation between MLP layers
        return self.w2(nn.silu(self.w1(h)))


class COROSredBlock(nn.Module):
    """Transformer block with dynamic Causal (Draft) vs. Bidirectional (Refine) attention."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        self.norm1 = MLXRMSNorm(d_model)
        self.norm2 = MLXRMSNorm(d_model)
        self.head_dim = d_model // n_heads
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.kv_groups = n_heads // n_kv_heads

        # Fused QKV projection for single-dispatch Metal execution
        self.qkv_proj = nn.Linear(d_model, (n_heads + 2 * n_kv_heads) * self.head_dim, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.mlp = MLXSwiGLU(d_model)

        self._q_end = n_heads * self.head_dim
        self._k_end = self._q_end + n_kv_heads * self.head_dim

    def __call__(self, x, is_causal: bool = True):
        B, T, D = x.shape
        h = self.norm1(x)

        qkv = self.qkv_proj(h)
        q, k, v = mx.split(qkv, [self._q_end, self._k_end], axis=-1)
        q = q.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Apply rotary position embeddings using Apple Metal accelerated kernel
        q = mx.fast.rope(q, self.head_dim, traditional=False, base=10000.0, scale=1.0, offset=0)
        k = mx.fast.rope(k, self.head_dim, traditional=False, base=10000.0, scale=1.0, offset=0)

        scale = 1.0 / (self.head_dim ** 0.5)
        # Use native 'causal' string in draft mode, None (bidirectional) in refine mode
        mask_arg = "causal" if is_causal else None
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask_arg)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, D)

        x = x + self.out(out)
        x = x + self.mlp(self.norm2(x))
        return x


class COROSredTransformer(nn.Module):
    """
    COROSred Backbone:
    Unified single backbone capable of causal autoregressive drafting and bidirectional MDLM refinement.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        n_kv_heads: int,
        mask_token_id: int = 0,
        tied_embeddings: bool = True,
        use_grad_checkpoint: bool = False,
        **kwargs
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.mask_token_id = mask_token_id
        self.tied_embeddings = tied_embeddings
        self.use_grad_checkpoint = use_grad_checkpoint

        self.emb = nn.Embedding(vocab_size, d_model)
        self.layers = [COROSredBlock(d_model, n_heads, n_kv_heads) for _ in range(n_layers)]
        self.norm = MLXRMSNorm(d_model)
        self.reliability_head = COROSredReliabilityHead(d_model)

        if not tied_embeddings:
            self.head = nn.Linear(d_model, vocab_size, bias=False)

    def hidden_states(self, x, is_causal: bool = True):
        x = self.emb(x)
        for layer in self.layers:
            if self.use_grad_checkpoint:
                # Gradient checkpointing per block to reduce peak activation memory
                x = mx.checkpoint(layer)(x, is_causal=is_causal)
            else:
                x = layer(x, is_causal=is_causal)
        return self.norm(x)

    def logits_from_hidden(self, hidden):
        if self.tied_embeddings:
            return self.emb.as_linear(hidden)
        return self.head(hidden)

    def __call__(self, x, is_causal: bool = True, return_reliability: bool = False):
        """
        Forward pass for both causal drafting and bidirectional refinement.

        Router Indexing Rule:
        Hidden state h[i] represents context up to token x[i]. The prediction for token x[i+1]
        is parameterized by h[i], and reliability_head(h[i]) is its pre-decision reliability score.
        """
        h = self.hidden_states(x, is_causal=is_causal)
        logits = self.logits_from_hidden(h)

        if return_reliability:
            # Squeeze output to shape [B, T]
            r_scores = self.reliability_head(h).squeeze(-1)
            return logits, r_scores

        return logits
