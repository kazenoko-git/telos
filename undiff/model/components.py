"""
basis for the telos bidirectional transformer backbone

includes:
- RMSNorm: root mean square normalization
- RoPE (Rotary Position Embeddings): relative positional encodings applied to Q/K
- SwiGLU
- BidirectionalAttention: multi-head attention without casual mask
"""

# imports

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """
    simplifies LayerNorm by eliminating mean-centering and relying
    solely on the root mean square of feature activations.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) # learned scaling parameter gamma initialized to ones

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS = sqrt(mean(x^2) + eps)
        # keepdim=True allows broadcasting across the feature dimension
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # normalize inputs and scale by learned weight
        return (x / rms) * self.weight


class RotaryEmbedding(nn.Module):
    """
    rotates query and key vectors in 2D pairs to encode relative position.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # calculate inverse frequency scale theta_i = 10000^(-2(i-1)/dim)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # precompute cos and sin embeddings for max sequence length
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        # generate position index tensor: [0, 1, 2, ..., seq_len - 1]
        t = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
        # outer product of position index and inv_freq: [seq_len, dim // 2]
        freqs = torch.outer(t, self.inv_freq)
        # duplicate frequencies for paired rotation: [seq_len, dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        # cache cosine and sine tables as buffers
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        # return sliced cosine and sine matrices up to current sequence length
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

# rotates input vector half-way across feature dimension for RoPE.
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[
    torch.Tensor, torch.Tensor]:
    """applies rotary position embeddings to query and key tensors.

    q, k shapes: [batch, n_heads, seq_len, head_dim]
    cos, sin shapes: [seq_len, head_dim] -> unsqueezed to [1, 1, seq_len, head_dim]
    """
    cos = cos.to(dtype=q.dtype).unsqueeze(0).unsqueeze(0)  # Broadcast for batch and head dims
    sin = sin.to(dtype=q.dtype).unsqueeze(0).unsqueeze(0)

    # Apply rotation formula: R_theta * x = x * cos + rotate_half(x) * sin
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class SwiGLU(nn.Module):
    """
    SwiGLU(x) = (x * W1) * SiLU(x * V) * W2
    SiLU(x * V) = (x * V)/[1 + e^-(x * V)]      dont understand these formulas but its ok
    hidden dim is set to roughly 8/3 * d_model rounded to nearest multiple of 64.
    """

    def __init__(self, d_model: int, hidden_dim: int | None = None):
        super().__init__()
        if hidden_dim is None:
            # default SwiGLU expansion is around 2.67x, but rounded to nearest 64 for the big GPU efficiency
            hidden_dim = int(2 * 4 * d_model / 3)
            hidden_dim = 64 * ((hidden_dim + 63) // 64)

        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)  # Gate projection
        self.v = nn.Linear(d_model, hidden_dim, bias=False)  # Up projection
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)  # Down projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # element-wise product of Gate (w1 with SiLU) and Up projection (v)
        return self.w2(F.silu(self.w1(x)) * self.v(x))


class BidirectionalAttention(nn.Module):
    """Multi-Head Self-Attention WITHOUT causal mask (Full Bidirectional).
    
    Supports:
    - Multi-Head Attention (MHA): n_kv_heads == n_heads
    - Grouped-Query Attention (GQA): n_kv_heads < n_heads (e.g. 8 query heads, 2 KV heads)
    - Multi-Query Attention (MQA): n_kv_heads == 1
    """
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int | None = None, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        assert n_heads % self.n_kv_heads == 0, f"n_heads ({n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
        
        self.num_queries_per_kv = n_heads // self.n_kv_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        # Q projection: d_model -> n_heads * head_dim
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        # K projection: d_model -> n_kv_heads * head_dim
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        # V projection: d_model -> n_kv_heads * head_dim
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        # Output projection: d_model -> d_model
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    # thank you claude
    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        # Linear projections
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        # Apply Rotary Positional Embeddings (RoPE) to Q and K
        q, k = apply_rope(q, k, cos, sin)

        # Repeat K and V heads if using GQA (n_kv_heads < n_heads)
        if self.num_queries_per_kv > 1:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        # Compute scaled dot-product attention WITHOUT causal mask
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False  # Full bidirectional attention for MDLM
        )

        # Reshape back to [batch, seq_len, d_model] and apply output projection
        out = out.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, self.d_model)
        return self.out_proj(out)