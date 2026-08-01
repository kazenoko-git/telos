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
    cos = cos.unsqueeze(0).unsqueeze(0)  # Broadcast for batch and head dims
    sin = sin.unsqueeze(0).unsqueeze(0)

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
    """multi-head self-attention WITHOUT causal mask (Full Bidirectional).

    allows each token to attend to all other tokens in the sequence.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        # combined Q, K, V linear projections (no bias terms)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        # output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    # thank you claude
    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        # 1. Project input to Q, K, V tensors: [batch, seq_len, 3 * d_model]
        qkv = self.qkv_proj(x)
        # 2. Reshape & permute to [batch, n_heads, seq_len, head_dim]
        qkv = qkv.view(batch, seq_len, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # 3. Apply Rotary Positional Embeddings (RoPE) to Q and K
        q, k = apply_rope(q, k, cos, sin)
        # 4. Compute scaled dot-product attention WITHOUT causal mask (is_causal=False)
        # FlashAttention / Metal kernels selected automatically via F.scaled_dot_product_attention
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False  # CRITICAL: Full bidirectional attention for MDLM
        )
        # 5. Reshape back to [batch, seq_len, d_model]
        out = out.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, self.d_model)
        # 6. Apply final output projection
        return self.out_proj(out)