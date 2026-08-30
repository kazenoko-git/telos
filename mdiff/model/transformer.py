"""Bidirectional Transformer backbone for the télos MDLM.

Key characteristics:
- Full bidirectional attention (no causal mask)
- Rotary Positional Embeddings (RoPE)
- RMSNorm for layer normalization
- SwiGLU feed-forward networks
- Tied input/output embedding weights (halves embedding param footprint)
- NO timestep embedding (time-agnostic optimal ELBO per RADD/MDLM findings)
"""

import math
import torch
import torch.nn as nn
from dataclasses import dataclass
from .components import RMSNorm, RotaryEmbedding, SwiGLU, BidirectionalAttention


@dataclass
class TelosConfig:
    """Model configuration parameters."""
    vocab_size: int = 4096       # Size of tokenizer vocabulary
    d_model: int = 128           # Hidden embedding dimension
    n_layers: int = 6            # Number of transformer block layers
    n_heads: int = 4             # Number of query attention heads
    n_kv_heads: int | None = None # Number of key/value heads for GQA (defaults to n_heads if None)
    max_seq_len: int = 512       # Maximum supported sequence length
    seq_len: int | None = None   # Alias for max_seq_len
    dropout: float = 0.1         # Attention dropout rate
    mlp_type: str = "swiglu"     # MLP activation type ("swiglu" or "standard")
    tied_embeddings: bool = True # Tie input embeddings with output linear projection
    no_timestep_embed: bool = True # Omit timestep embedding (per RADD/MDLM paper)
    is_causal: bool = False      # Causal autoregressive mask vs bidirectional diffusion

    def __post_init__(self):
        if self.seq_len is not None:
            self.max_seq_len = self.seq_len


class TransformerBlock(nn.Module):
    """single transformer block with Pre-RMSNorm and Bidirectional/Causal attention."""

    def __init__(self, config: TelosConfig):
        super().__init__()
        # pre-attention RMSNorm
        self.attn_norm = RMSNorm(config.d_model)
        # Multi-Head / Grouped-Query Self-Attention (Bidirectional or Causal)
        self.attn = BidirectionalAttention(config.d_model, config.n_heads, config.n_kv_heads, config.dropout, is_causal=config.is_causal)
        
        # pre-MLP RMSNorm
        self.mlp_norm = RMSNorm(config.d_model)
        # swiGLU or standard MLP feed-forward block
        self.mlp = SwiGLU(config.d_model)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # pre-norm residual connection for attention
        x = x + self.attn(self.attn_norm(x), cos, sin)
        # pre-norm residual connection for MLP
        x = x + self.mlp(self.mlp_norm(x))
        return x


class TelosTransformer(nn.Module):
    """Full Bidirectional MDLM Transformer."""

    def __init__(self, config: TelosConfig):
        super().__init__()
        self.config = config

        # token embedding layer: maps token IDs to d_model vectors
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        # rotary Positional Embedding generator
        self.rope = RotaryEmbedding(
            dim=config.d_model // config.n_heads,
            max_seq_len=config.max_seq_len
        )

        # stack of N Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # final RMSNorm layer applied before output projection
        self.final_norm = RMSNorm(config.d_model)

        # output linear projection to vocabulary logits
        self.output_projection = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # scalar reliability head (for COROSred)
        self.reliability_head = nn.Linear(config.d_model, 1, bias=False)

        # weight tying: share weights between token embedding and output linear layer
        if config.tied_embeddings:
            self.output_projection.weight = self.tok_embeddings.weight

        # apply weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """xavier/normal initialization for weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, return_reliability: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """forward pass.
        
        Args:
            input_ids: Tensor of shape [batch_size, seq_len] containing token IDs
                       (which may include [MASK] tokens).
            return_reliability: Whether to return the (logits, reliability_scores) tuple.
                       
        Returns:
            logits: Unnormalized logits of shape [batch_size, seq_len, vocab_size].
            r_scores: (Optional) Reliability scores of shape [batch_size, seq_len].
        """
        batch, seq_len = input_ids.shape
        assert seq_len <= self.config.max_seq_len, \
            f"Sequence length {seq_len} exceeds max allowed {self.config.max_seq_len}"

        # lookup token embeddings: [batch, seq_len, d_model]
        h = self.tok_embeddings(input_ids)
        h = self.dropout(h)

        # get RoPE cos and sin tables sliced up to current sequence length
        cos, sin = self.rope(h, seq_len)

        for layer in self.layers:
            h = layer(h, cos, sin)

        h = self.final_norm(h)
        logits = self.output_projection(h)
        
        if return_reliability:
            r_scores = self.reliability_head(h).squeeze(-1)
            return logits, r_scores
            
        return logits
