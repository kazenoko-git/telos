"""
Unified PyTorch Transformer backbone for Télos.
"""

import math
import torch
import torch.nn as nn
from .components import RMSNorm, RotaryEmbedding, SwiGLU, Attention
from .config import TelosConfig



class TransformerBlock(nn.Module):
    """Transformer block with Pre-RMSNorm and Bidirectional/Causal attention."""

    def __init__(self, config: TelosConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = Attention(config.d_model, config.n_heads, config.n_kv_heads, config.dropout, is_causal=config.is_causal)
        self.mlp_norm = RMSNorm(config.d_model)
        self.mlp = SwiGLU(config.d_model)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask_override: bool | None = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin, is_causal_override=mask_override)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class TelosTransformer(nn.Module):
    """Unified Telos Transformer (AR, MDLM, UNDLM, COROSred)."""

    def __init__(self, config: TelosConfig | None = None, **kwargs):
        super().__init__()
        if config is None:
            # Filter kwargs to match TelosConfig fields
            valid_keys = {
                "vocab_size", "d_model", "n_layers", "n_heads", "n_kv_heads",
                "max_seq_len", "seq_len", "dropout", "tied_embeddings", "is_causal",
                "use_reliability_head"
            }
            cfg_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
            self.config = TelosConfig(**cfg_kwargs)
        else:
            self.config = config

        self.tok_embeddings = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.dropout = nn.Dropout(self.config.dropout)
        self.rope = RotaryEmbedding(dim=self.config.d_model // self.config.n_heads, max_seq_len=self.config.max_seq_len)
        self.layers = nn.ModuleList([TransformerBlock(self.config) for _ in range(self.config.n_layers)])
        self.final_norm = RMSNorm(self.config.d_model)
        self.output_projection = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)
        
        # Scalar reliability head for COROSred (only instantiated when enabled)
        if self.config.use_reliability_head:
            self.reliability_head = nn.Sequential(
                nn.Linear(self.config.d_model, self.config.d_model),
                nn.SiLU(),
                nn.Linear(self.config.d_model, 1)
            )
        else:
            self.reliability_head = None

        if self.config.tied_embeddings:
            self.output_projection.weight = self.tok_embeddings.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, return_reliability: bool = False, mask_override: bool | None = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len = input_ids.shape
        assert seq_len <= self.config.max_seq_len, f"Sequence length {seq_len} exceeds max allowed {self.config.max_seq_len}"

        h = self.tok_embeddings(input_ids)
        h = self.dropout(h)
        cos, sin = self.rope(h, seq_len)

        for layer in self.layers:
            h = layer(h, cos, sin, mask_override=mask_override)

        h = self.final_norm(h)
        logits = self.output_projection(h)
        
        if return_reliability:
            r_scores = self.reliability_head(h.detach()).squeeze(-1)
            return logits, r_scores
            
        return logits
