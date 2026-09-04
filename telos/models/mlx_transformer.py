"""
Unified MLX Transformer backbone for Télos.
"""

import mlx.core as mx
import mlx.nn as nn
from .mlx_components import MLXBlock, MLXRMSNorm

class MLXTelosTransformer(nn.Module):
    """Unified Telos Transformer for Apple Silicon (MLX)."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        tied_embeddings: bool = True,
        use_grad_checkpoint: bool = False,
        is_causal: bool = False,
        **kwargs
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.tied_embeddings = tied_embeddings
        self.use_grad_checkpoint = use_grad_checkpoint
        self.is_causal = is_causal
        
        # Default n_kv_heads to n_heads (standard Multi-Head Attention) if not explicitly set for GQA
        actual_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.emb = nn.Embedding(vocab_size, d_model)
        self.layers = [MLXBlock(d_model, n_heads, actual_kv_heads, is_causal=is_causal) for _ in range(n_layers)]
        self.norm = MLXRMSNorm(d_model)
        
        if not tied_embeddings:
            self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Scalar reliability head for COROSred
        self.reliability_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1)
        )

    def hidden_states(self, x, mask_override=None, is_causal: bool | None = None):
        effective_mask = mask_override
        if is_causal is not None:
            effective_mask = "causal" if is_causal else None

        x = self.emb(x)
        for layer in self.layers:
            if self.use_grad_checkpoint:
                x = mx.checkpoint(layer)(x, effective_mask)
            else:
                x = layer(x, effective_mask)
        return self.norm(x)

    def logits_from_hidden(self, hidden):
        if self.tied_embeddings:
            return self.emb.as_linear(hidden)
        return self.head(hidden)

    def __call__(self, x, return_reliability: bool = False, mask_override=None, is_causal: bool | None = None):
        h = self.hidden_states(x, mask_override=mask_override, is_causal=is_causal)
        logits = self.logits_from_hidden(h)
        if return_reliability:
            r_scores = mx.squeeze(self.reliability_head(mx.stop_gradient(h)), -1)
            return logits, r_scores
        return logits
