"""
Parameter counting utility and parameter solver for Télos models.
Accurately computes trainable parameters based on configuration, accounting for
weight-tied embeddings, SwiGLU expansion, and RMSNorm layers.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .transformer import TelosConfig


def count_parameters(config: TelosConfig) -> dict[str, int]:
    """Calculate exact parameter counts analytically from configuration."""
    v = config.vocab_size
    d = config.d_model
    l = config.n_layers
    h = config.n_heads
    head_dim = d // h
    n_kv = h if config.n_kv_heads is None else config.n_kv_heads

    # token embeddings: V * d
    embedding_params = v * d

    # attention per block:
    # Q proj: d * d
    # K proj: d * (n_kv * head_dim)
    # V proj: d * (n_kv * head_dim)
    # Out proj: d * d
    attn_params_per_layer = (d * d) + 2 * (d * (n_kv * head_dim)) + (d * d)

    # SwiGLU MLP per block:
    hidden_dim = int(2 * 4 * d / 3)
    hidden_dim = 64 * ((hidden_dim + 63) // 64)
    # W1 (gate), V (up), W2 (down): 3 * d * hidden_dim
    mlp_params_per_layer = 3 * d * hidden_dim

    # RMSNorms per block (2 norms)
    norm_params_per_layer = 2 * d

    total_per_layer = attn_params_per_layer + mlp_params_per_layer + norm_params_per_layer

    # final norm
    final_norm_params = d

    # output projection: 0 if tied, V * d if untied
    out_proj_params = 0 if config.tied_embeddings else (v * d)

    # reliability head: Linear(d, d) + SiLU + Linear(d, 1) -> (d*d + d) + (d + 1)
    rel_head_params = (d * d + 2 * d + 1) if config.use_reliability_head else 0

    total_params = embedding_params + (l * total_per_layer) + final_norm_params + out_proj_params + rel_head_params

    return {
        "embedding": embedding_params,
        "per_layer": total_per_layer,
        "all_layers": l * total_per_layer,
        "final_norm": final_norm_params,
        "output_proj": out_proj_params,
        "reliability_head": rel_head_params,
        "total": total_params
    }


def verify_with_model(config: TelosConfig) -> int:
    """Instantiates PyTorch model and counts numel() directly to verify formula."""
    from .transformer import TelosTransformer
    model = TelosTransformer(config)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
