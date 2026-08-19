"""parameter counting utility and parameter solver for télos models.

accurately computes trainable parameters based on configuration, accounting for
weight-tied embeddings, SwiGLU expansion, and RMSNorm layers.
"""

from .transformer import TelosConfig, TelosTransformer


def count_parameters(config: TelosConfig) -> dict[str, int]:
    """calculate exact parameter counts analytically from configuration."""
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
    # W1 (gate), V (up), W2 (down): (d * hidden) + (d * hidden) + (hidden * d) = 3 * d * hidden
    mlp_params_per_layer = 3 * d * hidden_dim

    # RMSNorms per block
    norm_params_per_layer = 2 * d

    total_per_layer = attn_params_per_layer + mlp_params_per_layer + norm_params_per_layer

    # final norm
    final_norm_params = d

    # output projection: 0 if tied, V * d if untied
    out_proj_params = 0 if config.tied_embeddings else (v * d)

    total_params = embedding_params + (l * total_per_layer) + final_norm_params + out_proj_params

    return {
        "embedding": embedding_params,
        "per_layer": total_per_layer,
        "all_layers": l * total_per_layer,
        "final_norm": final_norm_params,
        "output_proj": out_proj_params,
        "total": total_params
    }


def verify_with_model(config: TelosConfig) -> int:
    """instantiates PyTorch model and counts numel() directly to verify formula."""
    model = TelosTransformer(config)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def solve_config(target_params: int, vocab_size: int = 8192, max_seq_len: int = 512) -> TelosConfig:
    """finds optimal (d_model, n_layers, n_heads) closest to target_params."""
    best_config = None
    best_diff = float("inf")

    # search candidate grid
    for d_model in range(128, 1536, 64):
        for n_layers in range(4, 32, 2):
            for n_heads in [4, 8, 12, 16]:
                if d_model % n_heads != 0:
                    continue
                cfg = TelosConfig(
                    vocab_size=vocab_size,
                    d_model=d_model,
                    n_layers=n_layers,
                    n_heads=n_heads,
                    max_seq_len=max_seq_len,
                    tied_embeddings=True
                )
                params = count_parameters(cfg)["total"]
                diff = abs(params - target_params)
                if diff < best_diff:
                    best_diff = diff
                    best_config = cfg

    return best_config
