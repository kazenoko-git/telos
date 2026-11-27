"""
Model configuration dataclasses for Télos.
"""

from dataclasses import dataclass


@dataclass
class TelosConfig:
    """Model configuration parameters."""
    vocab_size: int = 8192       # Size of tokenizer vocabulary
    d_model: int = 512           # Hidden embedding dimension
    n_layers: int = 8            # Number of transformer block layers
    n_heads: int = 8             # Number of query attention heads
    n_kv_heads: int | None = None # Number of key/value heads for GQA (defaults to n_heads if None)
    max_seq_len: int = 512       # Maximum supported sequence length
    seq_len: int | None = None   # Alias for max_seq_len
    dropout: float = 0.0         # Attention dropout rate
    tied_embeddings: bool = True # Tie input embeddings with output linear projection
    is_causal: bool = False      # Causal autoregressive mask vs bidirectional diffusion
    use_reliability_head: bool = False  # Scalar reliability head for COROSred
    packed_projections: bool = False  # Pack QKV and SwiGLU GEMMs; canonical checkpoints
    use_grad_checkpoint: bool = False   # Enable gradient checkpointing across transformer layers

    # v1.9.2: BitNet b1.58 QAT, Hybrid Recurrent/SWA, Fixed-Capacity MoE, and MTP
    use_bitlinear: bool = False         # Enable W1.58A8 BitLinear QAT on projections & FFNs
    use_subln: bool = False             # Enable BitNet SubLN before O_proj and W_down
    ffn_type: str = "swiglu"            # "swiglu" | "relu2" | "moe_relu2"
    d_ff: int | None = None             # Optional explicit FFN hidden dim (for relu2/swiglu)
    sliding_window: int | None = None   # Sliding-window size for attention layers (None = full)
    layer_types: list[str] | None = None # Per-layer mixer type: "attention" | "swa" | "gdn"
    hybrid_swa_every: int = 0           # If >0 and layer_types is None: every Nth layer is "swa", else "gdn"
    gdn_n_heads: int | None = None      # Recurrent Gated DeltaNet head count (defaults to n_heads)
    gdn_head_dim: int | None = None     # Recurrent Gated DeltaNet head dim (defaults to d_model // gdn_n_heads)
    gdn_conv_kernel: int = 4            # Short causal depthwise conv kernel for GDN
    gdn_chunk_size: int = 64            # Chunk size for chunkwise parallel GDN

    moe_num_experts: int = 0            # Number of routed experts (0 = dense FFN)
    moe_top_k: int = 2                  # Active routed experts per token
    moe_d_ff: int | None = None         # Hidden dim per routed expert
    moe_shared_d_ff: int | None = None  # Hidden dim for always-on shared expert (0/None = disabled)
    moe_capacity_factor: float = 1.25   # Static capacity multiplier for XLA-safe dispatch
    moe_z_loss_coef: float = 1e-3       # Router z-loss coefficient
    moe_bias_update_rate: float = 1e-3  # Auxiliary-loss-free router bias update rate

    mtp_num_heads: int = 0              # Number of sequential Multi-Token Prediction heads
    mtp_d_ff: int | None = None         # Optional hidden dim for MTP head dense blocks
    mtp_loss_weight: float = 0.3        # Weight applied to averaged MTP cross-entropy loss

    def __post_init__(self):
        if self.seq_len is not None:
            self.max_seq_len = self.seq_len

    def resolved_layer_types(self) -> list[str]:
        """Return the per-layer sequence mixer type list of length `n_layers`."""
        if self.layer_types is not None:
            if len(self.layer_types) != self.n_layers:
                raise ValueError(
                    f"layer_types length ({len(self.layer_types)}) must match n_layers ({self.n_layers})"
                )
            return list(self.layer_types)
        if self.hybrid_swa_every and self.hybrid_swa_every > 0:
            return [
                "swa" if ((idx + 1) % self.hybrid_swa_every == 0) else "gdn"
                for idx in range(self.n_layers)
            ]
        return ["attention"] * self.n_layers
