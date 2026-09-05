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

    def __post_init__(self):
        if self.seq_len is not None:
            self.max_seq_len = self.seq_len
