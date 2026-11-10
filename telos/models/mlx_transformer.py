"""
Unified MLX Transformer backbone for Télos.
"""

from telos.training.core import metal_usable

if not metal_usable():
    raise ImportError("MLX is installed but Metal is unavailable in this session.")

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
        use_reliability_head: bool = False,
        precision: str = "bfloat16",
        **kwargs
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.tied_embeddings = tied_embeddings
        self.use_grad_checkpoint = use_grad_checkpoint
        self.is_causal = is_causal
        self.use_reliability_head = use_reliability_head
        
        # Default n_kv_heads to n_heads (standard Multi-Head Attention) if not explicitly set for GQA
        actual_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.emb = nn.Embedding(vocab_size, d_model)
        self.layers = [MLXBlock(d_model, n_heads, actual_kv_heads, is_causal=is_causal) for _ in range(n_layers)]
        self.norm = MLXRMSNorm(d_model)
        
        if not tied_embeddings:
            self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Scalar reliability head for COROSred (only when enabled)
        if use_reliability_head:
            self.reliability_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.Linear(d_model, 1)
            )
        else:
            self.reliability_head = None

        # Precision casting: cast model weights to bfloat16 for high-throughput Metal compute
        self.precision = precision
        if precision in ("bfloat16", "bf16"):
            self.to_bfloat16()

    def to_bfloat16(self):
        """Casts all float32 model parameters to bfloat16 to optimize memory and Metal compute throughput."""
        from mlx.utils import tree_map
        self.update(tree_map(lambda p: p.astype(mx.bfloat16) if isinstance(p, mx.array) and p.dtype == mx.float32 else p, self.parameters()))
        return self

    def load_weights(self, file_or_weights, strict: bool = True):
        """Load both current weights and legacy split SwiGLU checkpoints."""
        weights = file_or_weights
        if isinstance(weights, str):
            weights = list(mx.load(weights).items())
        else:
            weights = list(weights)

        packed = {}
        remaining = []
        for name, value in weights:
            if ".mlp.w1.weight" in name:
                packed[name.replace(".mlp.w1.weight", ".mlp.gate_up.weight")] = [value, None]
            elif ".mlp.w2.weight" in name:
                packed.setdefault(name.replace(".mlp.w2.weight", ".mlp.gate_up.weight"), [None, None])[1] = value
            else:
                remaining.append((name, value))
        for name, (gate, up) in packed.items():
            if gate is not None and up is not None:
                remaining.append((name, mx.concatenate((gate, up), axis=0)))

        return super().load_weights(remaining, strict=strict)

    def hidden_states(self, x, mask_override=None, is_causal: bool | None = None):
        effective_mask = mask_override
        if is_causal is not None:
            effective_mask = "causal" if is_causal else None

        x = self.emb(x)
        for layer in self.layers:
            if self.use_grad_checkpoint:
                # Parameters must be explicit checkpoint inputs. Capturing the
                # module alone disconnects its parameters from autodiff.
                def checkpointed(parameters, inputs, layer=layer):
                    layer.update(parameters)
                    return layer(inputs, effective_mask)

                x = mx.checkpoint(checkpointed)(layer.trainable_parameters(), x)
            else:
                x = layer(x, effective_mask)
        return self.norm(x)

    def logits_from_hidden(self, hidden):
        if self.tied_embeddings:
            return self.emb.as_linear(hidden)
        return self.head(hidden)

    def next_token_logits(self, input_ids):
        """Project only positions that have a next-token target."""
        hidden = self.hidden_states(input_ids)
        return self.logits_from_hidden(hidden[:, :-1, :])

    def output_cross_entropy(self, input_ids, targets, *, chunk_size=256):
        """Exact tied-head CE without building the full token-by-vocabulary matrix.

        The output projection is evaluated in vocabulary tiles. Checkpointed
        tile reductions recompute their logits in backward, keeping activation
        memory proportional to ``tokens * chunk_size``. This is experimental:
        it currently supports tied embeddings, class-index targets and
        first-order training only.
        """
        if not self.tied_embeddings:
            raise ValueError("Tiled output cross entropy currently requires tied embeddings")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")

        hidden = self.hidden_states(input_ids, is_causal=True)
        hidden = hidden.reshape(-1, self.d_model)
        targets = targets.reshape(-1)
        weight = self.emb.weight

        def tile_logsumexp(x, w):
            tile_logits = mx.matmul(x, w.T)
            return mx.logsumexp(tile_logits.astype(mx.float32), axis=-1)

        log_partition = None
        for start in range(0, self.vocab_size, chunk_size):
            tile = weight[start : start + chunk_size]
            tile_lse = mx.checkpoint(tile_logsumexp)(hidden, tile)
            log_partition = tile_lse if log_partition is None else mx.logaddexp(log_partition, tile_lse)

        target_weight = mx.take(weight, targets, axis=0).astype(mx.float32)
        target_logits = mx.sum(hidden.astype(mx.float32) * target_weight, axis=-1)
        return (log_partition - target_logits).reshape(input_ids.shape)

    def __call__(self, x, return_reliability: bool = False, mask_override=None, is_causal: bool | None = None):
        h = self.hidden_states(x, mask_override=mask_override, is_causal=is_causal)
        logits = self.logits_from_hidden(h)
        if return_reliability:
            if self.reliability_head is None:
                raise ValueError("return_reliability=True requested but use_reliability_head was not enabled.")
            r_scores = mx.squeeze(self.reliability_head(mx.stop_gradient(h)), -1)
            return logits, r_scores
        return logits
