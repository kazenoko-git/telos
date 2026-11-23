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

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask_override: bool | None = None,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_kv: bool = False,
        cache_position: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if return_kv:
            attn_out, present_kv = self.attn(
                self.attn_norm(x), cos, sin,
                is_causal_override=mask_override,
                kv_cache=kv_cache,
                return_kv=True,
                cache_position=cache_position,
            )
            x = x + attn_out
            x = x + self.mlp(self.mlp_norm(x))
            return x, present_kv
        x = x + self.attn(self.attn_norm(x), cos, sin, is_causal_override=mask_override, kv_cache=kv_cache)
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
                "use_reliability_head", "use_grad_checkpoint", "packed_projections"
            }
            cfg_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
            self.config = TelosConfig(**cfg_kwargs)
        else:
            self.config = config

        self.use_grad_checkpoint = getattr(self.config, "use_grad_checkpoint", False)
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

        self.tie_weights()
        self.apply(self._init_weights)
        if self.config.packed_projections:
            from .packed import pack_projections
            pack_projections(self)

    def tie_weights(self):
        """Explicitly re-ties output projection weights to token embeddings."""
        if self.config.tied_embeddings:
            self.output_projection.weight = self.tok_embeddings.weight

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def hidden_states(self, input_ids: torch.Tensor, mask_override: bool | None = None) -> torch.Tensor:
        """Run embedding, transformer blocks, and final RMSNorm without the LM head."""
        batch, seq_len = input_ids.shape
        assert seq_len <= self.config.max_seq_len, f"Sequence length {seq_len} exceeds max allowed {self.config.max_seq_len}"

        h = self.tok_embeddings(input_ids)
        h = self.dropout(h)
        cos, sin = self.rope(h, seq_len)
        cos = cos.to(dtype=h.dtype).unsqueeze(0).unsqueeze(0)
        sin = sin.to(dtype=h.dtype).unsqueeze(0).unsqueeze(0)

        use_checkpoint = self.use_grad_checkpoint and self.training and torch.is_grad_enabled()
        xla_checkpoint = None
        if use_checkpoint and h.device.type == "xla":
            try:
                from torch_xla.utils.checkpoint import checkpoint as xla_checkpoint
            except ImportError:
                xla_checkpoint = None

        for layer in self.layers:
            if use_checkpoint:
                if xla_checkpoint is not None:
                    h = xla_checkpoint(layer, h, cos, sin, mask_override)
                else:
                    h = torch.utils.checkpoint.checkpoint(
                        layer, h, cos, sin, mask_override,
                        use_reentrant=False
                    )
            else:
                h = layer(h, cos, sin, mask_override=mask_override)

        return self.final_norm(h)

    def hidden_states_with_cache(
        self,
        input_ids: torch.Tensor,
        kv_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        mask_override: bool | None = None,
        *,
        cache_position: int | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Run transformer blocks with incremental KV-cache for autoregressive decoding."""
        _, seq_len = input_ids.shape
        if cache_position is None:
            offset = 0 if kv_cache is None else kv_cache[0][0].shape[2]
        else:
            if (torch.is_grad_enabled() or self.training or not kv_cache or seq_len != 1
                    or not 0 <= cache_position < kv_cache[0][0].shape[2]):
                raise ValueError("A reserved KV cache requires gradient-free single-token decoding within capacity")
            offset = cache_position
        assert offset + seq_len <= self.config.max_seq_len, (
            f"Sequence length {offset + seq_len} exceeds max allowed {self.config.max_seq_len}"
        )

        h = self.tok_embeddings(input_ids)
        h = self.dropout(h)
        cos_full, sin_full = self.rope(h, offset + seq_len)
        cos = cos_full[offset : offset + seq_len].to(dtype=h.dtype).unsqueeze(0).unsqueeze(0)
        sin = sin_full[offset : offset + seq_len].to(dtype=h.dtype).unsqueeze(0).unsqueeze(0)

        new_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for idx, layer in enumerate(self.layers):
            layer_cache = None if kv_cache is None else kv_cache[idx]
            h, present_kv = layer(
                h, cos, sin,
                mask_override=mask_override,
                kv_cache=layer_cache,
                return_kv=True,
                cache_position=cache_position,
            )
            new_caches.append(present_kv)
        return self.final_norm(h), new_caches

    def reliability_scores(self, input_ids: torch.Tensor, mask_override: bool | None = None) -> torch.Tensor:
        """Compute scalar reliability scores without projecting vocabulary logits."""
        if self.reliability_head is None:
            raise ValueError("reliability_scores requested but use_reliability_head was not enabled.")
        h = self.hidden_states(input_ids, mask_override=mask_override)
        return self.reliability_head(h.detach()).squeeze(-1)

    def forward(self, input_ids: torch.Tensor, return_reliability: bool = False, mask_override: bool | None = None, next_token_only: bool = False, loss_targets: torch.Tensor | None = None, vocab_chunk_size: int = 256, vocab_chunk_axis: str = "vocab", *, logit_positions: torch.Tensor | None = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch = input_ids.shape[0]
        h = self.hidden_states(input_ids, mask_override=mask_override)
        if logit_positions is not None:
            # Rollouts only need one logits row per sequence, including when
            # prompts have different lengths and are padded on the right.
            projected_hidden = h[torch.arange(batch, device=h.device), logit_positions]
        else:
            projected_hidden = h[:, :-1] if next_token_only else h
        if loss_targets is not None:
            from .vocab_loss import tiled_cross_entropy
            return tiled_cross_entropy(projected_hidden, self.output_projection.weight,
                                       loss_targets, vocab_chunk_size, chunk_axis=vocab_chunk_axis)
        logits = self.output_projection(projected_hidden)
        
        if return_reliability:
            r_scores = self.reliability_head(h.detach()).squeeze(-1)
            return logits, r_scores
            
        return logits
