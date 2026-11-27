"""
Unified PyTorch Transformer backbone for Télos.

Supports:
- Dense Transformer (AR, MDLM, UNDLM, COROSred)
- v1.9.2 Hybrid Recurrent (Gated DeltaNet) + Sliding-Window GQA
- v1.9.2 BitNet b1.58 (W1.58A8) QAT + SubLN
- v1.9.2 XLA-safe Fixed-Capacity MoE (ReLU^2 + Shared Expert)
- v1.9.2 Multi-Token Prediction (MTP) heads
"""

from dataclasses import fields
import torch
import torch.nn as nn
from .components import RMSNorm, RotaryEmbedding, SwiGLU, Attention
from .config import TelosConfig


def _build_ffn(config: TelosConfig) -> nn.Module:
    ffn_type = getattr(config, "ffn_type", "swiglu")
    use_bitlinear = bool(getattr(config, "use_bitlinear", False))
    use_subln = bool(getattr(config, "use_subln", False))

    if ffn_type == "moe_relu2" or getattr(config, "moe_num_experts", 0) > 0:
        from .moe import FixedCapacityMoE
        d_ff = config.moe_d_ff if config.moe_d_ff is not None else (config.d_ff or 2 * config.d_model)
        return FixedCapacityMoE(
            d_model=config.d_model,
            num_experts=config.moe_num_experts,
            top_k=config.moe_top_k,
            d_ff=d_ff,
            shared_d_ff=config.moe_shared_d_ff,
            capacity_factor=config.moe_capacity_factor,
            z_loss_coef=config.moe_z_loss_coef,
            bias_update_rate=config.moe_bias_update_rate,
            use_bitlinear=use_bitlinear,
            use_subln=use_subln,
        )
    if ffn_type == "relu2":
        from .moe import SquaredReLUMLP
        d_ff = config.d_ff if config.d_ff is not None else 2 * config.d_model
        return SquaredReLUMLP(
            config.d_model, d_ff, use_bitlinear=use_bitlinear, use_subln=use_subln
        )
    return SwiGLU(config.d_model, hidden_dim=config.d_ff)


class TransformerBlock(nn.Module):
    """Unified Transformer / Hybrid Recurrent + SWA block with optional MoE and BitLinear QAT."""

    def __init__(self, config: TelosConfig, layer_type: str = "attention"):
        super().__init__()
        self.layer_type = layer_type
        self.attn_norm = RMSNorm(config.d_model)

        if layer_type == "gdn":
            from .recurrent import GatedDeltaNet
            gdn_heads = config.gdn_n_heads if config.gdn_n_heads is not None else config.n_heads
            self.attn = GatedDeltaNet(
                d_model=config.d_model,
                n_heads=gdn_heads,
                head_dim=config.gdn_head_dim,
                conv_kernel=config.gdn_conv_kernel,
                chunk_size=config.gdn_chunk_size,
                use_bitlinear=config.use_bitlinear,
                use_subln=config.use_subln,
            )
        else:
            window = config.sliding_window if layer_type in ("swa", "attention") else None
            self.attn = Attention(
                config.d_model,
                config.n_heads,
                config.n_kv_heads,
                config.dropout,
                is_causal=config.is_causal,
                sliding_window=window,
                use_bitlinear=config.use_bitlinear,
                use_subln=config.use_subln,
            )

        self.mlp_norm = RMSNorm(config.d_model)
        self.mlp = _build_ffn(config)

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
        if self.layer_type == "gdn":
            if return_kv:
                mixer_out, present_state = self.attn(
                    self.attn_norm(x),
                    state=kv_cache,
                    return_state=True,
                )
                x = x + mixer_out
                x = x + self.mlp(self.mlp_norm(x))
                return x, present_state
            x = x + self.attn(self.attn_norm(x), state=kv_cache)
            x = x + self.mlp(self.mlp_norm(x))
            return x

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
    """Unified Telos Transformer (AR, MDLM, UNDLM, COROSred, BitMoE-Hybrid-MTP)."""

    def __init__(self, config: TelosConfig | None = None, **kwargs):
        super().__init__()
        if config is None:
            valid_keys = {f.name for f in fields(TelosConfig)}
            cfg_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
            self.config = TelosConfig(**cfg_kwargs)
        else:
            self.config = config

        self.use_grad_checkpoint = getattr(self.config, "use_grad_checkpoint", False)
        self.tok_embeddings = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.dropout = nn.Dropout(self.config.dropout)
        self.rope = RotaryEmbedding(dim=self.config.d_model // self.config.n_heads, max_seq_len=self.config.max_seq_len)

        layer_types = self.config.resolved_layer_types()
        self.layers = nn.ModuleList([
            TransformerBlock(self.config, layer_type=lt) for lt in layer_types
        ])
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

        # Multi-Token Prediction (MTP) heads (only instantiated when mtp_num_heads > 0)
        if getattr(self.config, "mtp_num_heads", 0) > 0:
            from .mtp import MultiTokenPrediction
            self.mtp = MultiTokenPrediction(
                num_heads=self.config.mtp_num_heads,
                d_model=self.config.d_model,
                d_ff=self.config.mtp_d_ff,
                loss_weight=self.config.mtp_loss_weight,
                use_bitlinear=self.config.use_bitlinear,
                use_subln=self.config.use_subln,
            )
        else:
            self.mtp = None

        self.tie_weights()
        self.apply(self._init_weights)
        # Restore GDN gate biases after generic _init_weights zeroing
        for layer in self.layers:
            if layer.layer_type == "gdn":
                nn.init.constant_(layer.attn.alpha_proj.bias, 3.0)
                nn.init.constant_(layer.attn.beta_proj.bias, -1.0)

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

    def collect_aux_loss(self) -> torch.Tensor | None:
        """Sum auxiliary losses (e.g. MoE router z-loss) collected during the last forward pass."""
        total = None
        for layer in self.layers:
            aux = getattr(layer.mlp, "last_aux_loss", None)
            if aux is not None:
                total = aux if total is None else total + aux
        return total

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
        position_offset: int | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Run transformer / hybrid blocks with incremental KV/recurrent cache for decoding."""
        _, seq_len = input_ids.shape
        if position_offset is not None:
            offset = position_offset
        elif cache_position is None:
            if kv_cache is None:
                offset = 0
            else:
                # Find the first attention/SWA layer cache to infer token offset, else 0
                offset = 0
                for idx, layer in enumerate(self.layers):
                    if layer.layer_type != "gdn" and kv_cache[idx] is not None:
                        offset = kv_cache[idx][0].shape[2]
                        break
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

    def forward(
        self,
        input_ids: torch.Tensor,
        return_reliability: bool = False,
        mask_override: bool | None = None,
        next_token_only: bool = False,
        loss_targets: torch.Tensor | None = None,
        vocab_chunk_size: int = 256,
        vocab_chunk_axis: str = "vocab",
        *,
        logit_positions: torch.Tensor | None = None,
        return_mtp_loss: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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
            main_losses = tiled_cross_entropy(
                projected_hidden, self.output_projection.weight,
                loss_targets, vocab_chunk_size, chunk_axis=vocab_chunk_axis
            )
            if return_mtp_loss and self.mtp is not None:
                mtp_loss = self.mtp.compute_loss(
                    h, input_ids, self.tok_embeddings, self.output_projection.weight,
                    use_tiled_ce=True,
                    vocab_chunk_size=vocab_chunk_size,
                    vocab_chunk_axis=vocab_chunk_axis,
                )
                return main_losses, mtp_loss
            return main_losses
        logits = self.output_projection(projected_hidden)

        if return_mtp_loss and self.mtp is not None:
            mtp_loss = self.mtp.compute_loss(
                h, input_ids, self.tok_embeddings, self.output_projection.weight,
                use_tiled_ce=False,
            )
            return logits, mtp_loss

        if return_reliability:
            r_scores = self.reliability_head(h.detach()).squeeze(-1)
            return logits, r_scores

        return logits
