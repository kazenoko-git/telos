"""
Multi-Token Prediction (MTP) heads for Télos (DeepSeek-V3 style).

Each MTP head k in {1..K} fuses the previous hidden state h_{t} with the
shifted token embedding e_{t+k} via a lightweight projection + dense block,
then shares the backbone's tied output projection (and tiled cross-entropy)
to predict token t+k+1.
"""

import torch
import torch.nn as nn

from .bitlinear import BitLinear
from .components import RMSNorm
from .moe import SquaredReLUMLP
from .vocab_loss import tiled_cross_entropy


class MTPHead(nn.Module):
    """Single sequential Multi-Token Prediction module."""

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        *,
        use_bitlinear: bool = False,
        use_subln: bool = False,
    ):
        super().__init__()
        self.h_norm = RMSNorm(d_model)
        self.e_norm = RMSNorm(d_model)
        if use_bitlinear:
            self.fuse_proj = BitLinear(2 * d_model, d_model, bias=False, use_subln=False)
        else:
            self.fuse_proj = nn.Linear(2 * d_model, d_model, bias=False)
        self.mlp_norm = RMSNorm(d_model)
        mlp_dim = d_ff if d_ff is not None else 2 * d_model
        self.mlp = SquaredReLUMLP(d_model, mlp_dim, use_bitlinear=use_bitlinear, use_subln=use_subln)
        self.out_norm = RMSNorm(d_model)

    def forward(self, prev_hidden: torch.Tensor, next_token_embed: torch.Tensor) -> torch.Tensor:
        fused = torch.cat((self.h_norm(prev_hidden), self.e_norm(next_token_embed)), dim=-1)
        h = self.fuse_proj(fused)
        h = h + self.mlp(self.mlp_norm(h))
        return self.out_norm(h)


class MultiTokenPrediction(nn.Module):
    """Stack of K sequential MTP heads sharing the backbone embeddings and LM head."""

    def __init__(
        self,
        num_heads: int,
        d_model: int,
        d_ff: int | None = None,
        loss_weight: float = 0.3,
        *,
        use_bitlinear: bool = False,
        use_subln: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.loss_weight = loss_weight
        self.heads = nn.ModuleList([
            MTPHead(d_model, d_ff=d_ff, use_bitlinear=use_bitlinear, use_subln=use_subln)
            for _ in range(num_heads)
        ])

    def compute_loss(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor,
        tok_embeddings: nn.Embedding,
        lm_head_weight: torch.Tensor,
        *,
        use_tiled_ce: bool = True,
        vocab_chunk_size: int = 256,
        vocab_chunk_axis: str = "vocab",
    ) -> torch.Tensor:
        """Compute averaged MTP cross-entropy loss across K future horizons.

        Given `hidden` of shape (B, T, D) aligned with `input_ids` (B, T):
        - Main AR loss predicts input_ids[:, 1:] from hidden[:, :-1].
        - MTP head k (1-indexed) takes h_prev[:, :-1] and embed(input_ids[:, k:-1])
          to predict input_ids[:, k+1:].
        """
        t = input_ids.shape[1]
        if self.num_heads <= 0 or t <= 2:
            return hidden.new_zeros(())

        # Start from backbone hidden states aligned with input_ids[:, :-1]
        h_curr = hidden[:, :-1]
        total_loss = hidden.new_zeros(())
        active_heads = 0

        for k, head in enumerate(self.heads, start=1):
            # Need at least 1 valid target token at offset k+1
            if k + 1 >= t:
                break
            # Shift h_curr to length (T - k - 1) and fuse with embed(input_ids[:, k : T - 1])
            h_in = h_curr[:, : t - k - 1]
            e_next = tok_embeddings(input_ids[:, k : t - 1])
            h_next = head(h_in, e_next)
            targets = input_ids[:, k + 1 :]

            if use_tiled_ce:
                tok_loss = tiled_cross_entropy(
                    h_next, lm_head_weight, targets,
                    chunk_size=vocab_chunk_size, chunk_axis=vocab_chunk_axis,
                )
                head_loss = tok_loss.mean()
            else:
                logits = torch.nn.functional.linear(h_next, lm_head_weight)
                head_loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]).float(),
                    targets.reshape(-1),
                )

            total_loss = total_loss + head_loss
            h_curr = h_next
            active_heads += 1

        if active_heads == 0:
            return hidden.new_zeros(())
        return self.loss_weight * (total_loss / active_heads)

    @torch.no_grad()
    def draft_tokens(
        self,
        last_hidden: torch.Tensor,
        last_token_id: torch.Tensor,
        tok_embeddings: nn.Embedding,
        lm_head_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Greedy-draft K future tokens from the last backbone hidden state.

        Args:
            last_hidden:   (B, 1, D)
            last_token_id: (B, 1)  token just sampled from main LM head
        Returns:
            drafted:       (B, K)  drafted future token IDs
        """
        h_curr = last_hidden
        tok_curr = last_token_id
        drafted = []
        for head in self.heads:
            e_curr = tok_embeddings(tok_curr)
            h_curr = head(h_curr, e_curr)
            logits = torch.nn.functional.linear(h_curr, lm_head_weight)
            tok_curr = logits.argmax(dim=-1)
            drafted.append(tok_curr)
        return torch.cat(drafted, dim=-1)
