"""
Squared-ReLU FFN and XLA-safe Fixed-Capacity Mixture of Experts (MoE) for Télos.

Features:
- SquaredReLUMLP: 2-matrix ReLU^2 FFN (BitNet b1.58 2B4T style) with optional BitLinear + SubLN
- FixedCapacityMoE: Static-shape padded routing for PyTorch/XLA TPU v5e (zero dynamic shapes,
  O(E*C*D) scatter/gather dispatch without materializing O(M*E*C) tensors),
  contiguous 3D expert tensors [E, d_ff, D] / [E, D, d_ff], always-on shared expert,
  auxiliary-loss-free router bias balancing, and optional router z-loss.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .bitlinear import BitLinear, bitlinear_activation_quant, bitlinear_weight_quant
from .components import RMSNorm


class SquaredReLUMLP(nn.Module):
    """Two-matrix Squared-ReLU FFN: W_down(SubLN(ReLU(W_up(x))^2))."""

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        *,
        use_bitlinear: bool = False,
        use_subln: bool = False,
    ):
        super().__init__()
        if use_bitlinear:
            self.w_up = BitLinear(d_model, hidden_dim, bias=False, use_subln=False)
            self.w_down = BitLinear(hidden_dim, d_model, bias=False, use_subln=use_subln)
        else:
            self.w_up = nn.Linear(d_model, hidden_dim, bias=False)
            self.subln = RMSNorm(hidden_dim) if use_subln else None
            self.w_down = nn.Linear(hidden_dim, d_model, bias=False)
        self._has_bitlinear = use_bitlinear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.w_up(x))
        h = h * h
        if not self._has_bitlinear and getattr(self, "subln", None) is not None:
            h = self.subln(h)
        return self.w_down(h)


class FixedCapacityMoE(nn.Module):
    """XLA-safe static-shape Mixture of Experts with optional Shared Expert and BitLinear QAT."""

    def __init__(
        self,
        d_model: int,
        num_experts: int = 24,
        top_k: int = 2,
        d_ff: int = 2048,
        shared_d_ff: int | None = 2048,
        capacity_factor: float = 1.25,
        z_loss_coef: float = 1e-3,
        bias_update_rate: float = 1e-3,
        *,
        use_bitlinear: bool = False,
        use_subln: bool = False,
    ):
        super().__init__()
        if num_experts < 1:
            raise ValueError("num_experts must be >= 1")
        if not 1 <= top_k <= num_experts:
            raise ValueError(f"top_k ({top_k}) must be in [1, num_experts={num_experts}]")

        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_ff = d_ff
        self.capacity_factor = capacity_factor
        self.z_loss_coef = z_loss_coef
        self.bias_update_rate = bias_update_rate
        self.use_bitlinear = use_bitlinear
        self.use_subln = use_subln

        # Router stays in full precision (BF16/FP32) for clean gradient signal
        self.router = nn.Linear(d_model, num_experts, bias=False)
        # Auxiliary-loss-free load-balancing bias (DeepSeek-V3 style)
        self.register_buffer("expert_bias", torch.zeros(num_experts, dtype=torch.float32))

        # Contiguous 3D expert weights for static-shape batched einsum & easy EP/FSDP sharding
        self.w_up = nn.Parameter(torch.empty(num_experts, d_ff, d_model))
        self.w_down = nn.Parameter(torch.empty(num_experts, d_model, d_ff))
        nn.init.normal_(self.w_up, mean=0.0, std=0.02)
        nn.init.normal_(self.w_down, mean=0.0, std=0.02)

        self.expert_subln = RMSNorm(d_ff) if use_subln else None

        # Always-on shared expert amortizes bandwidth across MTP draft tokens
        if shared_d_ff is not None and shared_d_ff > 0:
            self.shared_expert = SquaredReLUMLP(
                d_model, shared_d_ff, use_bitlinear=use_bitlinear, use_subln=use_subln
            )
        else:
            self.shared_expert = None

        self.last_aux_loss: torch.Tensor | None = None
        self.last_expert_load: torch.Tensor | None = None

    def _compute_capacity(self, num_tokens: int) -> int:
        # Static capacity per expert derived purely from static token count N
        raw = math.ceil((num_tokens * self.top_k / self.num_experts) * self.capacity_factor)
        return max(1, raw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        n = b * t
        x_flat = x.reshape(n, d)

        # Router logits in FP32 for numerical stability
        router_logits = self.router(x_flat).float()  # (N, E)
        if self.z_loss_coef > 0.0 and self.training:
            lse = torch.logsumexp(router_logits, dim=-1)
            self.last_aux_loss = self.z_loss_coef * (lse * lse).mean()
        else:
            self.last_aux_loss = router_logits.new_zeros(())

        # Add load-balancing bias only for expert selection, not for gating weights
        biased_logits = router_logits + self.expert_bias.unsqueeze(0)
        _, topk_idx = torch.topk(biased_logits, k=self.top_k, dim=-1)  # (N, K)
        # Gather unbiased logits at selected experts and normalize via softmax
        selected_logits = torch.gather(router_logits, dim=-1, index=topk_idx)
        gate_weights = F.softmax(selected_logits, dim=-1).to(x.dtype)  # (N, K)

        # Compute per-expert counts and static slot indices without O(M*E*C) tensors
        m = n * self.top_k
        flat_expert = topk_idx.reshape(m)                                        # (M,)
        one_hot = F.one_hot(flat_expert, num_classes=self.num_experts).to(torch.int32)  # (M, E)
        expert_counts = one_hot.sum(dim=0).float()                               # (E,)
        self.last_expert_load = expert_counts.detach()

        # Auxiliary-loss-free bias update (sign-based, no gradients)
        if self.training and self.bias_update_rate > 0.0:
            with torch.no_grad():
                mean_load = expert_counts.mean()
                self.expert_bias.add_(self.bias_update_rate * torch.sign(mean_load - expert_counts))

        capacity = self._compute_capacity(n)
        # Exclusive prefix sum along M gives each routed token its 0-indexed slot in its assigned expert
        pos_matrix = torch.cumsum(one_hot, dim=0) - 1                            # (M, E)
        slot_pos = pos_matrix.gather(1, flat_expert.unsqueeze(1)).squeeze(1)     # (M,)
        valid_mask = slot_pos < capacity                                         # (M,)
        valid_f = valid_mask.to(x.dtype).unsqueeze(-1)                           # (M, 1)

        # Route overflow tokens into a single trailing dummy slot (index = E * C)
        total_slots = self.num_experts * capacity
        dest_idx = torch.where(
            valid_mask,
            flat_expert * capacity + slot_pos.clamp(0, capacity - 1),
            torch.full_like(flat_expert, total_slots),
        )  # (M,)

        x_rep = x_flat.unsqueeze(1).expand(n, self.top_k, d).reshape(m, d)
        dispatched_buf = x_flat.new_zeros((total_slots + 1, d))
        dispatched_buf = dispatched_buf.index_add(0, dest_idx, x_rep * valid_f)
        dispatched = dispatched_buf[:total_slots].view(self.num_experts, capacity, d)

        # Expert forward with optional BitLinear W1.58A8 QAT + SubLN
        if self.use_bitlinear:
            x_e = bitlinear_activation_quant(dispatched)
            w_up_q = bitlinear_weight_quant(self.w_up)
            h_e = F.relu(torch.einsum("ecd,efd->ecf", x_e, w_up_q))
            h_e = h_e * h_e
            if self.expert_subln is not None:
                h_e = self.expert_subln(h_e)
            h_e_q = bitlinear_activation_quant(h_e)
            w_down_q = bitlinear_weight_quant(self.w_down)
            y_e = torch.einsum("ecf,edf->ecd", h_e_q, w_down_q)
        else:
            h_e = F.relu(torch.einsum("ecd,efd->ecf", dispatched, self.w_up))
            h_e = h_e * h_e
            if self.expert_subln is not None:
                h_e = self.expert_subln(h_e)
            y_e = torch.einsum("ecf,edf->ecd", h_e, self.w_down)

        # Gather back from (E*C + 1, D) -> (M, D), zero overflow, weight by gate_weights
        y_buf = torch.cat((y_e.reshape(total_slots, d), y_e.new_zeros((1, d))), dim=0)
        y_rep = y_buf.index_select(0, dest_idx) * valid_f
        y_routed = (y_rep.view(n, self.top_k, d) * gate_weights.unsqueeze(-1)).sum(dim=1)
        y_out = y_routed.view(b, t, d)

        if self.shared_expert is not None:
            y_out = y_out + self.shared_expert(x)

        return y_out
