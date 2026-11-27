"""
Gated DeltaNet (GDN) hybrid recurrent linear attention for Télos.

Designed for:
- Constant O(1) recurrent state size during long-context inference on Apple Silicon
- Static-shape chunkwise parallel execution on PyTorch/XLA TPU v5e MXUs
- Pure functional state returns for cheap speculative / MTP verification rollback
- Optional BitLinear W1.58A8 QAT on Q/K/V/G/O projections (gates stay in BF16/FP32)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bitlinear import BitLinear
from .components import RMSNorm


def _linear_cls(use_bitlinear: bool, use_subln: bool = False):
    if use_bitlinear:
        return lambda d_in, d_out: BitLinear(d_in, d_out, bias=False, use_subln=use_subln)
    return lambda d_in, d_out: nn.Linear(d_in, d_out, bias=False)


def _gdn_step_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact sequential Gated DeltaNet recurrence.

    Shapes:
        q, k, v: (B, H, T, D)
        alpha:   (B, H, T, 1)  retention gate in (0, 1)
        beta:    (B, H, T, 1)  delta write/erase step size in (0, 1)
        state:   (B, H, D, D)  optional initial recurrent memory matrix
    """
    b, h, t, d = q.shape
    s = torch.zeros((b, h, d, d), device=q.device, dtype=torch.float32) if state is None else state.float().clone()
    outs = []
    for idx in range(t):
        q_t = q[:, :, idx].float()        # (B, H, D)
        k_t = k[:, :, idx].float()        # (B, H, D)
        v_t = v[:, :, idx].float()        # (B, H, D)
        a_t = alpha[:, :, idx].float()    # (B, H, 1)
        b_t = beta[:, :, idx].float()     # (B, H, 1)

        # Decay previous state
        s = s * a_t.unsqueeze(-1)
        # Predict current value from decayed state: v_pred = k_t @ S
        v_pred = torch.einsum("bhd,bhde->bhe", k_t, s)
        # Delta rule update: S <- S + beta * k_t^T (v_t - v_pred)
        delta = b_t * (v_t - v_pred)
        s = s + torch.einsum("bhd,bhe->bhde", k_t, delta)
        # Read out: o_t = q_t @ S
        o_t = torch.einsum("bhd,bhde->bhe", q_t, s)
        outs.append(o_t.to(q.dtype))
    return torch.stack(outs, dim=2), s


def _invert_unit_lower_triangular_blocks(l_mat: torch.Tensor) -> torch.Tensor:
    """Exact pure-matmul block Schur-complement inverse of a unit lower-triangular matrix.

    For L = [[L11, 0], [L21, L22]], L^{-1} = [[L11^{-1}, 0], [-L22^{-1} L21 L11^{-1}, L22^{-1}]].
    Unlike Neumann/block-doubling powers A^{2^k} (which overflow when keys align and ||A|| > 1),
    every intermediate block here is the exact inverse of a contractive GDN sub-chunk (O(1) norm).
    Requires last two dims to be a power of 2; executes in log2(C) - 1 pure matmul levels.
    """
    c = l_mat.shape[-1]
    inv = torch.eye(c, device=l_mat.device, dtype=l_mat.dtype).expand_as(l_mat).clone()
    if c <= 1:
        return inv

    # Level 0 (size 1 -> 2): for each 2x2 block [[1, 0], [a, 1]], inverse is [[1, 0], [-a, 1]]
    idx_even = torch.arange(0, c, 2, device=l_mat.device)
    idx_odd = idx_even + 1
    inv[..., idx_odd, idx_even] = -l_mat[..., idx_odd, idx_even]

    # Bottom-up block merging: block size b = 2, 4, 8, ..., c // 2
    b_size = 2
    while b_size < c:
        next_b = 2 * b_size
        n_groups = c // next_b
        l_21_list = []
        inv_11_list = []
        inv_22_list = []
        for g_idx in range(n_groups):
            s0 = g_idx * next_b
            s1 = s0 + b_size
            s2 = s0 + next_b
            l_21_list.append(l_mat[..., s1:s2, s0:s1])
            inv_11_list.append(inv[..., s0:s1, s0:s1])
            inv_22_list.append(inv[..., s1:s2, s1:s2])
        # Stack all groups along a new axis and run ONE batched matmul per level
        l_21 = torch.stack(l_21_list, dim=-3)      # (..., n_groups, b_size, b_size)
        inv_11 = torch.stack(inv_11_list, dim=-3)  # (..., n_groups, b_size, b_size)
        inv_22 = torch.stack(inv_22_list, dim=-3)  # (..., n_groups, b_size, b_size)
        inv_21 = -torch.matmul(torch.matmul(inv_22, l_21), inv_11)

        new_inv = inv.clone()
        for g_idx in range(n_groups):
            s0 = g_idx * next_b
            s1 = s0 + b_size
            s2 = s0 + next_b
            new_inv[..., s1:s2, s0:s1] = inv_21[..., g_idx, :, :]
        inv = new_inv
        b_size = next_b
    return inv


def _gdn_chunkwise(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunkwise parallel Gated DeltaNet using WY representation in strict FP32.

    Exact mathematical equivalent to `_gdn_step_reference` (verified in unit tests),
    converting the intra-chunk delta rule into dense matrix multiplications that
    saturate TPU v5e MXUs with only `ceil(T / chunk_size)` inter-chunk state steps.
    """
    b, h, t, d = q.shape
    if t <= 1 or chunk_size <= 1:
        return _gdn_step_reference(q, k, v, alpha, beta, state=state)

    # Round chunk size up to a power of 2 for exact block Schur-complement inversion
    c_raw = min(chunk_size, t)
    c = 1 << (c_raw - 1).bit_length()
    pad = (-t) % c
    if pad:
        q_p = F.pad(q, (0, 0, 0, pad))
        k_p = F.pad(k, (0, 0, 0, pad))
        v_p = F.pad(v, (0, 0, 0, pad))
        alpha_p = F.pad(alpha, (0, 0, 0, pad), value=1.0)
        beta_p = F.pad(beta, (0, 0, 0, pad), value=0.0)
    else:
        q_p, k_p, v_p, alpha_p, beta_p = q, k, v, alpha, beta

    # Keep all delta-rule triangular inversion and state recurrence strictly in FP32
    # so BF16 autocast never downcasts the inverse or state matmuls.
    amp_device = q.device.type if q.device.type in ("cuda", "cpu", "xla", "mps") else "cpu"
    with torch.amp.autocast(device_type=amp_device, enabled=False):
        n_chunks = q_p.shape[2] // c
        q_c = q_p.float().view(b, h, n_chunks, c, d)
        k_c = k_p.float().view(b, h, n_chunks, c, d)
        v_c = v_p.float().view(b, h, n_chunks, c, d)
        # Cumulative log-decay within each chunk: g_i = sum_{m=0..i} log(alpha_m)
        log_a = torch.log(alpha_p.float().clamp(min=1e-6)).view(b, h, n_chunks, c)
        g = torch.cumsum(log_a, dim=-1)  # (B, H, N, C)
        beta_c = beta_p.float().view(b, h, n_chunks, c)

        # Strictly lower-triangular mask for intra-chunk dependencies (j < i)
        idx = torch.arange(c, device=q.device)
        strict_lower = idx[:, None] > idx[None, :]   # (C, C)
        causal_lower = idx[:, None] >= idx[None, :]  # (C, C)

        # Compute decay ratios exp(g_i - g_j) for j <= i (clamped for numerical safety)
        g_diff = (g.unsqueeze(-1) - g.unsqueeze(-2)).clamp(max=0.0)  # (B, H, N, C, C)
        decay_strict = torch.exp(g_diff.masked_fill(~strict_lower, -1e9)) * strict_lower
        decay_causal = torch.exp(g_diff.masked_fill(~causal_lower, -1e9)) * causal_lower

        # Intra-chunk Gram matrix K K^T weighted by beta_i * exp(g_i - g_j)
        kk = torch.matmul(k_c, k_c.transpose(-1, -2))  # (B, H, N, C, C)
        a_mat = beta_c.unsqueeze(-1) * decay_strict * kk  # strictly lower-triangular
        eye = torch.eye(c, device=q.device, dtype=torch.float32)
        inv_l = _invert_unit_lower_triangular_blocks(eye + a_mat)

        # Precompute W = (I + A)^{-1} (beta * exp(g) * K) and U = (I + A)^{-1} (beta * V)
        exp_g = torch.exp(g)                                    # (B, H, N, C)
        k_scaled = k_c * (beta_c * exp_g).unsqueeze(-1)         # (B, H, N, C, D)
        v_scaled = v_c * beta_c.unsqueeze(-1)                   # (B, H, N, C, D)

        w_c = torch.matmul(inv_l, k_scaled)
        u_c = torch.matmul(inv_l, v_scaled)

        # Intra-chunk attention matrix: (Q K^T) * causal decay
        qk = torch.matmul(q_c, k_c.transpose(-1, -2)) * decay_causal  # (B, H, N, C, C)

        # Inter-chunk recurrence (only n_chunks steps, e.g. 32 for T=2048, C=64!)
        s = torch.zeros((b, h, d, d), device=q.device, dtype=torch.float32) if state is None else state.float().clone()
        chunk_outs = []
        for ci in range(n_chunks):
            q_i = q_c[:, :, ci]           # (B, H, C, D)
            k_i = k_c[:, :, ci]           # (B, H, C, D)
            w_i = w_c[:, :, ci]           # (B, H, C, D)
            u_i = u_c[:, :, ci]           # (B, H, C, D)
            qk_i = qk[:, :, ci]           # (B, H, C, C)
            g_i = g[:, :, ci]             # (B, H, C)
            exp_g_i = exp_g[:, :, ci]     # (B, H, C)

            # Effective value updates incorporating previous chunk state S:
            # u_eff = u_i - w_i @ S
            u_eff = u_i - torch.matmul(w_i, s)  # (B, H, C, D)

            # Chunk output: inter-chunk (exp(g_i) * q_i @ S) + intra-chunk (qk_i @ u_eff)
            o_inter = torch.matmul(q_i * exp_g_i.unsqueeze(-1), s)
            o_intra = torch.matmul(qk_i, u_eff)
            chunk_outs.append((o_inter + o_intra).to(q.dtype))

            # Advance state to end of chunk:
            # S_next = exp(g_last) * S + (exp(g_last - g_j) * k_j)^T @ u_eff
            g_last = g_i[:, :, -1:]                                      # (B, H, 1)
            tail_decay = torch.exp((g_last - g_i).clamp(max=0.0))        # (B, H, C)
            k_tail = k_i * tail_decay.unsqueeze(-1)                      # (B, H, C, D)
            s = s * torch.exp(g_last).unsqueeze(-1) + torch.matmul(k_tail.transpose(-1, -2), u_eff)

        out = torch.cat(chunk_outs, dim=2)
        if pad:
            out = out[:, :, :t, :]
        return out, s


class GatedDeltaNet(nn.Module):
    """Hybrid-ready Gated DeltaNet recurrent sequence mixer."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        head_dim: int | None = None,
        conv_kernel: int = 4,
        chunk_size: int = 64,
        *,
        use_bitlinear: bool = False,
        use_subln: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim if head_dim is not None else d_model // n_heads
        self.inner_dim = self.n_heads * self.head_dim
        self.conv_kernel = conv_kernel
        self.chunk_size = chunk_size

        make_proj = _linear_cls(use_bitlinear, use_subln=False)
        make_out = _linear_cls(use_bitlinear, use_subln=use_subln)

        self.q_proj = make_proj(d_model, self.inner_dim)
        self.k_proj = make_proj(d_model, self.inner_dim)
        self.v_proj = make_proj(d_model, self.inner_dim)
        self.g_proj = make_proj(d_model, self.inner_dim)
        self.o_proj = make_out(self.inner_dim, d_model)

        # Keep scalar decay (alpha) and delta write (beta) gates in BF16/FP32 for stability
        self.alpha_proj = nn.Linear(d_model, n_heads, bias=True)
        self.beta_proj = nn.Linear(d_model, n_heads, bias=True)
        # Initialize alpha bias positive (~0.95 retention) so early training preserves memory
        nn.init.constant_(self.alpha_proj.bias, 3.0)
        nn.init.constant_(self.beta_proj.bias, -1.0)

        # Short causal depthwise conv1d over concatenated (q, k, v)
        self.conv_dim = 3 * self.inner_dim
        self.short_conv = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            kernel_size=conv_kernel,
            groups=self.conv_dim,
            padding=0,
            bias=True,
        )
        self.head_norm = RMSNorm(self.head_dim)

    def _apply_short_conv(
        self,
        qkv: torch.Tensor,
        conv_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Causal depthwise Conv1d with optional state carry-over.

        Args:
            qkv:        (B, T, 3 * inner_dim)
            conv_state: (B, 3 * inner_dim, conv_kernel - 1) or None
        Returns:
            out:            (B, T, 3 * inner_dim)
            new_conv_state: (B, 3 * inner_dim, conv_kernel - 1)
        """
        b, t, c = qkv.shape
        x = qkv.transpose(1, 2)  # (B, C, T)
        k_minus_1 = self.conv_kernel - 1
        if k_minus_1 == 0:
            return F.silu(self.short_conv(x)).transpose(1, 2), x[:, :, :0]

        if conv_state is None:
            padded = F.pad(x, (k_minus_1, 0))
        else:
            padded = torch.cat((conv_state.to(dtype=x.dtype), x), dim=-1)
        new_conv_state = padded[:, :, -k_minus_1:].detach()
        conv_out = F.silu(self.short_conv(padded))
        return conv_out.transpose(1, 2), new_conv_state

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_state: bool = False,
        use_chunkwise: bool | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        b, t, _ = x.shape
        conv_state, rec_state = (None, None) if state is None else state

        qkv = torch.cat((self.q_proj(x), self.k_proj(x), self.v_proj(x)), dim=-1)
        qkv, new_conv_state = self._apply_short_conv(qkv, conv_state=conv_state)
        q, k, v = qkv.split(self.inner_dim, dim=-1)

        q = q.view(b, t, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(b, t, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(b, t, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # L2-normalize Q and K for bounded delta-rule eigenvalues
        q = F.normalize(q.float(), p=2, dim=-1, eps=1e-6).to(x.dtype)
        k = F.normalize(k.float(), p=2, dim=-1, eps=1e-6).to(x.dtype)

        alpha = torch.sigmoid(self.alpha_proj(x).float()).permute(0, 2, 1).unsqueeze(-1)
        beta = torch.sigmoid(self.beta_proj(x).float()).permute(0, 2, 1).unsqueeze(-1)

        run_chunkwise = (t > 1) if use_chunkwise is None else use_chunkwise
        if run_chunkwise:
            out, new_rec_state = _gdn_chunkwise(q, k, v, alpha, beta, chunk_size=self.chunk_size, state=rec_state)
        else:
            out, new_rec_state = _gdn_step_reference(q, k, v, alpha, beta, state=rec_state)

        # Per-head RMSNorm + SiLU output gate
        out = self.head_norm(out.permute(0, 2, 1, 3))  # (B, T, H, D_h)
        out = out.reshape(b, t, self.inner_dim)
        out = out * F.silu(self.g_proj(x))
        proj = self.o_proj(out)

        if return_state:
            return proj, (new_conv_state, new_rec_state)
        return proj
