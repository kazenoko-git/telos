#!/usr/bin/env python3
"""Unified architecture and weight loader for DataDecide 530M models (AR & MDLM).

Both models share the OLMo-1 530M backbone:
- Hidden dimension: 1344
- Layers: 16
- Heads: 16 (dim 84)
- SwiGLU MLP (ratio 8: intermediate 10752 / projection 5376)
- RMSNorm (eps 1e-6)
- RoPE (theta 10000)
- Vocabulary: 50280 (embeddings padded to 50304)
- Mask token ID: 50280 (MDLM)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AR_WEIGHTS = (
    Path.home()
    / ".cache/huggingface/hub/models--michaelchenkj--datadecide-dclm-530m-53b/snapshots/213f7e96745ee3d397834821fe2b397019bdde98/weights/step57786/model.safetensors"
)
DEFAULT_MDLM_WEIGHTS = ROOT / "scratch/matched_530m/mdlm_step57786.safetensors"


@dataclass
class DCLMConfig:
    d_model: int = 1344
    n_heads: int = 16
    n_layers: int = 16
    mlp_ratio: int = 8
    vocab_size: int = 50280
    embedding_size: int = 50304
    mask_id: int = 50280
    eos_id: int = 50279
    pad_id: int = 1
    max_sequence_length: int = 2048
    rope_theta: float = 10000.0
    ln_eps: float = 1e-6

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def hidden_size(self) -> int:
        return self.mlp_ratio * self.d_model


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization with affine gain."""

    def __init__(self, size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMSNorm computed in float32 for numerical stability
        orig_dtype = x.dtype
        x32 = x.float()
        normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * normed).to(orig_dtype)


class RotaryEmbedding(nn.Module):
    """OLMo / GPT-NeoX rotary position embedding."""

    def __init__(self, cfg: DCLMConfig):
        super().__init__()
        self.dim = cfg.head_dim
        self.theta = cfg.rope_theta
        self.max_seq = cfg.max_sequence_length
        # Inverse frequencies for half of head dimensions
        inv = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv, persistent=False)
        t = torch.arange(self.max_seq).float()
        freqs = torch.outer(t, inv)
        # Duplicate frequencies to match full head_dim
        positions = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", positions.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin", positions.sin()[None, None, :, :], persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        # Split into consecutive pairs and rotate: (-x2, x1)
        b, nh, t, hs = x.size()
        x = x.view(b, nh, t, 2, hs // 2)
        x1, x2 = x.unbind(dim=-2)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t = k.size(-2)
        cos = self.cos[:, :, :t, :].to(dtype=q.dtype, device=q.device)
        sin = self.sin[:, :, :t, :].to(dtype=q.dtype, device=q.device)
        q = (q * cos) + (self._rotate_half(q) * sin)
        k = (k * cos) + (self._rotate_half(k) * sin)
        return q, k


class DCLMBlock(nn.Module):
    """Transformer block with pre-RMSNorm, RoPE, SDPA, and SwiGLU MLP."""

    def __init__(self, cfg: DCLMConfig):
        super().__init__()
        self.cfg = cfg
        hidden = cfg.hidden_size
        self.attn_norm = RMSNorm(cfg.d_model, cfg.ln_eps)
        self.att_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.attn_out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.ff_norm = RMSNorm(cfg.d_model, cfg.ln_eps)
        self.ff_proj = nn.Linear(cfg.d_model, hidden, bias=False)
        self.ff_out = nn.Linear(hidden // 2, cfg.d_model, bias=False)
        self.rotary = RotaryEmbedding(cfg)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        cfg = self.cfg
        b, t, _ = x.size()
        h = self.attn_norm(x)
        qkv = self.att_proj(h)
        q, k, v = qkv.chunk(3, dim=-1)
        # Reshape to (batch, heads, seq, head_dim)
        q = q.view(b, t, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        k = k.view(b, t, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        v = v.view(b, t, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        q, k = self.rotary(q, k)
        # Scaled dot-product attention
        att = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=is_causal
        )
        att = att.transpose(1, 2).contiguous().view(b, t, cfg.d_model)
        x = x + self.attn_out(att)

        # SwiGLU feedforward network: silu(gate) * linear_proj
        h = self.ff_norm(x)
        h = self.ff_proj(h)
        a, gate = h.chunk(2, dim=-1)
        h = F.silu(gate) * a
        return x + self.ff_out(h)


class DCLMModel(nn.Module):
    """Complete OLMo-1 530M model supporting both AR and Diffusion execution."""

    def __init__(self, cfg: Optional[DCLMConfig] = None):
        super().__init__()
        self.cfg = cfg or DCLMConfig()
        cfg = self.cfg
        self.wte = nn.Embedding(cfg.embedding_size, cfg.d_model)
        self.blocks = nn.ModuleList(DCLMBlock(cfg) for _ in range(cfg.n_layers))
        self.ln_f = RMSNorm(cfg.d_model, cfg.ln_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(
        self,
        idx: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        input_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_embeddings is not None:
            x = input_embeddings
        else:
            assert idx is not None
            x = self.wte(idx)
        for block in self.blocks:
            x = block(x, mask=mask, is_causal=is_causal)
        return self.lm_head(self.ln_f(x))


def load_dclm_model(
    checkpoint_path: Path,
    is_ar: bool,
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> DCLMModel:
    """Load AR or MDLM safetensors into DCLMModel."""
    cfg = DCLMConfig()
    model = DCLMModel(cfg)

    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as f:
        state = {}
        for k in f.keys():
            tensor = f.get_tensor(k)
            if is_ar:
                # Map OLMo HF safetensors naming
                if k == "model.transformer.wte.weight":
                    state["wte.weight"] = tensor
                elif k == "model.transformer.ln_f.weight":
                    state["ln_f.weight"] = tensor
                elif k == "model.transformer.ff_out.weight":
                    # Slice from 50304 to 50280 for lm_head
                    state["lm_head.weight"] = tensor[: cfg.vocab_size, :]
                elif k.startswith("model.transformer.blocks."):
                    state[k.replace("model.transformer.", "")] = tensor
            else:
                # MDLM naming matches DCLMModel attributes directly
                state[k] = tensor

    model.load_state_dict(state, strict=True)
    model.to(device=device, dtype=dtype)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model
