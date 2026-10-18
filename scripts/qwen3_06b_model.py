#!/usr/bin/env python3
"""Unified modular architecture and weights loader for Qwen3-0.6B models (AR & MDLM).

Model dimensions:
- hidden_size: 1024
- intermediate_size: 3072 (SwiGLU)
- num_hidden_layers: 28
- num_attention_heads: 16 (head_dim: 128, q_dim: 2048)
- num_key_value_heads: 8 (kv_dim: 1024, GQA group size 2)
- vocab_size: 151936
- rms_norm_eps: 1e-6
- rope_theta: 1000000.0
- mask_token_id: 151669 (<|mask|>)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

AR_WEIGHTS = (
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/model.safetensors"
)
DIFF_WEIGHTS = (
    Path.home()
    / ".cache/huggingface/hub/models--dllm-hub--Qwen3-0.6B-diffusion-mdlm-v0.1/snapshots/c8d24a3f4adaeef46881b450e1bf7d1005203bd7/model.safetensors"
)


@dataclass
class Qwen3_06B_Config:
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 151936
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    mask_token_id: int = 151669  # <|mask|> token added in dLLM
    max_position_embeddings: int = 40960

    @property
    def q_dim(self) -> int:
        return self.num_attention_heads * self.head_dim  # 2048

    @property
    def kv_dim(self) -> int:
        return self.num_key_value_heads * self.head_dim  # 1024

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads  # 2


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization with affine gain."""

    def __init__(self, size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x32 = x.float()
        normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * normed).to(orig_dtype)


class RotaryEmbedding(nn.Module):
    """Rotary position embeddings matching Qwen3 / Hugging Face RoPE conventions."""

    def __init__(self, cfg: Qwen3_06B_Config):
        super().__init__()
        self.dim = cfg.head_dim
        self.theta = cfg.rope_theta
        self.max_seq = cfg.max_position_embeddings
        inv = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv, persistent=False)
        t = torch.arange(self.max_seq).float()
        freqs = torch.outer(t, inv)
        positions = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", positions.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin", positions.sin()[None, None, :, :], persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        # Standard Hugging Face rotate_half: cat((-x2, x1), dim=-1)
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, start_pos: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        t = k.size(-2)
        cos = self.cos[:, :, start_pos : start_pos + t, :].to(dtype=q.dtype, device=q.device)
        sin = self.sin[:, :, start_pos : start_pos + t, :].to(dtype=q.dtype, device=q.device)
        q = (q * cos) + (self._rotate_half(q) * sin)
        k = (k * cos) + (self._rotate_half(k) * sin)
        return q, k


class Qwen3Attention(nn.Module):
    """Grouped-Query Attention (16 Q heads, 8 KV heads) with RoPE and optional KV caching."""

    def __init__(self, cfg: Qwen3_06B_Config):
        super().__init__()
        self.cfg = cfg
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.q_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.kv_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.kv_dim, bias=False)
        self.o_proj = nn.Linear(cfg.q_dim, cfg.hidden_size, bias=False)
        self.q_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
        self.rotary = RotaryEmbedding(cfg)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        cfg = self.cfg
        b, t, _ = x.size()
        q = self.q_proj(x).view(b, t, cfg.num_attention_heads, cfg.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, cfg.num_key_value_heads, cfg.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, cfg.num_key_value_heads, cfg.head_dim).transpose(1, 2)

        # Apply QK-norm to query and key head dimensions
        q = self.q_norm(q)
        k = self.k_norm(k)

        q, k = self.rotary(q, k, start_pos=start_pos)

        # Append to past key-value cache if provided
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=-2)
            v = torch.cat([past_kv[1], v], dim=-2)
        new_kv = (k, v)

        # Repeat KV heads for Grouped-Query Attention (GQA)
        if cfg.num_key_value_groups > 1:
            k_rep = k.repeat_interleave(cfg.num_key_value_groups, dim=1)
            v_rep = v.repeat_interleave(cfg.num_key_value_groups, dim=1)
        else:
            k_rep, v_rep = k, v

        att = F.scaled_dot_product_attention(
            q, k_rep, v_rep, attn_mask=mask, dropout_p=0.0, is_causal=(is_causal and past_kv is None)
        )
        att = att.transpose(1, 2).contiguous().view(b, t, cfg.q_dim)
        return self.o_proj(att), new_kv


class Qwen3MLP(nn.Module):
    """SwiGLU MLP: down_proj(silu(gate_proj(x)) * up_proj(x))."""

    def __init__(self, cfg: Qwen3_06B_Config):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3DecoderLayer(nn.Module):
    """Pre-RMSNorm decoder layer with attention and MLP residuals."""

    def __init__(self, cfg: Qwen3_06B_Config):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Qwen3Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = Qwen3MLP(cfg)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        # Pre-attention norm and residual
        attn_out, new_kv = self.self_attn(
            self.input_layernorm(x),
            mask=mask,
            is_causal=is_causal,
            past_kv=past_kv,
            start_pos=start_pos,
        )
        x = x + attn_out
        # Pre-MLP norm and residual
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, new_kv


class Qwen3_06B_Model(nn.Module):
    """Full Qwen3-0.6B model supporting Causal, Bidirectional, and Cached Attention."""

    def __init__(self, cfg: Optional[Qwen3_06B_Config] = None):
        super().__init__()
        self.cfg = cfg or Qwen3_06B_Config()
        cfg = self.cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(Qwen3DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        start_pos: int = 0,
        use_cache: bool = False,
    ) -> Any:
        if inputs_embeds is not None:
            h = inputs_embeds
        else:
            assert input_ids is not None
            h = self.embed_tokens(input_ids)

        new_kvs = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            p_kv = past_key_values[i] if past_key_values is not None else None
            h, n_kv = layer(h, mask=mask, is_causal=is_causal, past_kv=p_kv, start_pos=start_pos)
            if use_cache:
                new_kvs.append(n_kv)

        h = self.norm(h)
        logits = self.lm_head(h)
        if use_cache:
            return logits, new_kvs
        return logits


def load_qwen3_06b_model(
    checkpoint_path: Path,
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> Qwen3_06B_Model:
    """Load Qwen3-0.6B weights from safetensors into Qwen3_06B_Model."""
    cfg = Qwen3_06B_Config()
    model = Qwen3_06B_Model(cfg)

    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as f:
        state = {}
        for k in f.keys():
            tensor = f.get_tensor(k)
            # Map standard Hugging Face keys to Qwen3_06B_Model
            if k == "model.embed_tokens.weight":
                state["embed_tokens.weight"] = tensor
            elif k == "model.norm.weight":
                state["norm.weight"] = tensor
            elif k == "lm_head.weight":
                state["lm_head.weight"] = tensor
            elif k.startswith("model.layers."):
                state[k.replace("model.", "")] = tensor

        # If lm_head is tied and missing from checkpoint keys
        if "lm_head.weight" not in state and "embed_tokens.weight" in state:
            state["lm_head.weight"] = state["embed_tokens.weight"]

    model.load_state_dict(state, strict=True)
    model.to(device=device, dtype=dtype)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model
