#!/usr/bin/env python3
"""Unified modular architecture and weights loader for Qwen2.5-Coder-0.5B (AR and MDLM).

Dimensions:
- hidden_size: 896
- intermediate_size: 4864 (SwiGLU)
- num_hidden_layers: 24
- num_attention_heads: 14 (head_dim: 64, q_dim: 896)
- num_key_value_heads: 2 (kv_dim: 128, GQA group size 7)
- vocab_size: 151936
- rms_norm_eps: 1e-6
- rope_theta: 1000000.0
- mask_token_id: 151669 (<|mask|>)
- Attention biases: q_proj, k_proj, v_proj have biases; o_proj does not.
- Tie word embeddings: True (lm_head.weight == embed_tokens.weight).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open


def get_ar_weights_path() -> Path:
    p = list(Path.home().glob(".cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-0.5B-Instruct/snapshots/*/model.safetensors"))
    if not p:
        raise FileNotFoundError("Qwen2.5-Coder-0.5B-Instruct weights not found in Hugging Face cache.")
    return p[0]


def get_diff_weights_path() -> Path:
    p = list(Path.home().glob(".cache/huggingface/hub/models--dllm-hub--Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1/snapshots/*/model.safetensors"))
    if not p:
        raise FileNotFoundError("Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1 weights not found in Hugging Face cache.")
    return p[0]


@dataclass
class Qwen2_5_Coder_05B_Config:
    hidden_size: int = 896
    intermediate_size: int = 4864
    num_hidden_layers: int = 24
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    head_dim: int = 64
    vocab_size: int = 151936
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    mask_token_id: int = 151669  # <|mask|> token
    max_position_embeddings: int = 32768

    @property
    def q_dim(self) -> int:
        return self.num_attention_heads * self.head_dim  # 896

    @property
    def kv_dim(self) -> int:
        return self.num_key_value_heads * self.head_dim  # 128

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads  # 7


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (x * self.weight.to(torch.float32)).to(input_dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_pos: int = 32768, base: float = 1000000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def _update_cache(self, x: torch.Tensor, seq_len: int) -> None:
        if seq_len > self._seq_len_cached or self._cos_cached is None or self._cos_cached.device != x.device:
            self._seq_len_cached = max(seq_len, 2048)
            t = torch.arange(self._seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(x.device))
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos().to(x.dtype)
            self._sin_cached = emb.sin().to(x.dtype)

    def forward(self, x: torch.Tensor, seq_len: int, start_pos: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        self._update_cache(x, start_pos + seq_len)
        cos = self._cos_cached[start_pos : start_pos + seq_len]
        sin = self._sin_cached[start_pos : start_pos + seq_len]
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # q, k: [batch, heads, seq, head_dim]
    # cos, sin: [seq, head_dim] -> unsqueeze to [1, 1, seq, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen2_5_Attention(nn.Module):
    def __init__(self, cfg: Qwen2_5_Coder_05B_Config):
        super().__init__()
        self.cfg = cfg
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.q_dim, bias=True)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.kv_dim, bias=True)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.kv_dim, bias=True)
        self.o_proj = nn.Linear(cfg.q_dim, cfg.hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        is_causal: bool = True,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.cfg.num_attention_heads, self.cfg.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.cfg.num_key_value_heads, self.cfg.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.cfg.num_key_value_heads, self.cfg.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        new_kv = (k, v) if use_cache else None

        # Repeat KV for GQA
        k = k.repeat_interleave(self.cfg.num_key_value_groups, dim=1)
        v = v.repeat_interleave(self.cfg.num_key_value_groups, dim=1)

        # Scaled dot-product attention
        # is_causal applies only when seq_len > 1 and no past KV is passed
        causal_flag = is_causal if (seq_len > 1 and past_key_value is None) else False
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=causal_flag)

        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, self.cfg.q_dim)
        return self.o_proj(attn_out), new_kv


class Qwen2_5_MLP(nn.Module):
    def __init__(self, cfg: Qwen2_5_Coder_05B_Config):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen2_5_DecoderLayer(nn.Module):
    def __init__(self, cfg: Qwen2_5_Coder_05B_Config):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.self_attn = Qwen2_5_Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.mlp = Qwen2_5_MLP(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        is_causal: bool = True,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = x
        normed = self.input_layernorm(x)
        attn_out, new_kv = self.self_attn(
            normed, cos, sin, is_causal=is_causal, past_key_value=past_key_value, use_cache=use_cache
        )
        x = residual + attn_out

        residual = x
        x = residual + self.mlp(self.post_attention_layernorm(x))
        return x, new_kv


class Qwen2_5_Coder_05B_Model(nn.Module):
    def __init__(self, cfg: Qwen2_5_Coder_05B_Config):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([Qwen2_5_DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.rotary = RotaryEmbedding(cfg.head_dim, max_pos=cfg.max_position_embeddings, base=cfg.rope_theta)

    def forward(
        self,
        input_ids: torch.Tensor,
        is_causal: bool = True,
        past_key_values: Optional[list] = None,
        start_pos: int = 0,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[list]]:
        bsz, seq_len = input_ids.shape
        x = self.embed_tokens(input_ids)
        cos, sin = self.rotary(x, seq_len=seq_len, start_pos=start_pos)

        new_kvs = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            layer_past = past_key_values[i] if past_key_values is not None else None
            x, new_kv = layer(
                x, cos, sin, is_causal=is_causal, past_key_value=layer_past, use_cache=use_cache
            )
            if use_cache:
                new_kvs.append(new_kv)

        x = self.norm(x)
        # Tied weights: project using embed_tokens weight
        logits = F.linear(x, self.embed_tokens.weight)
        return logits, new_kvs


def load_qwen25_coder_model(
    weights_path: Path | str,
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> Qwen2_5_Coder_05B_Model:
    cfg = Qwen2_5_Coder_05B_Config()
    model = Qwen2_5_Coder_05B_Model(cfg)

    with safe_open(str(weights_path), framework="pt", device=device) as f:
        state_dict = {}
        for k in f.keys():
            t = f.get_tensor(k)
            # Map checkpoint keys to model keys
            # Model layers start with model.layers -> strip "model."
            if k.startswith("model."):
                mapped_k = k[6:]
            else:
                mapped_k = k
            state_dict[mapped_k] = t.to(dtype)

    model.load_state_dict(state_dict, strict=False)
    model.to(device=device, dtype=dtype)
    model.eval()
    return model
