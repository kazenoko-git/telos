"""Opt-in packed projections with canonical split checkpoint keys.

Parameters are packed for training; state dictionaries retain the established
Q/K/V and gate/up names so existing native checkpoints and loaders remain usable.
"""
import torch
from torch import nn
from torch.nn import functional as F


class PackedSwiGLU(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.gate_up = nn.Linear(source.w1.in_features, source.w1.out_features * 2, bias=False)
        self.gate_up.weight = nn.Parameter(torch.cat((source.w1.weight, source.v.weight)))
        self.w2 = source.w2

    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        result = super().state_dict(*args, destination=destination, prefix=prefix, keep_vars=keep_vars)
        weight = result.pop(prefix + 'gate_up.weight')
        result[prefix + 'w1.weight'], result[prefix + 'v.weight'] = weight.chunk(2, 0)
        return result

    def _load_from_state_dict(self, state, prefix, *args, **kwargs):
        if prefix + 'w1.weight' in state:
            state[prefix + 'gate_up.weight'] = torch.cat((state.pop(prefix + 'w1.weight'), state.pop(prefix + 'v.weight')))
        super()._load_from_state_dict(state, prefix, *args, **kwargs)


def pack_attention(attention):
    """Pack projection parameters while retaining the common attention forward."""
    width = attention.q_proj.in_features
    sizes = [attention.q_proj.out_features, attention.k_proj.out_features, attention.v_proj.out_features]
    packed = nn.Linear(width, sum(sizes), bias=False)
    packed.weight = nn.Parameter(torch.cat((attention.q_proj.weight, attention.k_proj.weight, attention.v_proj.weight)))
    attention.qkv_proj = packed
    attention.qkv_sizes = sizes
    del attention.q_proj, attention.k_proj, attention.v_proj
    attention.register_state_dict_post_hook(_save_attention)
    attention.register_load_state_dict_pre_hook(_load_attention)


def _save_attention(module, state, prefix, metadata):
    weight = state.pop(prefix + 'qkv_proj.weight')
    for name, part in zip(('q_proj', 'k_proj', 'v_proj'), weight.split(module.qkv_sizes, 0)):
        state[prefix + name + '.weight'] = part


def _load_attention(module, state, prefix, *args):
    if prefix + 'q_proj.weight' in state:
        state[prefix + 'qkv_proj.weight'] = torch.cat([state.pop(prefix + name + '.weight')
                                                     for name in ('q_proj', 'k_proj', 'v_proj')])


def pack_projections(model):
    model.config.packed_projections = True
    for block in model.layers:
        if not hasattr(block.attn, 'qkv_proj'):
            pack_attention(block.attn)
            block.mlp = PackedSwiGLU(block.mlp)


def unpack_projections(model):
    """Restore individual modules before injecting LoRA adapters."""
    from .components import Attention, SwiGLU
    for block in model.layers:
        attention = block.attn
        if not hasattr(attention, 'qkv_proj'):
            continue
        state = attention.state_dict()
        replacement = Attention(attention.d_model, attention.n_heads, attention.n_kv_heads,
                                attention.dropout, attention.is_causal)
        replacement.to(device=attention.qkv_proj.weight.device, dtype=attention.qkv_proj.weight.dtype)
        replacement.load_state_dict(state)
        block.attn = replacement
        mlp = SwiGLU(attention.d_model, block.mlp.gate_up.out_features // 2)
        mlp.to(device=attention.qkv_proj.weight.device, dtype=attention.qkv_proj.weight.dtype)
        mlp.load_state_dict(block.mlp.state_dict())
        block.mlp = mlp
    model.config.packed_projections = False
