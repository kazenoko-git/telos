"""
BitNet b1.58 (W1.58A8) Quantization-Aware Training (QAT) primitives for Télos.

Implements:
- Absmean ternary weight quantization {-1, 0, +1} with Straight-Through Estimator (STE)
- Per-token absmax INT8 activation quantization with STE
- BitLinear drop-in replacement for nn.Linear with optional SubLN pre-normalization
- 2-bit packed ternary export/unpack helpers for Apple Silicon Metal / MLX inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import RMSNorm


def bitlinear_weight_quant(w: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Absmean ternary quantization {-1, 0, +1} with STE.

    Works for both 2D [out, in] and 3D batched MoE [E, out, in] weights.
    For 3D expert tensors, computes an independent absmean scale per expert.
    """
    if w.ndim == 3:
        scale = w.detach().abs().mean(dim=(1, 2), keepdim=True).clamp(min=eps)
    else:
        scale = w.detach().abs().mean().clamp(min=eps)
    w_scaled = w / scale
    w_q = torch.clamp(torch.round(w_scaled), -1.0, 1.0) * scale
    return w + (w_q - w).detach()


def bitlinear_activation_quant(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Per-token absmax INT8 activation quantization [-128, 127] with STE."""
    scale = 127.0 / x.detach().abs().amax(dim=-1, keepdim=True).clamp(min=eps)
    x_q = torch.clamp(torch.round(x * scale), -128.0, 127.0) / scale
    return x + (x_q - x).detach()


class BitLinear(nn.Linear):
    """BitNet b1.58 linear layer (W1.58A8 QAT) with optional SubLN.

    During training, master weights remain in FP32/BF16 and receive gradients
    via Straight-Through Estimator (STE). Optional SubLN applies RMSNorm
    immediately before activation quantization as in BitNet b1.58 2B4T.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        *,
        use_subln: bool = False,
        quantize_weights: bool = True,
        quantize_activations: bool = True,
        eps: float = 1e-5,
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.use_subln = use_subln
        self.quantize_weights = quantize_weights
        self.quantize_activations = quantize_activations
        self.eps = eps
        self.subln = RMSNorm(in_features) if use_subln else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.subln is not None:
            x = self.subln(x)
        x_in = bitlinear_activation_quant(x, eps=self.eps) if self.quantize_activations else x
        w_in = bitlinear_weight_quant(self.weight, eps=self.eps) if self.quantize_weights else self.weight
        return F.linear(x_in, w_in, self.bias)


def pack_ternary_2bit(w: torch.Tensor, eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack a weight tensor into GPU-friendly 2-bit codes (4 trits per uint8 byte).

    Encoding per 2-bit lane:
        0b00 (0) ->  0
        0b01 (1) -> +1
        0b10 (2) -> -1

    Returns:
        packed: uint8 tensor of shape (..., ceil(last_dim / 4))
        scale:  float32 absmean scale per matrix (or per expert for 3D weights)
    """
    with torch.no_grad():
        if w.ndim == 3:
            scale = w.abs().mean(dim=(1, 2), keepdim=True).clamp(min=eps).float()
        else:
            scale = w.abs().mean().clamp(min=eps).float()
        trits = torch.clamp(torch.round(w.float() / scale), -1.0, 1.0).to(torch.int8)
        codes = torch.where(trits == 1, torch.ones_like(trits, dtype=torch.uint8),
                            torch.where(trits == -1, torch.full_like(trits, 2, dtype=torch.uint8),
                                        torch.zeros_like(trits, dtype=torch.uint8)))
        last_dim = codes.shape[-1]
        pad = (-last_dim) % 4
        if pad:
            codes = F.pad(codes, (0, pad), value=0)
        grouped = codes.reshape(*codes.shape[:-1], -1, 4)
        packed = (
            grouped[..., 0]
            | (grouped[..., 1] << 2)
            | (grouped[..., 2] << 4)
            | (grouped[..., 3] << 6)
        ).contiguous()
        return packed, scale


def unpack_ternary_2bit(packed: torch.Tensor, scale: torch.Tensor, orig_last_dim: int) -> torch.Tensor:
    """Unpack 2-bit ternary codes back to scaled float32 weights."""
    c0 = packed & 0x03
    c1 = (packed >> 2) & 0x03
    c2 = (packed >> 4) & 0x03
    c3 = (packed >> 6) & 0x03
    codes = torch.stack((c0, c1, c2, c3), dim=-1).reshape(*packed.shape[:-1], -1)[..., :orig_last_dim]
    trits = torch.where(codes == 1, 1.0, torch.where(codes == 2, -1.0, 0.0)).to(torch.float32)
    return trits * scale.to(torch.float32)
