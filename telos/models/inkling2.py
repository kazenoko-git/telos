"""Checkpoint layout and demand-loaded experts for the fixed-loop Inkling 2 Mini."""

import json
import math
from dataclasses import dataclass

from .checkpoint import resolve_checkpoint
from .ssd_experts import TensorRange, tensor_ranges

_TEXT = "model.language_model."


def _integer(config, key, minimum=1):
    value = config.get(key)
    if type(value) is not int or value < minimum:
        raise ValueError(f"Invalid Inkling 2 {key}")
    return value


def _softcap(config, key):
    value = config.get(key)
    if type(value) not in (int, float) or not 0 < value < math.inf:
        raise ValueError(f"Invalid Inkling 2 {key}")
    return value


def _visits(text, layers):
    loop = text.get("loop_config")
    if not isinstance(loop, dict) or loop.get("is_varloop") is not False:
        raise ValueError("Inkling 2 Mini requires a fixed loop")
    start = _integer(loop, "core_start", minimum=0)
    end = _integer(loop, "core_end")
    passes = _integer(loop, "num_passes")
    if not start < end <= layers:
        raise ValueError("Invalid Inkling 2 loop bounds")
    # Repeated visits share weight IDs and get independent state slots.
    visits = [(layer, layer) for layer in range(end)]
    for iteration in range(passes - 1):
        offset = layers + iteration * (end - start) - start
        visits.extend((layer, offset + layer) for layer in range(start, end))
    visits.extend((layer, layer) for layer in range(end, layers))
    return tuple(visits)


def _take_range(ranges, key, dtype, shape):
    entry = ranges.get(key)
    if entry is None or entry.dtype != dtype or entry.shape != shape:
        raise ValueError(f"Missing or invalid Inkling 2 tensor: {key} (expected {dtype} {shape})")
    return entry


@dataclass(frozen=True)
class InklingExpertBank:
    layer: int
    mode: str
    ranges: dict[str, TensorRange]
    gate_softcap: float
    up_softcap: float

    @classmethod
    def from_ranges(cls, layer, ranges, *, experts, width, inner, gate_softcap, up_softcap):
        prefix = f"{_TEXT}layers.{layer}.mlp.experts."
        nvfp4 = prefix + "gate_up_proj.global_scale" in ranges
        bits, group = (4, 16) if nvfp4 else (8, 32)
        if width % group or inner % group:
            raise ValueError(f"Invalid Inkling 2 expert dimensions for layer {layer}")

        entries = {}
        for projection, rows, cols in (("gate_up_proj", 2 * inner, width), ("down_proj", width, inner)):
            expected = {
                "weight": ("U32", (experts, rows, cols * bits // 32)),
                "scales": ("U8", (experts, rows, cols // group)),
            }
            if nvfp4:
                expected["global_scale"] = ("F32", (experts,))
            for field, (dtype, shape) in expected.items():
                key = f"{projection}.{field}"
                entries[key] = _take_range(ranges, prefix + key, dtype, shape)
        if {name for name in ranges if name.startswith(prefix)} != {prefix + key for key in entries}:
            raise ValueError(f"Unexpected Inkling 2 expert tensor in layer {layer}")
        return cls(layer, "nvfp4" if nvfp4 else "mxfp8", entries, gate_softcap, up_softcap)

    def project(self, hidden, tensors):
        import mlx.core as mx

        def linear(inputs, name):
            # Dequantize only the selected expert, including its NVFP4 global scale.
            weight = mx.dequantize(
                tensors[name + ".weight"], tensors[name + ".scales"], mode=self.mode,
                global_scale=tensors.get(name + ".global_scale"), dtype=inputs.dtype,
            )
            return inputs @ weight.T

        fused = linear(hidden, "gate_up_proj").astype(mx.float32)
        gate, up = fused[..., ::2], fused[..., 1::2]
        activated = (
            self.gate_softcap * mx.tanh(gate / self.gate_softcap) * mx.sigmoid(gate)
            * (self.up_softcap * mx.tanh(up / self.up_softcap))
        ).astype(hidden.dtype)
        return linear(activated, "down_proj")


@dataclass(frozen=True)
class InklingTextLayout:
    visits: tuple[tuple[int, int], ...]
    expert_banks: tuple[InklingExpertBank, ...]

    @classmethod
    def from_checkpoint(cls, checkpoint):
        checkpoint = resolve_checkpoint(checkpoint)
        config = json.loads(checkpoint.with_name("config.json").read_text(encoding="utf-8"))
        text = config.get("text_config") if isinstance(config, dict) else None
        if not isinstance(text, dict) or config.get("model_type") != "inkling2_loop":
            raise ValueError("Expected an Inkling 2 Mini checkpoint")
        layers = _integer(text, "num_hidden_layers")
        visits = _visits(text, layers)
        options = {
            "experts": _integer(text, "n_routed_experts"),
            "width": _integer(text, "hidden_size"),
            "inner": _integer(text, "intermediate_size"),
            "gate_softcap": _softcap(text, "mlp_gate_softcap"),
            "up_softcap": _softcap(text, "mlp_up_softcap"),
        }
        if _integer(text, "num_experts_per_tok") > options["experts"]:
            raise ValueError("Inkling 2 router selects more experts than the checkpoint contains")
        ranges = tensor_ranges(checkpoint)
        banks = tuple(InklingExpertBank.from_ranges(layer, ranges, **options) for layer in range(layers))
        return cls(visits, banks)
