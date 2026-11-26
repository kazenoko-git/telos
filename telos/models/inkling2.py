"""Checkpoint layout and demand-loaded experts for the fixed-loop Inkling 2 Mini."""

import json
import math
from contextlib import nullcontext as _null_visit
from dataclasses import dataclass

from .checkpoint import resolve_checkpoint
from .ssd_experts import TensorRange, tensor_ranges

_TEXT = "model.language_model."


def _integer(config, key, minimum=1):
    value = config.get(key)
    if type(value) is not int or value < minimum:
        raise ValueError(f"Invalid Inkling 2 {key}")
    return value


def _positive_number(config, key):
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
    router_weight: TensorRange
    router_bias: TensorRange | None
    top_k: int
    route_scale: float
    gate_softcap: float
    up_softcap: float

    @classmethod
    def from_ranges(cls, layer, ranges, *, experts, width, inner, top_k, route_scale,
                    use_gate_bias, gate_softcap, up_softcap):
        mlp = f"{_TEXT}layers.{layer}.mlp."
        prefix = mlp + "experts."
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
        weight = _take_range(ranges, mlp + "gate.weight", "F32", (experts, width))
        bias = (_take_range(ranges, mlp + "gate.e_score_correction_bias", "F32", (experts,))
                if use_gate_bias else None)
        return cls(layer, "nvfp4" if nvfp4 else "mxfp8", entries, weight, bias,
                   top_k, route_scale, gate_softcap, up_softcap)

    def load_router(self):
        return self.router_weight.load(), self.router_bias.load() if self.router_bias is not None else None

    def route(self, hidden, router, *, controller=None, adapter=None):
        import mlx.core as mx
        from .window_routing import compute_route

        weight, bias = router
        return compute_route(
            hidden,
            lambda h: h.astype(mx.float32) @ weight.T,
            layer=self.layer,
            top_k=self.top_k,
            route_scale=self.route_scale,
            normalize=True,
            base_bias=bias,
            adapter=adapter,
            controller=controller,
            activation="sigmoid",
            eps=1e-20,
            output_dtype=mx.float32,
        )

    def routed(self, hidden, router, cache, *, trace=None, controller=None, adapter=None):
        import mlx.core as mx

        x = hidden.reshape(-1, hidden.shape[-1])
        indices, weights = self.route(x, router, controller=controller, adapter=adapter)
        routes = {}
        for row, selected in enumerate(indices.tolist()):
            for slot, expert in enumerate(selected):
                rows, slots = routes.setdefault(expert, ([], []))
                rows.append(row)
                slots.append(slot)
        ordered = sorted(routes)
        if trace is not None:
            hits_before = getattr(cache, "hits", None)
            misses_before = getattr(cache, "misses", None)
            bytes_before = getattr(cache, "bytes_read", None)
            trace["experts"] = ordered
            trace["byte_sizes"] = [sum(entry.expert(expert).size for entry in self.ranges.values())
                                   for expert in ordered]
            trace["routed_experts"] = indices.tolist()
        prepare = getattr(cache, "prepare", None)
        if prepare is not None and x.shape[0] == 1:
            prepare(self.layer, routes, self.ranges)
        visit = getattr(cache, "visit", None)
        entries = getattr(cache, "_entries", None)
        all_cached = entries is not None and x.shape[0] == 1 and all((self.layer, e) in entries for e in ordered)
        output = mx.zeros(x.shape, dtype=mx.float32)
        with visit(self.layer, ordered) if visit is not None else _null_visit():
            for expert in ordered:
                rows, slots = routes[expert]
                rows, slots = mx.array(rows), mx.array(slots)
                packed = cache.get(self.layer, expert, self.ranges)
                values = self.project(x[rows], packed).astype(mx.float32)
                output = output.at[rows].add(values * weights[rows, slots, None])
                if not all_cached:
                    # End the graph before the cache can evict these expert weights.
                    mx.eval(output)
                del packed, values
        if trace is not None and bytes_before is not None:
            trace["bytes_read"] = cache.bytes_read - bytes_before
            if hits_before is not None:
                trace["hits"] = cache.hits - hits_before
                trace["misses"] = cache.misses - misses_before
        return output.astype(hidden.dtype).reshape(hidden.shape)

    def project(self, hidden, tensors):
        import mlx.core as mx

        bits, group = (4, 16) if self.mode == "nvfp4" else (8, 32)

        def linear(inputs, name):
            out = mx.quantized_matmul(
                inputs, tensors[name + ".weight"], tensors[name + ".scales"], None,
                transpose=True, group_size=group, bits=bits, mode=self.mode,
            )
            if self.mode == "nvfp4":
                # MLX nvfp4 dequantizes with a 2688 (6 * 448) global_scale denominator.
                out = (out.astype(mx.float32) * (tensors[name + ".global_scale"] / 2688.0)).astype(inputs.dtype)
            return out

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
    def from_checkpoint(cls, checkpoint, *, config=None, ranges=None):
        checkpoint = resolve_checkpoint(checkpoint)
        if config is None:
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
            "top_k": _integer(text, "num_experts_per_tok"),
            "route_scale": _positive_number(text, "route_scale"),
            "use_gate_bias": text.get("use_gate_bias", False),
            "gate_softcap": _positive_number(text, "mlp_gate_softcap"),
            "up_softcap": _positive_number(text, "mlp_up_softcap"),
        }
        if type(options["use_gate_bias"]) is not bool:
            raise ValueError("Invalid Inkling 2 use_gate_bias")
        if options["top_k"] > options["experts"]:
            raise ValueError("Inkling 2 router selects more experts than the checkpoint contains")
        if ranges is None:
            ranges = tensor_ranges(checkpoint)
        banks = tuple(InklingExpertBank.from_ranges(layer, ranges, **options) for layer in range(layers))
        return cls(visits, banks)
