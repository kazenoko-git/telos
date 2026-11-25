"""Qwen3-MoE routing with resident or SSD-backed MLX-format expert banks."""

import mlx.core as mx
from mlx import nn

from .qwen3_mlx import Qwen3MLX, _positive
from .ssd_experts import (
    AppleMemoryGuard,
    SSDExpertCache,
    is_expert_tensor,
    tensor_ranges,
)

_MIB = 2**20
_DEFAULT_CONTEXT = 2048


class RoutedMLP(nn.Module):
    def __init__(self, weights, layer, config, width, ranges, cache):
        super().__init__()
        count = _positive(config, "num_experts")
        inner = _positive(config, "moe_intermediate_size")
        self.top_k = _positive(config, "num_experts_per_tok")
        if self.top_k > count:
            raise ValueError("Qwen3-MoE selects more experts than the checkpoint contains")
        self.normalize = config.get("norm_topk_prob", False)
        self.layer = layer
        self._cache = cache
        prefix = f"model.layers.{layer}.mlp"
        self.router = weights.matrix(prefix + ".gate", count, width)
        self.bank = {}
        self._ranges = {}
        self._quantization = {}
        for projection, rows, cols in (("gate", inner, width), ("up", inner, width), ("down", width, inner)):
            key = f"{prefix}.switch_mlp.{projection}_proj"
            quant = None
            expected = {".weight": ((count, rows, cols), {"F16", "F32", "BF16"})}
            if key + ".scales" in ranges:
                settings = weights.quantization.get(key, weights.quantization)
                if not isinstance(settings, dict) or settings.get("mode", "affine") != "affine":
                    raise ValueError(f"Unsupported expert quantization for {key}")
                group = _positive(settings, "group_size")
                bits = _positive(settings, "bits")
                if group not in (32, 64, 128) or bits not in (2, 3, 4, 5, 6, 8) or cols % group:
                    raise ValueError(f"Unsupported expert packing for {key}")
                quant = {"group_size": group, "bits": bits}
                expected[".weight"] = ((count, rows, cols * bits // 32), {"U32"})
                for suffix in (".scales", ".biases"):
                    expected[suffix] = ((count, rows, cols // group), {"F16", "F32", "BF16"})
            self._quantization[projection] = quant
            for suffix, (shape, dtypes) in expected.items():
                name = key + suffix
                entry = ranges.pop(name, None)
                if entry is None or entry.shape != shape or entry.dtype not in dtypes:
                    raise ValueError(f"Missing or invalid expert tensor {name}")
                if cache is None:
                    self.bank[projection + suffix] = weights.take(name)
                else:
                    self._ranges[projection + suffix] = entry

    def _project(self, x, tensors, name):
        weight = tensors[name + ".weight"]
        settings = self._quantization[name]
        if settings is None:
            return x @ weight.T
        return mx.quantized_matmul(x, weight, tensors[name + ".scales"], tensors[name + ".biases"],
                                   transpose=True, **settings)

    def __call__(self, hidden):
        x = hidden.reshape(-1, hidden.shape[-1])
        logits = self.router(x)
        probs = mx.softmax(logits.astype(mx.float32), axis=-1)
        indices = mx.argsort(-probs, axis=-1)[:, :self.top_k]
        scores = mx.take_along_axis(probs, indices, axis=-1)
        if self.normalize:
            scores = scores / scores.sum(axis=-1, keepdims=True)
        scores = scores.astype(logits.dtype)

        routes = {}
        for token, selected in enumerate(indices.tolist()):
            for slot, expert in enumerate(selected):
                tokens, slots = routes.setdefault(expert, ([], []))
                tokens.append(token)
                slots.append(slot)

        output = mx.zeros_like(x)
        for expert, (tokens, slots) in sorted(routes.items()):
            rows, columns = mx.array(tokens), mx.array(slots)
            if self._cache is None:
                tensors = {name: bank[expert] for name, bank in self.bank.items()}
            else:
                tensors = self._cache.get(self.layer, expert, self._ranges)
            inputs = x[rows]
            gate = self._project(inputs, tensors, "gate")
            up = self._project(inputs, tensors, "up")
            values = self._project(nn.silu(gate) * up, tensors, "down")
            values = (values * scores[rows, columns, None]).astype(output.dtype)
            output = output.at[rows].add(values)
            if self._cache is not None:
                # Release all lazy references before the next expert can evict these weights.
                mx.eval(output)
                del gate, up, values, tensors
        return output.reshape(hidden.shape)


def plan_experts(config, ranges, budget, context_length, prefix_cache_entries):
    total = sum(entry.size for entry in ranges.values())
    experts = _positive(config, "num_experts")
    layers = _positive(config, "num_hidden_layers")
    width = _positive(config, "hidden_size")
    heads = _positive(config, "num_attention_heads")
    kv_heads = _positive(config, "num_key_value_heads", heads)
    head_dim = _positive(config, "head_dim", width // heads)
    max_context = _positive(config, "max_position_embeddings", 32768)
    context = min(_DEFAULT_CONTEXT, max_context) if context_length is None else context_length
    if type(context) is not int or not 2 <= context <= max_context:
        raise ValueError(f"Context length must be between 2 and {max_context}")

    layer_bytes = {}
    for key, entry in ranges.items():
        if is_expert_tensor(key):
            layer = int(key.split(".")[2])
            if len(entry.shape) != 3 or entry.shape[0] != experts:
                raise ValueError(f"Expected an expert-major tensor for {key}")
            layer_bytes[layer] = layer_bytes.get(layer, 0) + entry.size // experts
        elif ".mlp.switch_mlp." in key or ".mlp.experts." in key:
            raise ValueError(f"Unsupported Qwen3-MoE expert tensor {key}")
    if not layer_bytes:
        raise ValueError("Qwen3-MoE requires expert-major MLX Safetensors (mlp.switch_mlp)")

    expert_bytes = sum(layer_bytes.values()) * experts
    # Use four-byte activations for admission and reserve space for prefill and read staging.
    per_cache = 2 * 4 * layers * kv_heads * head_dim * min(max_context, (context + 255) // 256 * 256)
    workspace = 512 * _MIB + 32 * context * width + 2 * max(layer_bytes.values())
    if total + per_cache * (1 + prefix_cache_entries) + workspace <= budget:
        return context, None
    remaining = min(budget - (total - expert_bytes) - per_cache - workspace, 512 * _MIB)
    if remaining < max(layer_bytes.values()):
        raise ValueError("Flash-MoE cannot fit resident weights, KV cache and one expert in the RAM budget. "
                         "Try a shorter context or a smaller checkpoint.")
    return context, remaining


def load_qwen3_moe(checkpoint, config, *, context_length, memory_budget_mib, prefix_cache_entries):
    ranges = tensor_ranges(checkpoint)
    guard = AppleMemoryGuard()
    recommended = mx.device_info()["max_recommended_working_set_size"]
    budget = min(guard.budget, recommended)
    if memory_budget_mib is not None:
        budget = min(budget, memory_budget_mib * _MIB)
    context, cache_bytes = plan_experts(config, ranges, budget, context_length, prefix_cache_entries)
    cache = SSDExpertCache(cache_bytes, guard) if cache_bytes is not None else None
    expert_ranges = {name: entry for name, entry in ranges.items() if is_expert_tensor(name)}
    arrays = {}
    for name, entry in ranges.items():
        if cache is None or name not in expert_ranges:
            guard.check()
            array = entry.load()
            mx.eval(array)
            arrays[name] = array

    def make_experts(weights, layer, config, width):
        return RoutedMLP(weights, layer, config, width, expert_ranges, cache)

    model = Qwen3MLX(config, arrays, expert_factory=make_experts).eval()
    if expert_ranges:
        raise ValueError(f"Unexpected Qwen3-MoE expert tensor: {next(iter(expert_ranges))}")
    model.max_seq_len = context
    model.weight_storage = "ssd_experts" if cache is not None else "resident"
    model._expert_cache = cache
    return model
