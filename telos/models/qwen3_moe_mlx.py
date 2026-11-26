"""Qwen3-MoE routing with resident or SSD-backed MLX-format expert banks."""

import mlx.core as mx
from mlx import nn

from .qwen3_mlx import Qwen3MLX, _positive
from .window_routing import compute_route
from .ssd_experts import (
    AppleMemoryGuard,
    SSDExpertCache,
    expert_layer_sizes,
    is_expert_tensor,
    tensor_ranges,
)

_MIB = 2**20
_DEFAULT_CONTEXT = 2048


class RoutedMLP(nn.Module):
    def __init__(
        self,
        weights,
        layer,
        config,
        width,
        ranges,
        cache,
        *,
        module="mlp",
        prefix_root="model",
        switch_name="switch_mlp",
        load_router=True,
        activation="silu",
        inner_size=None,
        count=None,
        top_k=None,
    ):
        super().__init__()
        count = int(count if count is not None else _positive(config, "num_experts"))
        inner = int(inner_size if inner_size is not None else _positive(config, "moe_intermediate_size"))
        self.top_k = int(top_k if top_k is not None else _positive(config, "num_experts_per_tok"))
        if self.top_k > count:
            raise ValueError("Router selects more experts than the checkpoint contains")
        self.normalize = config.get("norm_topk_prob", False)
        self.activation = activation
        self.layer = layer
        self._cache = cache
        self.router_adapter = None
        self._window_controller = None
        prefix = f"{prefix_root}.layers.{layer}.{module}"
        self.router = weights.matrix(prefix + ".gate", count, width) if load_router else None
        self._bank = {}
        self._ranges = {}
        self._quantization = {}
        for projection, rows, cols in (("gate", inner, width), ("up", inner, width), ("down", width, inner)):
            key = f"{prefix}.{switch_name}.{projection}_proj"
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
                    self._bank[projection + suffix] = weights.take(name)
                else:
                    self._ranges[projection + suffix] = entry

    def _project(self, x, tensors, name):
        weight = tensors[name + ".weight"]
        settings = self._quantization[name]
        if settings is None:
            return x @ weight.T
        return mx.quantized_matmul(x, weight, tensors[name + ".scales"], tensors[name + ".biases"],
                                   transpose=True, **settings)

    def _eval_expert(self, inputs, tensors):
        gate = self._project(inputs, tensors, "gate")
        up = self._project(inputs, tensors, "up")
        act = nn.gelu_approx(gate) if self.activation == "gelu_pytorch_tanh" else nn.silu(gate)
        return self._project(act * up, tensors, "down")

    def _route(self, x):
        return compute_route(
            x,
            self.router,
            layer=self.layer,
            top_k=self.top_k,
            route_scale=1.0,
            normalize=self.normalize,
            adapter=self.router_adapter,
            controller=self._window_controller,
            activation="softmax",
            eps=0.0,
            output_dtype=x.dtype,
        )

    def _eval_routed(self, hidden, indices=None, scores=None):
        x = hidden.reshape(-1, hidden.shape[-1])
        if indices is None or scores is None:
            indices, scores = self._route(x)

        routes = {}
        for token, selected in enumerate(indices.tolist()):
            for slot, expert in enumerate(selected):
                tokens, slots = routes.setdefault(expert, ([], []))
                tokens.append(token)
                slots.append(slot)

        # Multi-token prefill retains synchronous demand reads. With one token,
        # the known misses can be queued while earlier experts execute.
        prepare = getattr(self._cache, "prepare", None)
        if prepare is not None and x.shape[0] == 1:
            prepare(self.layer, routes, self._ranges)
        output = mx.zeros_like(x)
        for expert, (tokens, slots) in sorted(routes.items()):
            rows, columns = mx.array(tokens), mx.array(slots)
            inputs = x[rows]
            if self._cache is None:
                tensors = {name: bank[expert] for name, bank in self._bank.items()}
            else:
                tensors = self._cache.get(self.layer, expert, self._ranges)
            values = self._eval_expert(inputs, tensors)
            values = (values * scores[rows, columns, None]).astype(output.dtype)
            output = output.at[rows].add(values)
            if self._cache is not None:
                # Release all lazy references before the next expert can evict these weights.
                mx.eval(output)
                del values, tensors
        return output.reshape(hidden.shape)

    def __call__(self, hidden):
        return self._eval_routed(hidden)


_SKIP_NON_TEXT_PREFIXES = (
    "vision_tower.",
    "visual.",
    "audio_tower.",
    "embed_vision.",
    "embed_audio.",
    "mtp.",
)


def _text_config(config):
    text = config.get("text_config")
    return {**config, **text} if isinstance(text, dict) else config


def plan_experts(config, ranges, budget, context_length, prefix_cache_entries):
    text = _text_config(config)
    family = config.get("model_type")
    total = sum(entry.size for entry in ranges.values())
    experts = text.get("num_experts")
    layers = _positive(text, "num_hidden_layers")
    cache_cap = 1024 * _MIB if family in {"qwen3_5_moe", "qwen3_5_moe_text", "gemma4", "gemma4_text", "lfm2_moe"} else 512 * _MIB
    conv_bytes = 0
    width = _positive(text, "hidden_size")
    heads = _positive(text, "num_attention_heads")
    kv_heads = _positive(text, "num_key_value_heads", heads)
    head_dim = _positive(text, "head_dim", width // heads)
    if family == "lfm2_moe":
        from .lfm2_moe_mlx import layer_types

        kinds = layer_types(text)
        conv_bytes = 4 * kinds.count("conv") * (_positive(text, "conv_L_cache", 3) - 1) * width
        layers = kinds.count("full_attention")
    elif family in {"qwen3_5_moe", "qwen3_5_moe_text"}:
        from .qwen3_5_moe_mlx import _layer_types

        kinds = _layer_types(text, layers)
        layers = kinds.count("full_attention")
        conv_bytes = 4 * kinds.count("linear_attention") * _positive(text, "linear_num_value_heads", heads) * _positive(
            text, "linear_key_head_dim", head_dim,
        ) * _positive(text, "linear_value_head_dim", head_dim) // 64
    elif family in {"gemma4", "gemma4_text"}:
        head_dim = max(head_dim, int(text.get("global_head_dim") or head_dim))

    max_context = _positive(text, "max_position_embeddings", 32768)
    context = min(_DEFAULT_CONTEXT, max_context) if context_length is None else context_length
    if type(context) is not int or not 2 <= context <= max_context:
        raise ValueError(f"Context length must be between 2 and {max_context}")

    layer_bytes = expert_layer_sizes(ranges, experts) if experts else {}
    if not layer_bytes:
        if family in {"gemma4", "gemma4_text"} and not text.get("enable_moe_block", False):
            return context, None
        raise ValueError("Flash-MoE requires expert-major MLX Safetensors (switch_mlp)")

    expert_bytes = sum(layer_bytes.values()) * experts
    # Use four-byte activations for admission and reserve space for prefill and read staging.
    per_cache = conv_bytes + 2 * 4 * layers * kv_heads * head_dim * min(max_context, (context + 255) // 256 * 256)
    workspace = 512 * _MIB + 32 * context * width + 2 * max(layer_bytes.values())
    if total + per_cache * (1 + prefix_cache_entries) + workspace <= budget:
        return context, None
    remaining = min(budget - (total - expert_bytes) - per_cache - workspace, cache_cap)
    if remaining < max(layer_bytes.values()):
        raise ValueError("Flash-MoE cannot fit resident weights, KV cache and one expert in the RAM budget. "
                         "Try a shorter context or a smaller checkpoint.")
    return context, remaining


def load_mlx_moe(checkpoint, config, *, context_length, memory_budget_mib, prefix_cache_entries, memory_guard=None):
    family = config["model_type"]
    text = _text_config(config)
    if family == "lfm2_moe":
        from .lfm2_moe_mlx import LFM2MoEMLX, LFM2RoutedMLP

        model_class = LFM2MoEMLX
        expert_class = LFM2RoutedMLP
        prefix_cache_entries = 0
    elif family in {"qwen3_5_moe", "qwen3_5_moe_text"}:
        from .qwen3_5_moe_mlx import Qwen3_5MoEMLX, Qwen3_5RoutedMLP

        model_class = Qwen3_5MoEMLX
        expert_class = Qwen3_5RoutedMLP
        prefix_cache_entries = 0
    elif family in {"gemma4", "gemma4_text"}:
        from .gemma4_mlx import Gemma4MLX, Gemma4RoutedMLP

        model_class = Gemma4MLX
        expert_class = Gemma4RoutedMLP
    else:
        model_class = Qwen3MLX
        expert_class = RoutedMLP

    raw_ranges = tensor_ranges(checkpoint)
    ranges = {k: v for k, v in raw_ranges.items() if not k.startswith(_SKIP_NON_TEXT_PREFIXES)}
    prefix_root = "language_model.model" if any(k.startswith("language_model.model.") for k in ranges) else "model"

    guard = memory_guard if memory_guard is not None else AppleMemoryGuard()
    recommended = mx.device_info()["max_recommended_working_set_size"]
    budget = min(guard.budget, recommended)
    if memory_budget_mib is not None:
        budget = min(budget, memory_budget_mib * _MIB)
    context, cache_bytes = plan_experts(config, ranges, budget, context_length, prefix_cache_entries)
    expert_ranges = {name: entry for name, entry in ranges.items() if is_expert_tensor(name)}
    cache = None
    if cache_bytes is not None:
        if family in {"lfm2_moe", "qwen3_5_moe", "qwen3_5_moe_text", "gemma4", "gemma4_text"}:
            from .ssd_expert_io import LFMSSDExpertCache

            experts = _positive(text, "num_experts")
            top_k = _positive(text, "num_experts_per_tok", text.get("top_k_experts"))
            cache = LFMSSDExpertCache(
                cache_bytes, guard, layer_sizes=expert_layer_sizes(expert_ranges, experts),
                fragment_size=max(entry.size // experts for entry in expert_ranges.values()),
                top_k=top_k,
            )
        else:
            cache = SSDExpertCache(cache_bytes, guard)
    arrays = {}
    for name, entry in ranges.items():
        if cache is None or name not in expert_ranges:
            guard.check()
            array = entry.load()
            mx.eval(array)
            arrays[name] = array

    def make_experts(weights, layer, layer_cfg, width):
        if family in {"qwen3_5_moe", "qwen3_5_moe_text", "gemma4", "gemma4_text"}:
            return expert_class(weights, layer, layer_cfg, width, expert_ranges, cache, prefix_root=prefix_root)
        return expert_class(weights, layer, layer_cfg, width, expert_ranges, cache)

    model = model_class(config, arrays, expert_factory=make_experts).eval()
    if expert_ranges:
        raise ValueError(f"Unexpected expert tensor: {next(iter(expert_ranges))}")
    model.max_seq_len = context
    model.weight_storage = "ssd_experts" if cache is not None else "resident"
    model._expert_cache = cache
    return model
