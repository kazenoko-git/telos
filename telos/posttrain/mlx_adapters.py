"""LoRA with MLX affine 4-bit frozen projections and native quantized matmul."""
import math
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


def _is_packed_quantized(layer):
    return hasattr(layer, "packed") and hasattr(layer.packed, "as_linear")


def _is_linear_like(layer):
    if isinstance(layer, (nn.Linear, nn.QuantizedLinear)) or _is_packed_quantized(layer):
        return True
    return hasattr(layer, "weight") and hasattr(layer, "bias") and getattr(layer.weight, "ndim", 0) == 2


def _linear_shape(layer):
    if isinstance(layer, LoRALinear):
        layer = layer.base
    if isinstance(layer, nn.QuantizedLinear):
        return layer.weight.shape[0], layer.weight.shape[1] * 32 // layer.bits
    if _is_packed_quantized(layer):
        packed = layer.packed
        return packed.weight.shape[0], packed.weight.shape[1] * 32 // packed.bits
    return layer.weight.shape[0], layer.weight.shape[1]


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha, dropout):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        out_features, in_features = _linear_shape(base)
        self.lora_a = mx.random.uniform(-1 / math.sqrt(in_features), 1 / math.sqrt(in_features),
                                        (rank, in_features), dtype=mx.float32)
        self.lora_b = mx.zeros((out_features, rank), dtype=mx.float32)
        self.adapter_disabled = False
        self.base.freeze()

    def __call__(self, x):
        out = self.base(x)
        if self.adapter_disabled:
            return out
        delta = (self.dropout(x).astype(mx.float32) @ self.lora_a.T) @ self.lora_b.T
        return out + (delta * self.scale).astype(out.dtype)

    def merged_weight(self):
        if isinstance(self.base, nn.QuantizedLinear):
            weight = mx.dequantize(self.base.weight, self.base.scales, getattr(self.base, "biases", None),
                                   group_size=self.base.group_size, bits=self.base.bits)
        elif _is_packed_quantized(self.base):
            packed = self.base.packed
            weight = mx.dequantize(packed.weight, packed.scales, packed.biases,
                                   group_size=packed.group_size, bits=packed.bits)
        else:
            weight = self.base.weight
        return weight.astype(mx.float32) + (self.lora_b @ self.lora_a) * self.scale


class disable_lora_adapters:
    """Temporarily disable MLX LoRA and router adapters without deepcopying base weights."""
    def __init__(self, model):
        self._modules = [
            m for _, m in model.named_modules()
            if isinstance(m, LoRALinear) or hasattr(m, "adapter_disabled")
        ]

    def __enter__(self):
        for m in self._modules:
            m.adapter_disabled = True
        return self

    def __exit__(self, *exc):
        for m in self._modules:
            m.adapter_disabled = False


def apply_lora(model, *, rank=16, alpha=32.0, dropout=0.0, quantized=False, targets=None):
    if rank < 1 or alpha <= 0 or not 0 <= dropout < 1:
        raise ValueError('LoRA requires rank>0, alpha>0 and 0<=dropout<1')
    selected = [(name, layer) for name, layer in model.named_modules()
                if name.startswith('layers.') and _is_linear_like(layer)
                and (not targets or any(name.endswith(target) for target in targets))]
    if not selected:
        raise ValueError('No transformer linear layers match LoRA targets')
    unquantized = [(name, layer) for name, layer in selected
                   if not isinstance(layer, nn.QuantizedLinear) and not _is_packed_quantized(layer)]
    if quantized and any(_linear_shape(layer)[1] % 64 for _, layer in unquantized):
        raise ValueError('MLX QLoRA needs projection input dimensions divisible by 64')
    model.freeze()
    # Module.update_modules handles list-index paths and nested projection names.
    from mlx.utils import tree_unflatten
    replacements = []
    for name, layer in selected:
        if not quantized or isinstance(layer, nn.QuantizedLinear) or _is_packed_quantized(layer):
            base = layer
        elif isinstance(layer, nn.Linear):
            base = nn.QuantizedLinear.from_linear(layer, group_size=64, bits=4)
        else:
            out_features, in_features = _linear_shape(layer)
            bias = getattr(layer, "bias", None)
            linear = nn.Linear(in_features, out_features, bias=bias is not None)
            linear.weight = layer.weight
            if bias is not None:
                linear.bias = bias
            base = nn.QuantizedLinear.from_linear(linear, group_size=64, bits=4)
        replacements.append((name, LoRALinear(base, rank, alpha, dropout)))
    model.update_modules(tree_unflatten(replacements))
    return [name for name, _ in selected]


def merged_weights(model):
    result = dict(tree_flatten(model.parameters()))
    for name, layer in model.named_modules():
        if not isinstance(layer, LoRALinear):
            continue
        for key in list(result):
            if key.startswith(name + '.'):
                del result[key]
        result[name + '.weight'] = layer.merged_weight()
        bias = getattr(layer.base, "bias", None)
        if bias is not None:
            result[name + '.bias'] = bias
    return list(result.items())


def adapter_weights(model):
    suffixes = ("lora_a", "lora_b", "selector_a", "selector_b", "selector_bias", "token_a", "token_b")
    return [(name, value) for name, value in tree_flatten(model.trainable_parameters())
            if name.endswith(suffixes)]


def load_adapter(model, path, *, base_checkpoint):
    """Attach an exported MLX adapter to the original, hash-verified base."""
    import hashlib
    import json
    from pathlib import Path
    from telos.models.checkpoint import resolve_checkpoint

    path = Path(path)
    base_file = resolve_checkpoint(base_checkpoint)
    options = json.loads(path.with_suffix('.json').read_text())
    digest = hashlib.sha256()
    with open(base_file, 'rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != options['base_sha256']:
        raise ValueError('Adapter base checkpoint hash mismatch')
    apply_lora(model, rank=options['lora_rank'], alpha=options['lora_alpha'],
               dropout=options['lora_dropout'], targets=options['lora_targets'],
               quantized=options['finetune'] == 'qlora')
    weights = mx.load(str(path))
    expected = dict(adapter_weights(model))
    if weights.keys() != expected.keys() or any(weights[k].shape != expected[k].shape for k in weights):
        raise ValueError('Adapter projections or shapes do not match the model')
    model.load_weights(list(weights.items()), strict=False)
    return model
