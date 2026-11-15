"""LoRA with MLX affine 4-bit frozen projections and native quantized matmul."""
import math
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha, dropout):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        if isinstance(base, nn.QuantizedLinear):
            in_features = base.weight.shape[1] * 32 // base.bits
        else:
            in_features = base.weight.shape[1]
        self.lora_a = mx.random.uniform(-1 / math.sqrt(in_features), 1 / math.sqrt(in_features),
                                        (rank, in_features), dtype=mx.float32)
        self.lora_b = mx.zeros((base.weight.shape[0], rank), dtype=mx.float32)
        self.base.freeze()

    def __call__(self, x):
        out = self.base(x)
        delta = (self.dropout(x).astype(mx.float32) @ self.lora_a.T) @ self.lora_b.T
        return out + (delta * self.scale).astype(out.dtype)

    def merged_weight(self):
        if isinstance(self.base, nn.QuantizedLinear):
            weight = mx.dequantize(self.base.weight, self.base.scales, self.base.biases,
                                   group_size=self.base.group_size, bits=self.base.bits)
        else:
            weight = self.base.weight
        return weight.astype(mx.float32) + (self.lora_b @ self.lora_a) * self.scale


def apply_lora(model, *, rank=16, alpha=32.0, dropout=0.0, quantized=False, targets=None):
    if rank < 1 or alpha <= 0 or not 0 <= dropout < 1:
        raise ValueError('LoRA requires rank>0, alpha>0 and 0<=dropout<1')
    selected = [(name, layer) for name, layer in model.named_modules()
                if name.startswith('layers.') and isinstance(layer, nn.Linear)
                and (not targets or any(name.endswith(target) for target in targets))]
    if not selected:
        raise ValueError('No transformer linear layers match LoRA targets')
    if quantized and any(layer.weight.shape[1] % 64 for _, layer in selected):
        raise ValueError('MLX QLoRA needs projection input dimensions divisible by 64')
    model.freeze()
    # Module.update_modules handles list-index paths and nested projection names.
    from mlx.utils import tree_unflatten
    replacements = []
    for name, layer in selected:
        base = nn.QuantizedLinear.from_linear(layer, group_size=64, bits=4) if quantized else layer
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
        if 'bias' in layer.base:
            result[name + '.bias'] = layer.base.bias
    return list(result.items())


def adapter_weights(model):
    return [(name, value) for name, value in tree_flatten(model.trainable_parameters())
            if name.endswith(('lora_a', 'lora_b'))]


def load_adapter(model, path, *, base_checkpoint):
    """Attach an exported MLX adapter to the original, hash-verified base."""
    import hashlib
    import json
    from pathlib import Path
    path = Path(path)
    options = json.loads(path.with_suffix('.json').read_text())
    digest = hashlib.sha256()
    with open(base_checkpoint, 'rb') as source:
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
