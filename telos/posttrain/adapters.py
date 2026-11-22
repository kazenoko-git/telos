"""Native LoRA and portable packed-NF4 QLoRA for PyTorch backends.

NF4 weights are stored packed and dequantized per projection. This portable
implementation does not promise bitsandbytes fused-kernel performance, double
quantization, or paged optimizers. It also runs on CPU for parity testing.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

NF4 = (-1.0, -0.6961928, -0.52507305, -0.3949175, -0.28444138,
       -0.18477343, -0.09105004, 0.0, 0.0795803, 0.1609302,
       0.2461123, 0.33791524, 0.44070983, 0.562617, 0.72295684, 1.0)


def _checkpoint_nf4(project, inputs):
    if inputs.device.type == 'xla':
        # XLA's wrapper supplies graph barriers, XLA autocast and RNG handling.
        from torch_xla.utils.checkpoint import checkpoint
        return checkpoint(project, inputs, use_reentrant=True)
    from torch.utils.checkpoint import checkpoint
    return checkpoint(project, inputs, use_reentrant=False)


class NF4Linear(nn.Module):
    """Frozen blockwise NF4 matrix; packed storage, FP32 scales/reductions."""
    def __init__(self, linear, block_size=64):
        super().__init__()
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.block_size = block_size
        self.compute_dtype = linear.weight.dtype
        # Quantize on CPU before moving the model to its training device.
        weight = linear.weight.detach().float().cpu().flatten()
        padded = F.pad(weight, (0, (-weight.numel()) % block_size)).reshape(-1, block_size)
        scales = padded.abs().amax(-1).clamp_min(torch.finfo(torch.float32).tiny)
        table = torch.tensor(NF4)
        normalized = padded / scales[:, None]
        # Search midpoint boundaries without allocating [weights, 16].
        indices = torch.bucketize(normalized.contiguous(), (table[:-1] + table[1:]) / 2).flatten().to(torch.uint8)
        packed = indices[::2] | (indices[1::2] << 4)
        self.register_buffer('packed_weight', packed)
        self.register_buffer('scales', scales)
        self.register_buffer('codebook', table)
        self.register_buffer('bias', linear.bias.detach().clone() if linear.bias is not None else None)

    def dequantize(self):
        # Use int32 indices to halve transient dequantization index memory.
        indices = torch.stack((self.packed_weight & 15, self.packed_weight >> 4), -1).flatten().to(torch.int32)
        values = self.codebook[indices].reshape(-1, self.block_size) * self.scales[:, None]
        return values.flatten()[:self.in_features * self.out_features].reshape(self.out_features, self.in_features)

    def _project(self, inputs):
        return F.linear(inputs, self.dequantize().to(inputs.dtype),
                        self.bias.to(inputs.dtype) if self.bias is not None else None)

    def forward(self, x):
        if torch.is_grad_enabled() and x.requires_grad:
            return _checkpoint_nf4(self._project, x)
        return self._project(x)


class LoRALinear(nn.Module):
    def __init__(self, base, rank=16, alpha=32.0, dropout=0.0):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32))
        self.adapter_disabled = False
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.base.requires_grad_(False)

    def forward(self, x):
        result = self.base(x)
        if self.adapter_disabled:
            return result
        delta = F.linear(F.linear(self.dropout(x).to(self.lora_a.dtype), self.lora_a), self.lora_b)
        return result + (delta * self.scale).to(result.dtype)


class disable_lora_adapters:
    """Temporarily disable LoRA deltas to evaluate the frozen base model without deepcopy."""
    def __init__(self, model):
        self._modules = [m for m in model.modules() if isinstance(m, LoRALinear)]

    def __enter__(self):
        for m in self._modules:
            m.adapter_disabled = True
        return self

    def __exit__(self, *exc):
        for m in self._modules:
            m.adapter_disabled = False

    def merged_weight(self):
        weight = self.base.dequantize() if isinstance(self.base, NF4Linear) else self.base.weight.float()
        return weight + (self.lora_b.float() @ self.lora_a.float()) * self.scale


def apply_lora(model, *, rank=16, alpha=32.0, dropout=0.0, quantized=False, targets=None):
    """Adapt transformer projections only; embeddings/output tying stays intact."""
    if rank < 1 or alpha <= 0 or not 0 <= dropout < 1:
        raise ValueError('LoRA requires rank>0, alpha>0 and 0<=dropout<1')
    from telos.models.packed import unpack_projections
    if any(hasattr(block.attn, 'qkv_proj') for block in getattr(model, 'layers', ())):
        unpack_projections(model)
    selected = [(name, layer) for name, layer in model.named_modules()
                if name.startswith('layers.') and isinstance(layer, nn.Linear)
                and (not targets or any(name.endswith(target) for target in targets))]
    if not selected:
        raise ValueError('No transformer linear layers match LoRA targets')
    model.requires_grad_(False)
    for name, layer in selected:
        parent_name, leaf = name.rsplit('.', 1)
        base = NF4Linear(layer) if quantized else layer
        setattr(model.get_submodule(parent_name), leaf, LoRALinear(base, rank, alpha, dropout).to(layer.weight.device))
    return [name for name, _ in selected]


def merged_state_dict(model):
    model = getattr(model, 'module', model)
    result = dict(model.state_dict())
    for name, layer in model.named_modules():
        if not isinstance(layer, LoRALinear):
            continue
        for key in list(result):
            if key.startswith(name + '.'):
                del result[key]
        # Merged exports use FP32; native loaders can use the usual architecture.
        result[name + '.weight'] = layer.merged_weight().detach()
        if layer.base.bias is not None:
            result[name + '.bias'] = layer.base.bias.detach()
    return result


def adapter_state_dict(model):
    return {name: value.detach() for name, value in getattr(model, 'module', model).named_parameters()
            if name.endswith(('lora_a', 'lora_b'))}


def load_adapter(model, path, *, base_checkpoint):
    """Attach an exported adapter to its exact starting checkpoint model.

    Load `model` from `base_checkpoint` first. Validating the source bytes prevents
    silently attaching an adapter to a different base revision.
    """
    import hashlib
    artifact = torch.load(path, map_location='cpu', weights_only=True)
    if artifact.get('format') != 'telos-lora-v1':
        raise ValueError('Unsupported adapter format')
    options = artifact['options']
    digest = hashlib.sha256()
    with open(base_checkpoint, 'rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != options['base_sha256']:
        raise ValueError('Adapter base checkpoint hash mismatch')
    apply_lora(model, rank=options['lora_rank'], alpha=options['lora_alpha'],
               dropout=options['lora_dropout'], targets=options['lora_targets'],
               quantized=options['finetune'] == 'qlora')
    expected = set(adapter_state_dict(model))
    if set(artifact['adapter_state_dict']) != expected:
        raise ValueError('Adapter projection names do not match the model')
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in artifact['adapter_state_dict'].items():
            if value.shape != parameters[name].shape:
                raise ValueError(f'Adapter shape mismatch: {name}')
            parameters[name].copy_(value)
    return model
