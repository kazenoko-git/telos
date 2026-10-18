"""Fixed-input COROSred backend probe; not a distributed trainer or exact resume.

prepare freezes one checkpoint, token batch, and infill mask. capture compares
the production objective with a fresh, identical AdamW state on one device.
Use FP32 first, then BF16. A successful single-device probe does not validate
distributed reduction, accumulation, data streaming, or long-run schedules.
"""
import argparse
from contextlib import nullcontext
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from telos.models import TelosConfig, TelosTransformer
from telos.diffusion.corosred_unified import corosred_unified_step_pytorch


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def prepare(args):
    import numpy as np
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config = checkpoint['config']
    width = config['model'].get('seq_len') or config['model'].get('max_seq_len', 512)
    if args.batch_size < 2 or not 2 <= args.seq_len <= width or args.row < 0:
        raise ValueError('Need batch >= 2, 2 <= seq_len <= model context, and row >= 0')
    data = np.memmap(args.data, mode='r', dtype=np.uint16)
    end = (args.row + args.batch_size) * width
    if end > len(data):
        raise ValueError('Selected token rows exceed the data file')
    batch = torch.from_numpy(np.array(
        data[args.row * width:end].reshape(args.batch_size, width)[:, :args.seq_len],
        dtype=np.int64,
    ))
    ratio = config.get('corosred', {}).get('causal_ratio', 0.65)
    causal_rows = max(1, min(args.batch_size - 1, round(args.batch_size * ratio)))
    generator = torch.Generator().manual_seed(42)
    mask = torch.rand((args.batch_size - causal_rows, args.seq_len), generator=generator) < 0.2
    mask[:, 0] = False
    mask[:, 1] = True
    state = {k.removeprefix('module.').removeprefix('_orig_mod.'): v
             for k, v in checkpoint['model_state_dict'].items()}
    fixture = {
        'config': config, 'state': state, 'batch': batch, 'mask': mask,
        'causal_ratio': ratio,
        # Diagnostic weights exercise all components, independent of EMA gates.
        'weights': {'alpha': 0.7, 'beta': 0.3, 'gamma': 0.1, 'mask_blend': 0.0},
        'checkpoint_sha256': digest(args.checkpoint),
        'checkpoint_step': checkpoint.get('global_step'),
        'data_path': str(args.data), 'data_row': args.row,
        'optimizer': {'lr': 1e-4, 'betas': (0.9, 0.95), 'eps': 1e-8, 'weight_decay': 0.1},
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(fixture, args.output)
    print(json.dumps({'fixture': str(args.output), 'sha256': digest(args.output),
                      'batch_shape': list(batch.shape), 'masked_tokens': int(mask.sum())}, indent=2))


def capture(args):
    fixture = torch.load(args.fixture, map_location='cpu', weights_only=True)
    if args.device == 'xla':
        import torch_xla
        import torch_xla.core.xla_model as xm
        import torch_xla.runtime as xr
        from telos.training.xla_utils import get_xla_device
        device = get_xla_device()
        if xr.is_spmd() or xr.world_size() != 1:
            raise RuntimeError('First probe requires one XLA replica with SPMD disabled')
    else:
        device = torch.device(args.device)
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(42)
    keys = {f.name for f in fields(TelosConfig)}
    config = {k: v for k, v in fixture['config']['model'].items() if k in keys}
    config.update(dropout=0.0, use_grad_checkpoint=False)
    model = TelosTransformer(TelosConfig(**config))
    model.load_state_dict(fixture['state'], strict=True)
    model.to(device=device, dtype=torch.float32)
    model.tie_weights()
    model.train()
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim == 1 or 'embed' in name else decay).append(p)
    optimizer = torch.optim.AdamW(
        [{'params': decay, 'weight_decay': 0.1}, {'params': no_decay, 'weight_decay': 0.0}],
        lr=fixture['optimizer']['lr'], betas=fixture['optimizer']['betas'],
        eps=fixture['optimizer']['eps'], foreach=False, fused=False,
    )
    batch = fixture['batch'].to(device)
    mask = fixture['mask'].to(device)

    class FixedMask:
        def pop_or_generate_mask(self, seqs, mask_token_id, **kwargs):
            if tuple(seqs.shape) != tuple(mask.shape):
                raise ValueError('Unexpected infill partition')
            return torch.where(mask, torch.full_like(seqs, mask_token_id), seqs), mask

    forward_outputs = []

    def record_forward(module, inputs, output):
        if isinstance(output, tuple):
            forward_outputs.extend(output)
        else:
            forward_outputs.append(output)

    hook = model.register_forward_hook(record_forward)
    context = (torch.autocast(device_type=device.type, dtype=torch.bfloat16)
               if args.precision == 'bf16' else nullcontext())
    special = (torch.arange(config['vocab_size'], device=device) < 4)
    with context:
        loss, metrics = corosred_unified_step_pytorch(
            model, batch, config['vocab_size'], fixture['weights'],
            special_token_lut=special, causal_ratio=fixture['causal_ratio'],
            routing_cache=FixedMask(), metric_tracker=None,
        )
    loss.backward()
    hook.remove()

    def cpu(tensor):
        return tensor.detach().float().cpu().clone()

    gradients = {name: cpu(p.grad) for name, p in model.named_parameters() if p.grad is not None}
    initial = {name: cpu(p) for name, p in model.named_parameters()}
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if device.type == 'xla':
        xm.mark_step()
    updates = {name: cpu(p) - initial[name] for name, p in model.named_parameters()}
    tensors = {**{'gradient/' + k: v for k, v in gradients.items()},
               **{'update/' + k: v for k, v in updates.items()},
               **{f'forward/{i}': cpu(v) for i, v in enumerate(forward_outputs)}}
    for name, tensor in tensors.items():
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f'Nonfinite capture: {name}')
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True,
                                         cwd=Path(__file__).resolve().parents[1]).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        commit = None
    metadata = {
        'fixture_sha256': digest(args.fixture), 'device': str(device),
        'precision': args.precision, 'torch': str(torch.__version__), 'git_commit': commit,
        'hardware': torch.cuda.get_device_name(device) if device.type == 'cuda' else str(device),
        'torch_xla': str(torch_xla.__version__) if device.type == 'xla' else None,
        'probe_source_sha256': digest(__file__),
        'objective_source_sha256': digest(Path(__file__).resolve().parents[1] / 'telos/diffusion/corosred_unified.py'),
        'model_source_sha256': digest(Path(__file__).resolve().parents[1] / 'telos/models/transformer.py'),
        'components_source_sha256': digest(Path(__file__).resolve().parents[1] / 'telos/models/components.py'),
        'fresh_optimizer': True, 'diagnostic_weights': fixture['weights'],
        'loss': float(loss.detach().cpu()), 'preclip_gradient_norm': float(norm.detach().cpu()),
        'metrics': {k: float(v.detach().cpu()) if isinstance(v, torch.Tensor) else v
                    for k, v in metrics.items()},
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'metadata': metadata, 'tensors': tensors}, args.output)
    print(json.dumps(metadata, indent=2))


def compare(args):
    left, right = [torch.load(p, map_location='cpu', weights_only=True) for p in (args.left, args.right)]
    for key in ('fixture_sha256', 'probe_source_sha256', 'objective_source_sha256', 'model_source_sha256', 'components_source_sha256'):
        if left['metadata'][key] != right['metadata'][key]:
            raise ValueError(f'Unmatched probe: {key}')
    if left['tensors'].keys() != right['tensors'].keys():
        raise ValueError('Different gradient/forward/update keys')
    differences = {}
    for key, a in left['tensors'].items():
        b = right['tensors'][key]
        if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError(f'Invalid comparable tensors: {key}')
        a, b = a.double().flatten(), b.double().flatten()
        delta = a - b
        na, nb = a.norm().item(), b.norm().item()
        differences[key] = {
            'max_absolute': delta.abs().max().item(),
            'relative_l2': delta.norm().item() / max(na, 1e-30),
            'cosine': (a @ b).item() / (na * nb) if na and nb else None,
        }
    report = {'left': left['metadata'], 'right': right['metadata'], 'differences': differences}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    worst = sorted(differences.items(), key=lambda kv: kv[1]['relative_l2'], reverse=True)[:8]
    print(json.dumps({'worst_relative_l2': worst, 'report': str(args.output)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--seq-len', type=int, default=64)
    p.add_argument('--row', type=int, default=0)
    p = commands.add_parser('capture')
    p.add_argument('--fixture', required=True)
    p.add_argument('--device', choices=['cpu', 'cuda', 'xla'], required=True)
    p.add_argument('--precision', choices=['fp32', 'bf16'], default='fp32')
    p = commands.add_parser('compare')
    p.add_argument('--left', required=True)
    p.add_argument('--right', required=True)
    for p in commands.choices.values():
        p.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    {'prepare': prepare, 'capture': capture, 'compare': compare}[args.command](args)


if __name__ == '__main__':
    main()
