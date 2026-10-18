"""CPU endpoint comparison of saved 50M COROSred checkpoints, with explicit masks.

This describes learned checkpoints under one declared objective. It does not
compare accelerator kernels, reconstruct Adam state, or identify a training cause.
"""
import argparse
from dataclasses import fields
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from telos.models import TelosConfig, TelosTransformer

DEFAULT_CHECKPOINTS = {
    'cuda_lightning': 'checkpoints/corosred/50m_lightning/checkpoint_final.pt',
    'tpu_baseline': 'checkpoints/corosred/unified/50m_python/checkpoint_final.pt',
    'tpu_schedstep1': 'checkpoints/corosred/unified/50m_tpu_schedstep1/checkpoint_final.pt',
    'tpu_seed': 'checkpoints/corosred/unified/50m_seed/checkpoint_final.pt',
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def block_name(name):
    parts = name.split('.')
    return '.'.join(parts[:2]) if parts[0] == 'layers' else parts[0]


def gradient_geometry(causal, infill, alpha, beta):
    """Compute task geometry without assuming equal gradient norms or directions."""
    if causal.keys() != infill.keys():
        raise ValueError('Causal and infill have different active parameter sets')
    accum = {}
    for name in causal:
        c, i = causal[name].double(), infill[name].double()
        values = (float((c * c).sum()), float((i * i).sum()), float((c * i).sum()))
        for group in ('all_backbone', block_name(name)):
            totals = accum.setdefault(group, [0.0, 0.0, 0.0])
            for index, value in enumerate(values):
                totals[index] += value
    a, b = alpha / (alpha + beta), beta / (alpha + beta)
    result = {}
    for name, (cc, ii, dot) in accum.items():
        nc, ni = cc ** .5, ii ** .5
        result[name] = {
            'causal_gradient_norm': nc, 'infill_gradient_norm': ni,
            'causal_infill_cosine': max(-1., min(1., dot / (nc * ni))) if nc and ni else None,
            'weighted_causal_norm': a * nc, 'weighted_infill_norm': b * ni,
            'weighted_causal_to_infill_ratio': a * nc / (b * ni) if b * ni else None,
            'combined_gradient_norm': max(0., a*a*cc + b*b*ii + 2*a*b*dot) ** .5,
            # -grad(total) has this first-order effect on the individual loss.
            'causal_loss_directional_derivative': -(a * cc + b * dot),
            'infill_loss_directional_derivative': -(a * dot + b * ii),
        }
    return result


def task_pass(model, batches, masks, *, infill):
    model.zero_grad(set_to_none=True)
    denominator = sum(int(mask.sum()) for mask in masks) if infill else sum(
        batch.shape[0] * (batch.shape[1] - 1) for batch in batches)
    total_loss, total_correct = 0.0, 0
    batch_metrics = []
    for index, (batch, mask) in enumerate(zip(batches, masks)):
        if infill:
            corrupted = torch.where(mask, torch.ones_like(batch), batch)
            logits = model(corrupted, mask_override=False, return_reliability=False)
            flat = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   batch.reshape(-1), reduction='none').view_as(batch)
            loss_sum = (flat * mask).sum()
            correct = int(((logits.detach().argmax(-1) == batch) & mask).sum())
            count = int(mask.sum())
        else:
            logits = model(batch, mask_override=True, return_reliability=False)
            shifted = logits[:, :-1, :]
            target = batch[:, 1:]
            loss_sum = F.cross_entropy(shifted.reshape(-1, shifted.shape[-1]),
                                       target.reshape(-1), reduction='sum')
            correct = int((shifted.detach().argmax(-1) == target).sum())
            count = target.numel()
        if not torch.isfinite(loss_sum):
            raise RuntimeError('Nonfinite loss')
        (loss_sum / denominator).backward()
        value = float(loss_sum.detach())
        total_loss += value
        total_correct += correct
        batch_metrics.append({'batch': index, 'tokens': count, 'ce': value / count,
                              'top1_pct': 100 * correct / count})
        del logits, loss_sum
    gradients = {name: p.grad.detach().clone() for name, p in model.named_parameters()
                 if p.grad is not None}
    if not all(torch.isfinite(g).all() for g in gradients.values()):
        raise RuntimeError('Nonfinite gradient')
    return {'ce': total_loss / denominator, 'top1_pct': 100 * total_correct / denominator,
            'tokens': denominator, 'batches': batch_metrics}, gradients


def compare_checkpoint(path, batches, masks, alpha, beta):
    checkpoint = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
    embedded = checkpoint['config']
    keys = {f.name for f in fields(TelosConfig)}
    config = {k: v for k, v in embedded['model'].items() if k in keys}
    config.update(dropout=0.0, use_grad_checkpoint=False)
    state = {k.removeprefix('module.').removeprefix('_orig_mod.'): v
             for k, v in checkpoint['model_state_dict'].items()}
    if config.get('tied_embeddings', True) and not torch.equal(
            state['tok_embeddings.weight'], state['output_projection.weight']):
        raise ValueError('Saved tied weights disagree; refusing to silently overwrite')
    if max(int(batch.max()) for batch in batches) >= config['vocab_size']:
        raise ValueError('Evaluation token ID outside checkpoint vocabulary')
    model = TelosTransformer(TelosConfig(**config)).float().eval()
    model.load_state_dict(state, strict=True)
    model.tie_weights()
    metadata = {'checkpoint': str(path), 'checkpoint_sha256': sha256(path),
                'step': checkpoint.get('global_step'), 'config': embedded,
                'parameter_count': sum(p.numel() for p in model.parameters()),
                'saved_tensor_dtypes': sorted({str(p.dtype) for p in state.values()})}
    del checkpoint, state
    causal_metrics, causal_gradients = task_pass(model, batches, masks, infill=False)
    infill_metrics, infill_gradients = task_pass(model, batches, masks, infill=True)
    geometry = gradient_geometry(causal_gradients, infill_gradients, alpha, beta)
    result = {'metadata': metadata, 'causal': causal_metrics, 'infill': infill_metrics,
              'gradient_geometry': geometry}
    del model, causal_gradients, infill_gradients
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/python_corpus_15b.bin')
    parser.add_argument('--offset-tokens', type=int, default=6_000_000_000)
    parser.add_argument('--batches', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--seq-len', type=int, default=512)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--alpha', type=float, default=.50)
    parser.add_argument('--beta', type=float, default=.35)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.batches, args.batch_size, args.threads) < 1 or args.seq_len < 2:
        raise ValueError('Positive batch/thread sizes and sequence length >= 2 required')
    if args.alpha < 0 or args.beta < 0 or args.alpha + args.beta <= 0:
        raise ValueError('Nonnegative weights with a positive sum required')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    total = args.batches * args.batch_size * args.seq_len
    if args.offset_tokens < 0 or (args.offset_tokens + total) * 2 > args.data.stat().st_size:
        raise ValueError('Selected uint16 token slice is outside the data file')
    tokens = np.memmap(args.data, mode='r', dtype=np.uint16,
                       offset=args.offset_tokens * 2, shape=(total,))
    batch = torch.from_numpy(np.array(tokens, dtype=np.int64)).reshape(
        args.batches, args.batch_size, args.seq_len)
    generator = torch.Generator().manual_seed(args.seed)
    mask = torch.rand(batch.shape, generator=generator) < .20
    mask[:, :, 0] = False
    if not mask.flatten(1).any(1).all():
        raise ValueError('An evaluation batch contains no masked tokens')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fixture_path = args.output.with_suffix('.fixture.pt')
    torch.save({'tokens': batch, 'infill_mask': mask}, fixture_path)
    report = {'purpose': 'Saved endpoint diagnostic, not runtime backend parity or training causality',
              'device': 'cpu', 'compute_dtype': 'float32', 'torch': str(torch.__version__),
              'seed': args.seed, 'threads': args.threads, 'sequence_length': args.seq_len,
              'data_path': str(args.data), 'offset_tokens': args.offset_tokens,
              'data_dtype': 'uint16 (explicit assumption matching prior audit)',
              'evaluation_sequences': args.batches * args.batch_size,
              'evaluation_tokens': total,
              'input_sha256': hashlib.sha256(batch.numpy().tobytes()).hexdigest(),
              'mask_sha256': hashlib.sha256(mask.numpy().tobytes()).hexdigest(),
              'fixture': str(fixture_path), 'alpha': args.alpha, 'beta': args.beta,
              'reliability_loss': 'excluded; compare causal and infill backbone gradients only',
              'gradient_aggregation': 'token-weighted mean over the complete fixture per task',
              'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in
                                [Path(__file__), ROOT / 'telos/models/transformer.py',
                                 ROOT / 'telos/models/components.py']},
              'models': {}}
    try:
        report['git_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except subprocess.SubprocessError:
        report['git_commit'] = None
    expected_architecture = None
    for label, relative in DEFAULT_CHECKPOINTS.items():
        started = time.perf_counter()
        print(f'[Starting] {label}', flush=True)
        result = compare_checkpoint(ROOT / relative, batch, mask, args.alpha, args.beta)
        architecture = {key: result['metadata']['config']['model'].get(key)
                        for key in ('vocab_size', 'd_model', 'n_layers', 'n_heads', 'n_kv_heads')}
        if expected_architecture is not None and architecture != expected_architecture:
            raise ValueError('Checkpoint architectures differ')
        expected_architecture = architecture
        result['seconds'] = time.perf_counter() - started
        report['models'][label] = result
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2), encoding='utf-8')
        temporary.replace(args.output)
        geometry = result['gradient_geometry']['all_backbone']
        print(f'[Done] {label}: causal CE={result["causal"]["ce"]:.6f}, '
              f'infill CE={result["infill"]["ce"]:.6f}, '
              f'gradient cosine={geometry["causal_infill_cosine"]:.6f}, '
              f'{result["seconds"]:.1f}s', flush=True)
    print(f'Report: {args.output}', flush=True)


if __name__ == '__main__':
    main()
