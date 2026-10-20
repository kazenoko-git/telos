"""Matched forward-only CPU FP32 evaluation of completed unified 100M endpoints."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import gc
import hashlib
import json
import math
from dataclasses import fields
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from telos.models import TelosConfig, TelosTransformer

OUT = Path(__file__).resolve().parent
DATA = ROOT / 'data/python_corpus_15b.bin'
LABELS = ['100m_r50_modal_h100_seed42_cad1', '100m_5b', '100m_5b_alpha075', '100m_schedstep1', '100m_tpu_2']
OFFSETS = [6_000_000_000, 8_000_000_000, 10_000_000_000, 12_000_000_000]
SEQUENCES_PER_SLICE = 32
SEQ_LEN = 512
MICROBATCH = 4
SEED = 42


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def auroc(scores, labels):
    scores, labels = np.asarray(scores), np.asarray(labels, dtype=np.int64)
    npos, nneg = int(labels.sum()), int((1 - labels).sum())
    if not npos or not nneg:
        return None
    order = np.argsort(scores, kind='stable')
    ordered_scores, ordered_labels = scores[order], labels[order]
    _, starts, counts = np.unique(ordered_scores, return_index=True, return_counts=True)
    pos_counts = np.add.reduceat(ordered_labels, starts)
    mean_ranks = starts + (counts + 1) / 2
    return float((np.dot(pos_counts, mean_ranks) - npos * (npos + 1) / 2) / (npos * nneg))


def atomic_report(report):
    temporary = OUT / 'results.tmp.json'
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(OUT / 'results.json')


def model_metadata(path):
    checkpoint = torch.load(path, weights_only=True, mmap=True, map_location='cpu')
    cfg = checkpoint['config']
    crsr = cfg.get('corosred', cfg.get('crsr', {}))
    if not crsr.get('unified', False) or str(crsr.get('phase', '')).lower() != 'unified':
        raise ValueError('Not a declared unified COROSred checkpoint')
    state = {k.removeprefix('module.').removeprefix('_orig_mod.'): v for k, v in checkpoint['model_state_dict'].items()}
    if cfg['model'].get('tied_embeddings', True) and not torch.equal(state['tok_embeddings.weight'], state['output_projection.weight']):
        raise ValueError('Declared tied embedding/output tensors differ; refusing silent overwrite')
    keys = {f.name for f in fields(TelosConfig)}
    config = {k: v for k, v in cfg['model'].items() if k in keys}
    config.update(dropout=0., use_grad_checkpoint=False)
    model = TelosTransformer(TelosConfig(**config)).float().eval()
    model.load_state_dict(state, strict=True)
    model.tie_weights()
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('Nonfinite model parameters')
    metadata = {
        'checkpoint': str(path), 'sha256': digest_file(path),
        'step': checkpoint.get('global_step'), 'embedded_config': cfg,
        'unique_parameters': sum(p.numel() for p in model.parameters()),
        'saved_dtypes': sorted({str(v.dtype) for v in state.values()}),
        'declared_backend': cfg.get('_backend'), 'declared_device': cfg.get('_device'),
        'saved_lrh_acc_ema': checkpoint.get('lrh_acc_ema'),
        'saved_lrh_auc_ema': checkpoint.get('lrh_auc_ema'),
    }
    del checkpoint, state
    return model, metadata


@torch.inference_mode()
def score(model, tokens, masks):
    per_sequence, head_scores, head_labels = [], [], []
    for start in range(0, len(tokens), MICROBATCH):
        x, mask = tokens[start:start + MICROBATCH], masks[start:start + MICROBATCH]
        logits, reliability = model(x, mask_override=True, return_reliability=True)
        target = x[:, 1:]
        causal_logits = logits[:, :-1, :].contiguous()
        nll_c = F.cross_entropy(causal_logits.reshape(-1, causal_logits.shape[-1]), target.reshape(-1), reduction='none').reshape_as(target)
        correct_c = causal_logits.argmax(-1).eq(target)
        valid = target >= 4
        head_scores.append(reliability[:, :-1][valid].numpy().copy())
        head_labels.append(correct_c[valid].numpy().copy())
        sums_c, hits_c = nll_c.sum(1).tolist(), correct_c.sum(1).tolist()
        del logits, reliability, causal_logits, nll_c, correct_c

        corrupted = torch.where(mask, torch.ones_like(x), x)
        logits = model(corrupted, mask_override=False, return_reliability=False)
        nll_i = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), x.reshape(-1), reduction='none').reshape_as(x)
        sums_i = (nll_i * mask).sum(1).tolist()
        hits_i = (logits.argmax(-1).eq(x) & mask).sum(1).tolist()
        counts_i = mask.sum(1).tolist()
        for j in range(len(x)):
            per_sequence.append({
                'slice': (start + j) // SEQUENCES_PER_SLICE,
                'causal_nll': sums_c[j], 'causal_correct': hits_c[j], 'causal_tokens': SEQ_LEN - 1,
                'infill_nll': sums_i[j], 'infill_correct': hits_i[j], 'infill_tokens': counts_i[j],
            })
        del logits, nll_i, corrupted
    scores, labels = np.concatenate(head_scores), np.concatenate(head_labels)
    result = {'per_sequence': per_sequence, 'lrh_correctness_auroc': auroc(scores, labels), 'lrh_tokens': int(len(labels))}
    for task in ['causal', 'infill']:
        count = sum(r[task + '_tokens'] for r in per_sequence)
        ce = sum(r[task + '_nll'] for r in per_sequence) / count
        result[task] = {'ce': ce, 'perplexity': math.exp(ce), 'top1_pct': 100 * sum(r[task + '_correct'] for r in per_sequence) / count, 'tokens': count}
    result['slices'] = []
    for index in range(len(OFFSETS)):
        subset = [r for r in per_sequence if r['slice'] == index]
        result['slices'].append({task + '_ce': sum(r[task + '_nll'] for r in subset) / sum(r[task + '_tokens'] for r in subset) for task in ['causal', 'infill']})
    return result


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    mapped = np.memmap(DATA, dtype=np.uint16, mode='r')
    arrays = []
    for offset in OFFSETS:
        end = offset + SEQUENCES_PER_SLICE * SEQ_LEN
        if end > len(mapped):
            raise ValueError('Evaluation slice outside corpus')
        arrays.append(np.array(mapped[offset:end], dtype=np.int64).reshape(SEQUENCES_PER_SLICE, SEQ_LEN))
    tokens = torch.from_numpy(np.concatenate(arrays))
    if tokens.min() < 0 or tokens.max() >= 8192:
        raise ValueError('Fixture tokens incompatible with 8192 vocabulary')
    masks = (torch.rand(tokens.shape, generator=torch.Generator().manual_seed(SEED)) < .20) & (tokens >= 4)
    masks[:, 0] = False
    if not masks.any(1).all():
        raise ValueError('Sequence has no infill targets')
    fixture = OUT / 'fixture.pt'
    torch.save({'tokens': tokens, 'infill_mask': masks, 'offsets': OFFSETS}, fixture)
    report = {
        'purpose': 'Descriptive common-fixture endpoint evaluation; not causal backend parity or a public benchmark',
        'device': 'cpu', 'dtype': 'float32', 'torch': str(torch.__version__), 'seed': SEED,
        'data': str(DATA), 'dtype_assumption': 'uint16, matching prior corpus audits; no sidecar for 15B file',
        'offsets_tokens': OFFSETS, 'sequences_per_slice': SEQUENCES_PER_SLICE,
        'sequences': len(tokens), 'sequence_length': SEQ_LEN, 'microbatch': MICROBATCH,
        'input_sha256': hashlib.sha256(tokens.numpy().tobytes()).hexdigest(),
        'mask_sha256': hashlib.sha256(masks.numpy().tobytes()).hexdigest(),
        'fixture_path': str(fixture),
        'infill': '20% uniform corruption on token IDs>=4; first position protected; bidirectional attention; score masked positions only',
        'causal': 'Explicit causal attention; shifted next-token loss across all511 targets per sequence',
        'reliability': 'Raw head scores predict causal top1 correctness; excludes special target IDs<4; AUROC uses average ranks for ties',
        'unseen_status': 'Offsets exceed 5B positions in the15B file; train/eval document disjointness and tokenizer/corpus lineage across all runs are not established',
        'excluded': {'checkpoints/corosred/100m/checkpoint_final.pt': 'global_step1; synthetic data; legacy Phase A'},
        'source_sha256': {str(p.relative_to(ROOT)): digest_file(p) for p in [Path(__file__), ROOT/'telos/models/transformer.py', ROOT/'telos/models/components.py']},
        'models': {}, 'errors': {},
    }
    atomic_report(report)
    expected_arch = None
    for label in LABELS:
        started = time.perf_counter()
        path = ROOT / 'checkpoints/corosred/unified' / label / 'checkpoint_final.pt'
        print(f'[Starting] {label}', flush=True)
        try:
            model, metadata = model_metadata(path)
            arch = {k: metadata['embedded_config']['model'].get(k) for k in ['vocab_size','d_model','n_layers','n_heads','n_kv_heads']}
            if expected_arch is not None and arch != expected_arch:
                raise ValueError('Architecture differs from new H100 model')
            expected_arch = arch
            result = score(model, tokens, masks)
            result['metadata'], result['seconds'] = metadata, time.perf_counter() - started
            report['models'][label] = result
            print(f'[Done] {label}: C-CE={result["causal"]["ce"]:.6f}, I-CE={result["infill"]["ce"]:.6f}, C-top1={result["causal"]["top1_pct"]:.2f}%, I-top1={result["infill"]["top1_pct"]:.2f}%, {result["seconds"]:.1f}s', flush=True)
            del model
            gc.collect()
        except Exception as exc:
            report['errors'][label] = {'type': type(exc).__name__, 'message': str(exc)}
            print(f'[Excluded/error] {label}: {type(exc).__name__}: {exc}', flush=True)
        atomic_report(report)
    print(f'Results: {OUT / "results.json"}', flush=True)


if __name__ == '__main__':
    main()
