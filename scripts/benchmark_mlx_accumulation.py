"""Short, real-data 50M accumulation comparison; run each trial in a fresh process."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import subprocess

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from telos.configs.solver import solve_transformer_geometry
from telos.models.mlx_transformer import MLXTelosTransformer
from telos.training.trainer_mlx import UnifiedMLXTrainer
from scripts.profile_mlx_training import system_sample
from scripts.mlx_accumulation_experiments import ExperimentalAccumulationStep


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pipeline', action='store_true')
    parser.add_argument('--compiled-add', action='store_true')
    parser.add_argument('--defer-last', action='store_true')
    parser.add_argument('--grad-accum', type=int, default=8)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=4)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    mx.set_memory_limit(4 * 1024**3)
    mx.set_cache_limit(128 * 1024**2)
    mx.random.seed(42)
    geo = solve_transformer_geometry('50M')
    geo.update(seq_len=256, is_causal=True)
    model = MLXTelosTransformer(**geo)
    cfg = dict(model=geo, training=dict(precision='bfloat16', seed=42))
    trainer = UnifiedMLXTrainer('ar', model, cfg)
    fn, state = trainer._setup_training_functions()
    state.append(mx.random.state)
    fn = mx.compile(fn, inputs=state, outputs=state)
    opt = optim.AdamW(4e-4, weight_decay=.01, betas=[.9, .95], bias_correction=True)
    # Freeze the pre-deferral reference so historical comparisons remain reproducible.
    step = ExperimentalAccumulationStep(model, opt, fn, pipeline_accumulation=args.pipeline,
                                       compiled_accumulation=args.compiled_add,
                                       defer_last_microbatch=args.defer_last)
    corpus = np.memmap('data/english_corpus_5b.bin', dtype=np.uint16, mode='r')
    cursor = 0

    def batches():
        nonlocal cursor
        for _ in range(args.grad_accum):
            batch = mx.array(np.asarray(corpus[cursor:cursor + 8 * 256], dtype=np.int32).reshape(8, 256))
            cursor += 8 * 256
            yield batch

    initial = system_sample()
    times, losses = [], []
    for i in range(args.warmup + args.steps):
        begin = time.perf_counter()
        loss, _ = step(batches(), grad_accum=args.grad_accum)
        duration = time.perf_counter() - begin
        if i >= args.warmup:
            times.append(duration)
            losses.append(loss.item() / args.grad_accum)
        sample = system_sample()
        if (mx.get_peak_memory() > 4 * 1024**3 or sample['memory_free_percent'] <= 12
                or sample['swap_free_mb'] <= 512):
            raise RuntimeError('Memory guard stopped short benchmark')
    result = dict(pipeline=args.pipeline, compiled_add=args.compiled_add,
                  defer_last=args.defer_last,
                  grad_accum=args.grad_accum, microbatch=8,
                  sequence_length=256, warmup=args.warmup, steps=args.steps,
                  tokens_per_second=len(times) * 8 * args.grad_accum * 256 / sum(times),
                  step_seconds=times, losses=losses, peak_memory_bytes=mx.get_peak_memory(),
                  initial_system=initial, final_system=system_sample(), mlx_version=mx.__version__)
    result['source_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    digest = hashlib.sha256()
    for name, value in tree_flatten([model.parameters(), opt.state, opt.master_params]):
        digest.update(name.encode())
        if isinstance(value, mx.array):
            if value.dtype == mx.bfloat16:
                value = value.astype(mx.float32)
            digest.update(np.asarray(value).tobytes())
        else:
            digest.update(repr(value).encode())
    result['final_state_sha256'] = digest.hexdigest()
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in {'losses', 'step_seconds'}}), flush=True)


if __name__ == '__main__':
    main()
