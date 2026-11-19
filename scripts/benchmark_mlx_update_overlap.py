"""Synchronized ABBA comparison of accumulated optimizer dispatch.

Times whole blocks with a final fence, so async dispatch cannot inflate measured
throughput by returning early. Checks saved numerical state exactly. This is a
short local experiment, not a clock/thermal-controlled performance claim.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from telos.training.core import metal_usable
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from telos.models.mlx_transformer import MLXTelosTransformer
from telos.diffusion.ar import ar_loss_fn_mlx
from telos.training.mlx_step import build_training_step


def run(args, asynchronous):
    mx.random.seed(73)
    model = MLXTelosTransformer(vocab_size=args.vocab, d_model=args.d_model,
        n_layers=args.layers, n_heads=args.d_model // 64, is_causal=True)
    model.use_master_weights = True
    opt = optim.AdamW(.001, betas=[.9, .95], bias_correction=True)
    grad = nn.value_and_grad(model, ar_loss_fn_mlx)
    def micro(x):
        (a, b), gradients = grad(model, x, args.vocab)
        return a, b, gradients
    step = build_training_step(model, opt, micro, [model.state, mx.random.state], {}, grad_clip=1.)
    data = np.random.default_rng(71).integers(4, args.vocab,
        size=(args.accum, args.batch, args.seq), dtype=np.int32)
    def batches():
        return (mx.array(data[i], dtype=mx.int32) for i in range(args.accum))
    for _ in range(args.warmup):
        step(batches(), grad_accum=args.accum, asynchronous=asynchronous)
    step.synchronize(); mx.synchronize(); mx.reset_peak_memory()
    started = time.perf_counter()
    losses = []
    for index in range(args.steps):
        opt.learning_rate = .001 * (1 - index / (args.steps + 1))
        losses.append(step(batches(), grad_accum=args.accum, asynchronous=asynchronous))
    step.synchronize(); mx.synchronize()
    elapsed = time.perf_counter() - started
    peak = mx.get_peak_memory()
    state = [model.parameters(), opt.state, opt.master_params, losses, list(mx.random.state)]
    mx.eval(state)
    digest = hashlib.sha256()
    for key, value in tree_flatten(state):
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.astype(mx.float32) if value.dtype == mx.bfloat16 else value).tobytes())
    return dict(asynchronous=asynchronous, elapsed_seconds=elapsed,
        tokens_per_second=args.steps * args.accum * args.batch * args.seq / elapsed,
        peak_bytes=peak, state_sha256=digest.hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in [('d-model', 512), ('layers', 14), ('vocab', 8192),
                          ('seq', 256), ('batch', 8), ('accum', 8), ('steps', 12), ('warmup', 5)]:
        parser.add_argument('--' + name, type=int, default=default)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not metal_usable():
        parser.error('Metal is required')
    trials = []
    for asynchronous in [False, True, True, False]:
        trial = run(args, asynchronous)
        trials.append(trial)
        print(json.dumps(trial), flush=True)
    tokens = args.steps * args.accum * args.batch * args.seq
    pooled = {str(flag): 2 * tokens / sum(t['elapsed_seconds'] for t in trials if t['asynchronous'] == flag)
              for flag in [False, True]}
    report = dict(configuration={k: v for k, v in vars(args).items() if k != 'output'},
        trials=trials, exact_saved_numerical_state=len({t['state_sha256'] for t in trials}) == 1,
        pooled_tokens_per_second=pooled, asynchronous_change_percent=100 * (pooled['True'] / pooled['False'] - 1),
        limitation='Short ABBA experiment; clocks and thermals were not controlled.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    if not report['exact_saved_numerical_state']:
        raise SystemExit('Saved numerical state differs')


if __name__ == '__main__':
    main()
