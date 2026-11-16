"""Bounded paired probe; rejects timings when another Python worker is seen."""
import argparse
import inspect
import json
import os
from pathlib import Path
import statistics
import subprocess
import textwrap
import time

import mlx.core as mx
import mlx.nn as nn
from residual_norm import residual_norm


def check_workers():
    workers = []
    for line in subprocess.check_output(['ps', '-axo', 'pid=,comm='], text=True).splitlines():
        pid, command = line.strip().split(None, 1)
        if int(pid) != os.getpid() and Path(command).name.startswith('python'):
            workers.append(dict(pid=int(pid), command=command))
    if workers:
        raise RuntimeError('Another Python worker is active; discard timings: ' + json.dumps(workers))


def reference(x, branch, weight):
    residual = x + branch
    return residual, mx.fast.rms_norm(residual, weight, 1e-5)


def paired(functions, arguments, pairs, repeats):
    for function in functions:
        for _ in range(4):
            mx.eval(function(*arguments))
    results = []
    for i in range(pairs):
        check_workers()
        sample = {}
        for variant in ((0, 1) if i % 2 == 0 else (1, 0)):
            mx.synchronize()
            start = time.perf_counter()
            for _ in range(repeats):
                mx.eval(functions[variant](*arguments))
            mx.synchronize()
            sample[str(variant)] = (time.perf_counter() - start) / repeats
        check_workers()
        results.append(sample)
    return dict(samples=results, baseline_median_seconds=statistics.median(r['0'] for r in results),
                candidate_median_seconds=statistics.median(r['1'] for r in results),
                median_paired_speedup_percent=statistics.median(100 * (r['0'] / r['1'] - 1) for r in results))


def patched_block_call():
    from telos.models import mlx_components
    original = mlx_components.MLXBlock.__call__
    source = textwrap.dedent(inspect.getsource(original))
    before = '    x = x + self.out(out)\n    x = x + self.mlp(self.norm2(x))'
    after = '    x, normalized = residual_norm(x, self.out(out), self.norm2.weight, self.norm2.eps)\n    x = x + self.mlp(normalized)'
    assert source.count(before) == 1
    scope = dict(vars(mlx_components), residual_norm=residual_norm)
    exec(source.replace(before, after), scope)
    return original, scope['__call__']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pairs', type=int, default=16)
    args = parser.parse_args()
    check_workers()
    mx.set_memory_limit(512 * 1024**2)
    mx.set_cache_limit(32 * 1024**2)
    mx.random.seed(71)
    inputs = (mx.random.normal((8, 256, 512)).astype(mx.bfloat16),
              mx.random.normal((8, 256, 512)).astype(mx.bfloat16),
              mx.random.uniform(low=.5, high=1.5, shape=(512,)).astype(mx.bfloat16))
    mx.eval(inputs)
    report = {'forward': paired([mx.compile(reference), mx.compile(residual_norm)], inputs, args.pairs, 8)}
    cotangents = tuple(mx.random.normal(inputs[0].shape).astype(mx.bfloat16) for _ in range(2))
    mx.eval(cotangents)
    def transformed(function):
        return mx.compile(lambda *xs: mx.vjp(function, xs, cotangents))
    report['forward_backward'] = paired([transformed(reference), transformed(residual_norm)], inputs, args.pairs, 8)

    from telos.models.mlx_transformer import MLXTelosTransformer
    from telos.models.mlx_components import MLXBlock
    from telos.diffusion.ar import ar_loss_fn_mlx
    from mlx.utils import tree_flatten
    model = MLXTelosTransformer(vocab_size=8192, d_model=512, n_layers=1, n_heads=8, is_causal=True)
    ids = mx.random.randint(4, 8192, (2, 128))
    mx.eval(model.parameters(), ids)
    def loss(module, tokens):
        return ar_loss_fn_mlx(module, tokens, 8192)[0]
    def compiled():
        grad = nn.value_and_grad(model, loss)
        return mx.compile(lambda tokens: grad(model, tokens), inputs=[model.state], outputs=[model.state])
    baseline = compiled()
    expected = baseline(ids)
    mx.eval(expected)
    original, candidate_call = patched_block_call()
    try:
        MLXBlock.__call__ = candidate_call
        candidate = compiled()
        actual = candidate(ids)
        mx.eval(actual)
        assert mx.allclose(expected[0], actual[0], rtol=1e-5, atol=1e-6).item()
        differences = []
        for (ka, a), (kb, b) in zip(tree_flatten(expected[1]), tree_flatten(actual[1])):
            assert ka == kb
            a, b = a.astype(mx.float32), b.astype(mx.float32)
            assert mx.allclose(a, b, rtol=.009, atol=2e-5).item(), ka
            differences.append(dict(parameter=ka, max_abs=mx.max(mx.abs(a-b)).item()))
        report['block_gradient_differences'] = differences
        report['block'] = paired([baseline, candidate],
                                 (ids,), args.pairs, 3)
    finally:
        MLXBlock.__call__ = original
    report['peak_mlx_bytes'] = mx.get_peak_memory()
    report['limitations'] = ['Synchronous synthetic microbenchmarks, not full-model updates or sustained throughput.',
                            'No other Python worker observed at pair boundaries; other desktop GPU work is not excluded.']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('block_gradient_differences',) and not isinstance(v,dict)}, indent=2))
    for key in ('forward','forward_backward','block'):
        print(key, report[key]['median_paired_speedup_percent'])


if __name__ == '__main__':
    main()
