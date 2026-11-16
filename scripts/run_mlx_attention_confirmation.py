"""Fresh-process ABBA full-update confirmation of two attention shapes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path.cwd()
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=Path('reports/mlx_fusion_2026-11-16/full_attention'))
parser.add_argument('--data', type=Path, default=Path('reports/mlx_fusion_2026-11-16/corpus_prefix.bin'))
parser.add_argument('--warmup', type=int, default=16)
parser.add_argument('--swap-growth-baseline-updates', type=int, default=0)
args = parser.parse_args()
if not 0 <= args.warmup < 96 or not 0 <= args.swap_growth_baseline_updates <= args.warmup:
    parser.error('Require 0 <= swap baseline <= warmup < 96')
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=True)
(output / 'power.txt').write_text(subprocess.check_output(['pmset', '-g', 'custom'], text=True))
results_file = output / 'results.json'
results = json.loads(results_file.read_text()) if results_file.exists() else []
if results and not results[-1]['valid']:
    failed = results.pop()
    with (output / 'invalid_attempts.jsonl').open('a') as stream:
        stream.write(json.dumps(failed) + '\n')
ordinal = 0
for length, microbatch, accum, updates in ((2048, 1, 8, 128), (256, 8, 8, 96)):
    for index, mode in enumerate(('native', 'fallback', 'fallback', 'native')):
        ordinal += 1
        if ordinal <= len(results):
            continue
        name = f't{length}_{index}_{mode}'
        trial = output / name
        attempt = 0
        while trial.exists():
            attempt += 1
            trial = output / f'{name}_retry{attempt}'
        env = {**os.environ, 'PYTHONPATH': '/private/tmp/telos-mlx-0.32.4:' + str(root),
               'MLX_ENABLE_TF32': '0', 'MLX_SDPA_VJP_FALLBACK': '0' if mode == 'native' else '1'}
        for key in ('MLX_MAX_OPS_PER_BUFFER', 'MLX_MAX_MB_PER_BUFFER', 'MLX_METAL_FAST_SYNCH'):
            env.pop(key, None)
        command = [sys.executable, str(root / 'scripts/profile_mlx_training.py'), '--output', str(trial),
                   '--data', str(args.data.resolve()),
                   '--tokens', str(updates * microbatch * accum * length), '--seq-len', str(length),
                   '--microbatch', str(microbatch), '--grad-accum', str(accum),
                   '--max-mlx-gib', '4', '--critical-memory-free-percent', '25',
                   '--critical-swap-free-mib', '256', '--critical-swap-growth-mib', '512',
                   '--swap-growth-baseline-updates', str(args.swap_growth_baseline_updates),
                   '--save-initial-weights',
                   '--experimental-packed-swiglu', '--compiled-accumulation', '--force-fused-attention']
        competition = []
        with (output / (name + '.log')).open('w') as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            started = time.monotonic()
            while process.poll() is None:
                for line in subprocess.check_output(['ps', '-axo', 'pid=,comm='], text=True).splitlines():
                    pid, worker = line.strip().split(None, 1)
                    if int(pid) not in (os.getpid(), process.pid) and Path(worker).name.startswith('python') and '/uv/tools/google-colab-cli/bin/' not in worker:
                        competition.append(line.strip())
                if competition or time.monotonic() - started > 300:
                    process.terminate()
                    process.wait()
                    break
                time.sleep(1)
        metadata = json.loads((trial / 'metadata.json').read_text()) if (trial / 'metadata.json').exists() else {}
        summary = json.loads((trial / 'summary.json').read_text()) if (trial / 'summary.json').exists() else {}
        rows = [json.loads(x) for x in (trial / 'steps.jsonl').read_text().splitlines()] if (trial / 'steps.jsonl').exists() else []
        valid = (process.returncode == 0 and not competition and summary.get('status') == 'complete'
                 and len(rows) == updates and metadata.get('mlx_version') == '0.32.4'
                 and metadata.get('attention_backward_env') == env['MLX_SDPA_VJP_FALLBACK'])
        warm = rows[args.warmup:]
        result = dict(name=name, trial_dir=str(trial), mode=mode, length=length, valid=valid, competition=competition,
                      metadata=metadata, summary=summary, warmup=args.warmup,
                      tokens_per_second=len(warm) * microbatch * accum * length / sum(r['training_step_seconds'] for r in warm) if valid else None,
                      peak_mlx_bytes=max((r['mlx_peak_bytes'] for r in rows), default=0))
        results.append(result)
        (output / 'results.json').write_text(json.dumps(results, indent=2))
        print(json.dumps(result), flush=True)
        if not valid:
            raise RuntimeError('Discarded invalid full-update trial; inspect saved telemetry')
assert len({r['metadata']['initial_weights_sha256'] for r in results}) == 1, 'Initial weights differ'
for length in (2048, 256):
    plans = [r['metadata']['memory_plan'] for r in results if r['length'] == length]
    assert all(p == plans[0] for p in plans), 'Memory plans differ within a shape'
