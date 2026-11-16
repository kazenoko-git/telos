"""Run fresh-process projection probes; head-only timings are not training rates."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=Path('reports/mlx_fusion_2026-11-16/projection_matrix'))
args = parser.parse_args()
root = Path.cwd()
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=True)
results = []
for vocab in (8192, 32768, 65536):
    for variant, chunk in (('full', 256), ('vocab', 1024), ('vocab', 4096), ('token', 256), ('token', 512)):
        name = f'v{vocab}_{variant}_{chunk}'
        target = output / (name + '.json')
        with (output / (name + '.log')).open('w') as log:
            process = subprocess.run([
                sys.executable, str(root / 'scripts/probe_mlx_projection_loss.py'),
                '--variant', variant, '--chunk-size', str(chunk), '--vocab', str(vocab),
                '--output', str(target),
            ], env={**os.environ, 'PYTHONPATH': str(root)}, stdout=log, stderr=subprocess.STDOUT)
        result = dict(name=name, returncode=process.returncode)
        if target.exists():
            result.update(json.loads(target.read_text()))
        results.append(result)
        (output / 'results.json').write_text(json.dumps(results, indent=2))
        print(name, process.returncode, result.get('tokens_per_second'), result.get('peak_mlx_bytes'), flush=True)
        if process.returncode:
            raise RuntimeError(f'{name} failed its numerical or runtime gate; inspect the saved log')
