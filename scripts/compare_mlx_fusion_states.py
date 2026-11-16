"""Compare recorded final tensor groups one NPZ member at a time on the CPU."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def leaves(node, path=''):
    if 'array' in node:
        yield path, node['array']
    elif 'dict' in node:
        for key, value in node['dict']:
            yield from leaves(value, path + '/' + str(key))
    elif 'list' in node:
        for index, value in enumerate(node['list']):
            yield from leaves(value, path + '/' + str(index))


def numeric(value):
    # MLX serializes BF16 as raw two-byte NPY voids; preserve bit comparisons
    # before converting those bits to FP32 for diagnostic distance statistics.
    if value.dtype.kind == 'V' and value.dtype.itemsize == 2:
        return (value.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    return value


def compare(first, second):
    results = {}
    with np.load(first, allow_pickle=False) as a, np.load(second, allow_pickle=False) as b:
        left = dict(leaves(json.loads(a['structure'].tobytes())))
        right = dict(leaves(json.loads(b['structure'].tobytes())))
        if left.keys() != right.keys():
            raise ValueError('Tensor key sets differ')
        for group in ('model', 'master_params', 'optimizer', 'mlx_rng'):
            count = different = elements = 0
            squared = reference_squared = maximum = 0.
            for key in left:
                if not key.startswith('/' + group + '/'):
                    continue
                x, y = a[left[key]], b[right[key]]
                if x.dtype != y.dtype or x.shape != y.shape:
                    raise ValueError(f'{key}: shape/dtype differs')
                count += 1
                elements += x.size
                raw_type = np.dtype(f'V{x.dtype.itemsize}')
                different += int(np.count_nonzero(x.view(raw_type) != y.view(raw_type)))
                x, y = numeric(x).astype(np.float64), numeric(y).astype(np.float64)
                if not np.isfinite(x).all() or not np.isfinite(y).all():
                    raise ValueError(f'{key}: nonfinite values')
                squared += float(np.sum((x-y)**2))
                reference_squared += float(np.sum(x*x))
                maximum = max(maximum, float(np.max(np.abs(x-y), initial=0)))
            if not count:
                raise ValueError(f'No tensors checked in {group}')
            results[group] = dict(tensor_leaves=count, elements=elements, different_elements=different,
                                  bitwise_equal=different == 0, relative_l2=(squared/max(reference_squared, 1e-300))**.5,
                                  max_abs=maximum)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    runs = json.loads((args.directory / 'results.json').read_text())
    valid = [r for r in runs if r['valid']]
    trials = [Path(r.get('trial_dir', args.directory / f'{i:02d}_{r["name"]}')) for i, r in enumerate(valid)]
    initial = [hashlib.sha256((p / 'initial.safetensors').read_bytes()).hexdigest() for p in trials]
    assert len(set(initial)) == 1, 'Initial weights differ'
    baseline = trials[0] / 'checkpoints/checkpoint_final.resume.npz'
    results = dict(initial_sha256=initial[0], comparisons=[])
    for run, trial in zip(valid[1:], trials[1:]):
        results['comparisons'].append(dict(name=run['name'], groups=compare(
            baseline, trial / 'checkpoints/checkpoint_final.resume.npz')))
    (args.directory / 'state_comparison.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
