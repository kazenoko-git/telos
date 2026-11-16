"""Inspect full attention trials and evaluate a held-out real-corpus tail.

Run after timing, in a separate process. This is a finite short-run diagnostic,
not a convergence or general model-quality evaluation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from telos.diffusion.ar import ar_loss_fn_mlx
from telos.models.mlx_transformer import MLXTelosTransformer
from telos.training.core import build_special_token_lut
from compare_mlx_fusion_states import compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--data', type=Path, required=True)
    args = parser.parse_args()
    assert mx.__version__ == '0.32.4', 'Use the same isolated runtime as training'
    mx.set_memory_limit(2 * 1024**3)
    mx.set_cache_limit(32 * 1024**2)
    results = json.loads((args.directory / 'results.json').read_text())
    assert len(results) == 8 and all(r['valid'] for r in results)
    data = np.memmap(args.data, dtype=np.uint16, mode='r')
    offset = len(data) - 32 * 2048
    assert offset >= max(r['summary']['input_tokens_processed'] for r in results)
    output = dict(mlx_version=mx.__version__, mlx_module=mx.__file__, heldout_token_offset=offset,
                  limitations=['32 contiguous held-out sequences per length, not an independent convergence study.'],
                  tensor_comparisons=[], repeat_comparisons=[], heldout=[], backward_graphs=[])
    for length in (2048, 256):
        runs = [r for r in results if r['length'] == length]
        baseline = Path(runs[0]['trial_dir']) / 'checkpoints/checkpoint_final.resume.npz'
        for run in runs[1:]:
            output['tensor_comparisons'].append(dict(length=length, name=run['name'],
                groups=compare(baseline, Path(run['trial_dir']) / 'checkpoints/checkpoint_final.resume.npz')))
        output['repeat_comparisons'].append(dict(length=length, mode='fallback', groups=compare(
            Path(runs[1]['trial_dir']) / 'checkpoints/checkpoint_final.resume.npz',
            Path(runs[2]['trial_dir']) / 'checkpoints/checkpoint_final.resume.npz')))
        samples = np.array(data[offset:offset + 32 * length]).reshape(32, length)
        digest = hashlib.sha256(samples.tobytes()).hexdigest()
        for run in runs:
            trial = Path(run['trial_dir'])
            config = json.loads((trial / 'checkpoints/config.json').read_text())
            model = MLXTelosTransformer(**config['model'])
            model.load_weights(str(trial / 'checkpoints/model.safetensors'))
            model.eval()
            lut = build_special_token_lut(config['model']['vocab_size'])
            graph_path = args.directory / f'full_graph_t{length}_{run["mode"]}.dot'
            graph_key = (length, run['mode'])
            if graph_key not in {(g['length'], g['mode']) for g in output['backward_graphs']}:
                # Reconstruct the actual training shape and kernel options,
                # then inspect its backward graph without timing/evaluating it.
                if not graph_path.exists():
                    os.environ['MLX_SDPA_VJP_FALLBACK'] = '0' if run['mode'] == 'native' else '1'
                    for layer in model.layers:
                        layer.force_fused_attention = True
                        layer.mlp.use_experimental_packed_kernel = True
                    batch = mx.array(samples[:run['metadata']['actual_batch_size']].astype(np.int32))
                    graph = nn.value_and_grad(model, lambda m: ar_loss_fn_mlx(m, batch, model.vocab_size, lut)[0])(model)
                    mx.export_to_dot(str(graph_path), *[v for _, v in tree_flatten(graph)])
                    del graph, batch
                count = graph_path.read_text().count('ScaledDotProductAttentionVJP')
                assert (count > 0) == (run['mode'] == 'native'), (run['name'], count)
                output['backward_graphs'].append(dict(length=length, mode=run['mode'],
                    native_vjp_occurrences=count, batch_size=run['metadata']['actual_batch_size']))
                for layer in model.layers:
                    layer.force_fused_attention = False
                    layer.mlp.use_experimental_packed_kernel = False
            losses = []
            for sample in samples:
                loss, _ = ar_loss_fn_mlx(model, mx.array(sample[None, :].astype(np.int32)), model.vocab_size, lut)
                value = loss.item()
                assert np.isfinite(value)
                losses.append(value)
            output['heldout'].append(dict(name=run['name'], length=length, mode=run['mode'],
                sequences=32, token_sha256=digest, losses=losses, mean_ce=float(np.mean(losses))))
            del model, lut
            mx.clear_cache()
            (args.directory / 'checkpoint_validation.json').write_text(json.dumps(output, indent=2))
            print(run['name'], float(np.mean(losses)), flush=True)


if __name__ == '__main__':
    main()
