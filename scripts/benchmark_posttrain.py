"""Small reproducible SFT/DPO end-to-end benchmark (including setup/export).

Select a source checkout with --source to compare revisions in fresh processes.
This synthetic diagnostic is not a trained-model quality or sustained speed test.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--method', choices=['sft', 'dpo'], default='sft')
    parser.add_argument('--hardware', choices=['cpu', 'mlx', 'cuda'], default='mlx')
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--length', type=int, default=64)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--records', type=int, default=12)
    parser.add_argument('--precision', choices=['auto', 'float32', 'bf16'], default='auto')
    parser.add_argument('--no-cache-reference', action='store_true')
    parser.add_argument('--compile', action='store_true', help='Pass the v1.7 compile option; omit for older revisions')
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    import torch
    from telos.models import TelosTransformer
    from telos.posttrain.trainer import posttrain
    from telos.data.tokenizer import load_tokenizer
    import telos

    if args.output.exists():
        raise FileExistsError(f'Choose a fresh output directory: {args.output}')
    args.output.mkdir(parents=True)
    torch.manual_seed(29)
    tokenizer = load_tokenizer()
    model = TelosTransformer(vocab_size=len(tokenizer.get_vocab()), d_model=args.width,
                             n_layers=args.layers, n_heads=2, max_seq_len=args.length, is_causal=True)
    source = args.output / 'base.pt'
    torch.save({'model_state_dict': model.state_dict(), 'config': {
        'paradigm': 'ar', 'model': vars(model.config)}}, source)
    # Distinct answer prefixes survive truncation; whole-batch permutations
    # should not collapse into the same synthetic example/cache key.
    rows = [{'prompt': f'Question {i}?', 'response': f'Answer {i}: ' + 'hello ' * args.length}
            if args.method == 'sft' else {'prompt': f'Question {i}?',
                'chosen': f'Answer {i}: ' + 'hello ' * args.length,
                'rejected': f'Wrong {i}: ' + 'world ' * args.length}
            for i in range(args.records)]
    data = args.output / 'data.jsonl'
    data.write_text('\n'.join(json.dumps(row) for row in rows))
    options = dict(method=args.method, hardware=args.hardware, max_steps=args.steps,
                   batch_size=args.batch_size, max_length=args.length, save_every=0, seed=42,
                   precision=args.precision, cache_reference=not args.no_cache_reference)
    if args.compile:
        options['compile_step'] = True
    if args.hardware == 'mlx':
        import mlx.core as mx
        synchronize = mx.synchronize
        mx.reset_peak_memory()
    elif args.hardware == 'cuda':
        synchronize = torch.cuda.synchronize
    else:
        synchronize = lambda: None
    synchronize()
    started = time.perf_counter()
    result = posttrain(source, data, args.output / 'result', **options)
    synchronize()
    elapsed = time.perf_counter() - started
    package = Path(telos.__file__).parent
    manifest = {str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
                for directory in ('posttrain', 'models', 'training')
                for path in sorted((package / directory).glob('*.py'))}
    (args.output / 'measurement.json').write_text(json.dumps({
        'source': str(args.source.resolve()), 'method': args.method,
        'hardware': args.hardware, 'updates': args.steps, 'batch_size': args.batch_size,
        'sequence_length': args.length, 'width': args.width, 'layers': args.layers,
        'records': args.records, 'precision': args.precision,
        'cache_reference': not args.no_cache_reference,
        'elapsed_end_to_end_seconds': elapsed, 'updates_per_second': args.steps / elapsed,
        'peak_mlx_bytes': mx.get_peak_memory() if args.hardware == 'mlx' else None,
        'checkpoint': str(result), 'source_manifest': manifest,
    }, indent=2))
    print(f'Completed {args.steps} updates plus setup/export in {elapsed:.3f}s')


if __name__ == '__main__':
    main()
