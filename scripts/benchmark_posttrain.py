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
    model = TelosTransformer(vocab_size=len(tokenizer.get_vocab()), d_model=128,
                             n_layers=2, n_heads=2, max_seq_len=64, is_causal=True)
    source = args.output / 'base.pt'
    torch.save({'model_state_dict': model.state_dict(), 'config': {
        'paradigm': 'ar', 'model': vars(model.config)}}, source)
    rows = [{'prompt': f'Question {i}?', 'response': 'hello ' * 100} if args.method == 'sft'
            else {'prompt': f'Question {i}?', 'chosen': 'hello ' * 100, 'rejected': 'world ' * 100}
            for i in range(12)]
    data = args.output / 'data.jsonl'
    data.write_text('\n'.join(json.dumps(row) for row in rows))
    options = dict(method=args.method, hardware=args.hardware, max_steps=args.steps,
                   batch_size=2, max_length=64, save_every=0, seed=42)
    if args.compile:
        options['compile_step'] = True
    if args.hardware == 'mlx':
        import mlx.core as mx
        synchronize = mx.synchronize
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
        'hardware': args.hardware, 'updates': args.steps, 'batch_size': 2,
        'sequence_length': 64, 'elapsed_end_to_end_seconds': elapsed,
        'checkpoint': str(result), 'source_manifest': manifest,
    }, indent=2))
    print(f'Completed {args.steps} updates plus setup/export in {elapsed:.3f}s')


if __name__ == '__main__':
    main()
