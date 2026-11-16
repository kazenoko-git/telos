"""Copy and verify a bounded real-corpus prefix for short nonwrapping runs.

This fixture avoids fingerprinting an entire multi-gigabyte corpus during each
fresh-process benchmark. Production dataset fingerprints remain unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('source', type=Path)
parser.add_argument('output', type=Path)
parser.add_argument('--bytes', type=int, default=32 * 1024**2)
args = parser.parse_args()
if args.bytes < 1 or args.bytes % 2 or args.source.stat().st_size < args.bytes:
    parser.error('Prefix must be a positive, even byte count within the uint16 corpus')
with args.source.open('rb') as stream:
    data = stream.read(args.bytes)
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open('xb') as stream:
    stream.write(data)
if args.output.read_bytes() != data:
    raise RuntimeError('Copied corpus prefix differs from its source')
Path(str(args.output) + '.json').write_text(json.dumps({'dtype': 'uint16'}, indent=2))
args.output.with_name('corpus_prefix_provenance.json').write_text(json.dumps(dict(
    source=str(args.source.resolve()), source_bytes=args.source.stat().st_size,
    prefix_bytes=args.bytes, sha256=hashlib.sha256(data).hexdigest(), dtype='uint16',
    verified_byte_identical=True, tokens=args.bytes // 2,
    required_tokens_at_192_updates=192 * 8 * 8 * 256,
), indent=2))
