# Bounded corpus preparation

`telos dataprep` streams plain text and code files into packed token arrays. A
large `.txt` file does not need to be split manually, including when it contains
one very long line. Directory inputs use the same bounded reader.

```bash
telos dataprep --corpus /kaggle/input/my-corpus/corpus.txt \
  --tokenizer /kaggle/input/my-tokenizer/tokenizer.json \
  --output /kaggle/working/corpus.bin --batch-size 8
```

The tokenizer is loaded once and reused throughout the run. By default:

- `--chunk-chars 262144` limits each piece to 262,144 Unicode characters
  (at most 1 MiB of UTF-8 text).
- `--batch-size 8` limits the number of pieces passed to `encode_batch`.
- `--max-batch-chars 1048576` also caps the combined characters in a batch.
  This cap still applies when `--batch-size` is increased.

The effective chunk size is the smaller of `chunk_chars` and `max_batch_chars`.
All three settings must be positive. These are input bounds, not a total RAM
limit: tokenizer vocabulary, native workers, encodings and allocator caches add
memory overhead. To reduce that overhead on a constrained machine, try
`--chunk-chars 65536 --max-batch-chars 131072 --batch-size 2`.

Each encoding is converted and written separately; Telos does not concatenate
all batch token IDs into a Python list. `--tokens N` writes at most N tokens,
including when the limit falls inside a chunk. Zero produces an empty corpus.

Chunks preserve text order, Unicode characters, whitespace and short final
pieces after opening-header cleanup. License cleanup applies only to the opening
chunk of each document, using the existing 50-line rule. A header extending past
that chunk is not fully removed. Each piece is tokenized independently: BPE
merges, tokenizer normalization and configured special tokens can change output
at chunk boundaries. No overlap or extra separators are added by Telos. Keep
chunk settings fixed when reproducing a corpus; do not assume equivalence with
whole-document tokenization.

JSONL and streamed Hugging Face records are also split before tokenization.
Their readers still decode a complete JSON line or dataset record first, so a
single enormous record can consume substantial memory before chunking. For
such inputs, use bounded records or convert them to plain text. Tokenizer
training (`--train-tokenizer`) and synthetic data generation have separate
allocation behavior; these bounds apply to corpus encoding.

The Python API exposes the same controls:

```python
from telos import prepare_dataset

prepare_dataset(
    corpus="corpus.txt",
    output_path="corpus.bin",
    batch_size=8,
    chunk_chars=262144,
    max_batch_chars=1048576,
)
```

## Local validation (2026-11-18)

The targeted dataprep, dataloader and release-note tests passed (33 tests).
Regression coverage includes guarded bounded reads, Unicode and short tails,
JSONL/HF chunking, document limits, character/count batch limits, exact token
prefixes and metadata, tokenizer reuse, CLI forwarding and real BPE output.

Separate macOS processes using tokenizers 0.23.1 and the bundled 4,096-token
vocabulary encoded newline-free files containing repeated
`hello world with sample training text. ` with default memory controls:

| Input size | Peak process RSS | Written tokens |
| --- | --- | --- |
| 2 MiB | 248.6 MiB | 430,196 |
| 32 MiB | 513.0 MiB | 6,883,127 |
| 128 MiB | 494.3 MiB | 27,532,512 |

Files were generated incrementally before encoding, and peak RSS was read using
`resource.getrusage(resource.RUSAGE_SELF).ru_maxrss` (bytes on macOS). Input and
output files lived in temporary directories. These observations show similar
peak memory for the two larger files, including tokenizer worker/allocator
warmup; they are not a universal RAM ceiling or a Kaggle measurement.
