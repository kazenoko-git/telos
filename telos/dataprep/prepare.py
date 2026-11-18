"""
High-efficiency, multi-source data preparation engine for Télos.
Converts any type of corpus (plain text, directory of code files, JSONL,
Hugging Face datasets, or synthetic stream) into memory-mapped binary token arrays (.bin).
"""

import os
import re
import json
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

from telos.data.tokenizer import train_bpe_tokenizer, load_tokenizer

_LICENSE_REGEX = re.compile(
    r"(copyright\b|licensed?\s+under|apache\s+license|mit\s+license|gnu\s+general|mozilla\s+public|spdx-license|all\s+rights\s+reserved)",
    re.IGNORECASE
)


def strip_license_header(code: str) -> str:
    """
    Strips top-of-file license, copyright, and disclaimer headers from Python source.
    Scans the opening lines for comment blocks or docstrings containing legal boilerplate.
    Preserves module-level code, imports, and regular functional docstrings.
    """
    lines = code.splitlines(keepends=True)
    if not lines:
        return code

    header_end = 0
    in_block_comment = False
    block_quote_char = None
    has_license_keyword = False

    idx = 0
    while idx < min(len(lines), 50):
        line = lines[idx]
        stripped = line.strip()

        # Shebang (#!) or encoding header
        if idx == 0 and (stripped.startswith("#!") or "coding:" in stripped or "coding=" in stripped):
            idx += 1
            continue

        # Single-line comment (# ...)
        if stripped.startswith("#"):
            if _LICENSE_REGEX.search(stripped):
                has_license_keyword = True
            header_end = idx + 1
            idx += 1
            continue

        # Module-level docstring starting with triple quotes
        if not in_block_comment and (stripped.startswith('"""') or stripped.startswith("'''")):
            block_quote_char = stripped[:3]
            in_block_comment = True
            if _LICENSE_REGEX.search(stripped):
                has_license_keyword = True
            # Single-line docstring
            if len(stripped) > 3 and stripped[3:].endswith(block_quote_char):
                in_block_comment = False
                header_end = idx + 1
            idx += 1
            continue

        # Multi-line docstring continuation
        if in_block_comment:
            if _LICENSE_REGEX.search(stripped):
                has_license_keyword = True
            if block_quote_char in stripped:
                in_block_comment = False
                header_end = idx + 1
            idx += 1
            continue

        # Blank/empty lines within the opening header
        if not stripped:
            idx += 1
            continue

        # First line of actual executable Python code
        break

    if has_license_keyword and header_end > 0:
        return "".join(lines[header_end:]).lstrip()
    return code


# Unicode characters, rather than bytes: each chunk is at most 1 MiB in UTF-8.
DEFAULT_CHUNK_CHARS = 256 * 1024
DEFAULT_BATCH_CHARS = 1024 * 1024


def _clean_text_chunks(chunks):
    """Clean the opening chunk only; preserve subsequent whitespace and short tails."""
    chunks = iter(chunks)
    first = next(chunks, "")
    second = next(chunks, None)
    first = strip_license_header(first)
    if second is None:
        if len(first.strip()) >= 20:
            yield first
        return
    if first:
        yield first
    yield second
    yield from chunks


def _string_chunks(text, chunk_chars):
    return (text[start:start + chunk_chars] for start in range(0, len(text), chunk_chars))


def iterate_text_sources(
    source_path: str | Path | None = None,
    dataset_name: str | None = None,
    dataset_split: str = "train",
    text_key: str = "content",
    extensions: list[str] | None = None,
    limit_docs: int | None = None,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
):
    """Yield bounded text chunks; limit_docs counts source files or records.

    Plain files are read incrementally, including files without newlines. JSONL
    and HF records are decoded by their readers before their text is chunked.
    """
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be positive")
    doc_count = 0
    exts = tuple(extensions or [".py", ".txt", ".md", ".json", ".rs", ".js", ".ts", ".c", ".cpp"])

    def documents():
        if dataset_name:
            from datasets import load_dataset
            ds = load_dataset(dataset_name, split=dataset_split, streaming=True)
            for row in ds:
                text = row.get(text_key) or row.get("text") or row.get("code") or ""
                yield _string_chunks(text, chunk_chars)
        elif source_path:
            p = Path(source_path)
            if not p.exists():
                raise FileNotFoundError(f"Source path not found: {source_path}")
            if p.is_dir():
                paths = (Path(root) / name for root, _, files in os.walk(p)
                         for name in files if name.endswith(exts))
            elif p.suffix == ".jsonl":
                with open(p, "r", encoding="utf-8", errors="ignore") as stream:
                    for line in stream:
                        if line.strip():
                            row = json.loads(line)
                            text = row.get(text_key) or row.get("text") or row.get("code") or ""
                            yield _string_chunks(text, chunk_chars)
                return
            else:
                paths = [p]
            for path in paths:
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as stream:
                        yield iter(lambda: stream.read(chunk_chars), "")
                except OSError:
                    if not p.is_dir():
                        raise

    for chunks in documents():
        emitted = False
        for chunk in _clean_text_chunks(chunks):
            emitted = True
            yield chunk
        if emitted:
            doc_count += 1
            if limit_docs and doc_count >= limit_docs:
                return


def _text_batches(chunks, batch_size, max_batch_chars):
    """Bound concurrent tokenizer inputs by both count and character volume."""
    batch = []
    chars = 0
    for chunk in chunks:
        if batch and (len(batch) >= batch_size or chars + len(chunk) > max_batch_chars):
            yield batch
            batch = []
            chars = 0
        batch.append(chunk)
        chars += len(chunk)
    if batch:
        yield batch


def prepare_dataset(
    corpus: str | Path | None = None,
    output_path: str | Path = "data/python_corpus.bin",
    tokenizer_path: str | Path | None = None,
    dataset_name: str | None = None,
    dataset_split: str = "train",
    text_key: str = "content",
    train_tokenizer: bool = False,
    vocab_size: int = 8192,
    synthetic: bool = False,
    synthetic_tokens: int = 100_000,
    seq_len: int = 512,
    batch_size: int = 8,
    max_tokens: int | None = None,
    dtype: str = "uint16",
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    max_batch_chars: int = DEFAULT_BATCH_CHARS,
) -> str:
    """
    High-efficiency tokenization and binary packaging pipeline.
    Bounds text chunks and tokenizer batches, then writes each encoding directly.
    One tokenizer instance is reused for the complete run.
    """
    if batch_size <= 0 or chunk_chars <= 0 or max_batch_chars <= 0:
        raise ValueError("batch_size, chunk_chars, and max_batch_chars must be positive")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if max_tokens is not None and max_tokens < 0:
        raise ValueError("max_tokens must be non-negative")

    # Fast path: Synthetic data stream
    if synthetic:
        n_synth = max_tokens if max_tokens is not None else synthetic_tokens
        print(f"  [DataPrep] Generating {n_synth:,} synthetic tokens...")
        np_dtype = np.uint16 if vocab_size <= 65536 else np.int32
        dtype_str = "uint16" if np_dtype == np.uint16 else "int32"
        arr = np.random.randint(0, vocab_size, (n_synth,), dtype=np_dtype)
        with open(output_path, "wb") as f:
            f.write(arr.tobytes())
        
        meta = {
            "dtype": dtype_str,
            "vocab_size": vocab_size,
            "total_tokens": n_synth,
            "synthetic": True,
        }
        meta_path = Path(str(output_path) + ".json")
        with open(meta_path, "w", encoding="utf-8") as mf:
            json.dump(meta, mf, indent=2)
            
        print(f"  [DataPrep] Saved synthetic binary corpus to {output_path} ({os.path.getsize(output_path) / 1e6:.2f} MB)")
        return str(output_path)

    # Resolve Tokenizer
    tok = None
    if train_tokenizer:
        print(f"  [DataPrep] Training new ByteLevel BPE Tokenizer (vocab_size={vocab_size})...")
        # Collect sample text files for training
        sample_files = []
        if corpus and Path(corpus).is_dir():
            for root, _, files in os.walk(corpus):
                for fl in files:
                    if fl.endswith((".py", ".txt")):
                        sample_files.append(str(Path(root) / fl))
                        if len(sample_files) >= 500:
                            break
                if len(sample_files) >= 500:
                    break
        elif corpus:
            sample_files = [str(corpus)]

        tok_save_path = str(tokenizer_path or "configs/shared/tokenizer_custom.json")
        tok = train_bpe_tokenizer(sample_files, vocab_size=vocab_size, save_path=tok_save_path)
    else:
        # Check canonical candidate paths if tokenizer_path not explicitly supplied
        if tokenizer_path is None:
            for cand in [
                Path("configs/shared/tokenizer_mac.json"),
                Path("configs/tokenizer_mac.json"),
                Path("configs/tokenizer_0.json"),
                Path("configs/tokenizer.json"),
            ]:
                if cand.exists():
                    tokenizer_path = cand
                    break
        tok = load_tokenizer(str(tokenizer_path) if tokenizer_path else None)

    actual_vocab = tok.get_vocab_size() if tok else vocab_size
    np_dtype = np.uint16 if (dtype == "uint16" and actual_vocab <= 65536) else np.int32
    dtype_str = "uint16" if np_dtype == np.uint16 else "int32"

    print(f"  [DataPrep] Processing corpus into {output_path} (vocab_size={actual_vocab}, dtype={dtype_str})...")
    if max_tokens is not None:
        print(f"  [DataPrep] Target token limit: {max_tokens:,} tokens (~{max_tokens * (2 if np_dtype == np.uint16 else 4) / (1024**3):.2f} GB)")
    total_tokens = 0

    with tqdm(total=max_tokens, unit="tokens", unit_scale=True, desc="Tokenizing") as pbar:
        with open(output_path, "wb") as out_f:
            if max_tokens != 0:
                chunks = iterate_text_sources(
                    corpus, dataset_name, dataset_split, text_key,
                    chunk_chars=min(chunk_chars, max_batch_chars),
                )
                for batch in _text_batches(chunks, batch_size, max_batch_chars):
                    encodings = tok.encode_batch(batch)
                    for enc in encodings:
                        ids = enc.ids
                        if max_tokens is not None:
                            remaining = max_tokens - total_tokens
                            if len(ids) > remaining:
                                ids = ids[:remaining]
                        if ids:
                            arr = np.asarray(ids, dtype=np_dtype)
                            arr.tofile(out_f)
                            total_tokens += len(arr)
                            pbar.update(len(arr))
                        if max_tokens is not None and total_tokens >= max_tokens:
                            break
                    # Release Rust encodings before allocating the next batch.
                    del encodings, enc, ids
                    if max_tokens is not None and total_tokens >= max_tokens:
                        break

    # Save metadata sidecar
    meta = {
        "dtype": dtype_str,
        "vocab_size": actual_vocab,
        "total_tokens": total_tokens,
        "synthetic": False,
    }
    meta_path = Path(str(output_path) + ".json")
    with open(meta_path, "w", encoding="utf-8") as mf:
        json.dump(meta, mf, indent=2)

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"  [DataPrep] Finished! Total tokens: {total_tokens:,} | File size: {size_mb:.2f} MB | Path: {output_path}")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Télos High-Efficiency Data Preparation")
    parser.add_argument("--corpus", type=str, default=None, help="Path to text file, directory of code, or JSONL")
    parser.add_argument("--dataset", type=str, default=None, help="Hugging Face dataset name (e.g. codeparrot/codeparrot-clean)")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--text-key", type=str, default="content", help="Key for text in JSONL / HF dataset")
    parser.add_argument("--output", type=str, default="data/python_corpus.bin", help="Output binary array path (.bin)")
    parser.add_argument("--tokenizer", type=str, default=None, help="Path to BPE tokenizer JSON")
    parser.add_argument("--train-tokenizer", action="store_true", help="Train a new BPE tokenizer on the corpus")
    parser.add_argument("--vocab-size", type=int, default=8192, help="Vocabulary size")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic token stream directly")
    parser.add_argument("--tokens", type=int, default=None, help="Target total tokens (e.g. 15000000000 for 15B)")
    parser.add_argument("--batch-size", type=int, default=2000, help="Batch size for parallel Rust tokenization")
    parser.add_argument("--dtype", type=str, default="uint16", choices=["uint16", "uint32"], help="Data type for token storage")

    args = parser.parse_args()
    prepare_dataset(
        corpus=args.corpus,
        output_path=args.output,
        tokenizer_path=args.tokenizer,
        dataset_name=args.dataset,
        dataset_split=args.split,
        text_key=args.text_key,
        train_tokenizer=args.train_tokenizer,
        vocab_size=args.vocab_size,
        synthetic=args.synthetic,
        synthetic_tokens=args.tokens or 100_000,
        batch_size=args.batch_size,
        max_tokens=args.tokens,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()
