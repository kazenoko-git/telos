"""
High-efficiency, multi-source data preparation engine for Télos.
Converts any type of corpus (plain text, directory of code files, JSONL,
Hugging Face datasets, or synthetic stream) into memory-mapped binary token arrays (.bin).
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

from telos.data.tokenizer import train_bpe_tokenizer, load_tokenizer


def iterate_text_sources(
    source_path: str | Path | None = None,
    dataset_name: str | None = None,
    dataset_split: str = "train",
    text_key: str = "content",
    extensions: list[str] | None = None,
    limit_docs: int | None = None
):
    """
    Yields raw text documents from various sources:
    1. Directory of code files (recursively matching extensions)
    2. Single text file (.txt, .py, etc.)
    3. JSONL file (extracting text_key)
    4. Hugging Face dataset
    """
    doc_count = 0
    exts = tuple(extensions or [".py", ".txt", ".md", ".json", ".rs", ".js", ".ts", ".c", ".cpp"])

    # Source 1: Hugging Face dataset
    if dataset_name:
        from datasets import load_dataset
        ds = load_dataset(dataset_name, split=dataset_split, streaming=True)
        for row in ds:
            text = row.get(text_key) or row.get("text") or row.get("code") or ""
            if text.strip():
                yield text
                doc_count += 1
                if limit_docs and doc_count >= limit_docs:
                    return

    # Source 2: Local path (Directory, Text File, or JSONL)
    elif source_path:
        p = Path(source_path)
        if not p.exists():
            raise FileNotFoundError(f"Source path not found: {source_path}")

        if p.is_dir():
            for root, _, files in os.walk(p):
                for f in files:
                    if f.endswith(exts):
                        f_path = Path(root) / f
                        try:
                            with open(f_path, "r", encoding="utf-8", errors="ignore") as fl:
                                content = fl.read()
                                if content.strip():
                                    yield content
                                    doc_count += 1
                                    if limit_docs and doc_count >= limit_docs:
                                        return
                        except Exception:
                            continue

        elif p.suffix == ".jsonl":
            with open(p, "r", encoding="utf-8", errors="ignore") as fl:
                for line in fl:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        content = data.get(text_key) or data.get("text") or data.get("code") or ""
                        if content.strip():
                            yield content
                            doc_count += 1
                            if limit_docs and doc_count >= limit_docs:
                                return
        else:
            with open(p, "r", encoding="utf-8", errors="ignore") as fl:
                content = fl.read()
                if content.strip():
                    yield content


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
    batch_size: int = 1000
) -> str:
    """
    High-efficiency tokenization and binary packaging pipeline.
    Writes tokens in contiguous uint16 chunks directly to disk.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Fast path: Synthetic data stream
    if synthetic:
        print(f"  [DataPrep] Generating {synthetic_tokens:,} synthetic tokens...")
        dtype = np.uint16 if vocab_size <= 65536 else np.int32
        arr = np.random.randint(0, vocab_size, (synthetic_tokens,), dtype=dtype)
        with open(output_path, "wb") as f:
            f.write(arr.tobytes())
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
        elif corpus:
            sample_files = [str(corpus)]

        tok_save_path = str(tokenizer_path or "configs/shared/tokenizer_custom.json")
        tok = train_bpe_tokenizer(sample_files, vocab_size=vocab_size, save_path=tok_save_path)
    else:
        try:
            tok = load_tokenizer(str(tokenizer_path or "configs/shared/tokenizer_0.json"))
        except Exception:
            print("  [DataPrep] Notice: Standard tokenizer not found. Creating default tokenizer...")
            tok = train_bpe_tokenizer([], vocab_size=vocab_size, save_path="configs/shared/tokenizer_default.json")

    print(f"  [DataPrep] Processing corpus into {output_path}...")
    dtype = np.uint16 if vocab_size <= 65536 else np.int32
    total_tokens = 0
    buffer = []

    with open(output_path, "wb") as out_f:
        for doc in tqdm(iterate_text_sources(corpus, dataset_name, dataset_split, text_key), desc="Tokenizing"):
            buffer.append(doc)
            if len(buffer) >= batch_size:
                encodings = tok.encode_batch(buffer)
                flat_tokens = []
                for enc in encodings:
                    flat_tokens.extend(enc.ids)
                if flat_tokens:
                    arr = np.array(flat_tokens, dtype=dtype)
                    out_f.write(arr.tobytes())
                    total_tokens += len(flat_tokens)
                buffer = []

        # Flush remainder
        if buffer:
            encodings = tok.encode_batch(buffer)
            flat_tokens = []
            for enc in encodings:
                flat_tokens.extend(enc.ids)
            if flat_tokens:
                arr = np.array(flat_tokens, dtype=dtype)
                out_f.write(arr.tobytes())
                total_tokens += len(flat_tokens)

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
    parser.add_argument("--tokens", type=int, default=100_000, help="Number of synthetic tokens to generate")

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
        synthetic_tokens=args.tokens
    )


if __name__ == "__main__":
    main()
