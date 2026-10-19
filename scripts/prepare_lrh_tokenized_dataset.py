#!/usr/bin/env python3
"""Pre-tokenize balanced multi-domain code corpus on M5 Pro into uint32 memory-mapped binaries.

Multi-Domain Recipe for Qwen2.5-Coder:
- 50% Python: codeparrot/codeparrot-clean
- 30% Polyglot Code: code_search_net (all: Java, JavaScript, Go, Ruby, PHP)
- 20% Technical Text & Markdown: HuggingFaceFW/fineweb-edu (sample-10BT)

Outputs:
- data/qwen25_coder/qwen25_coder_10m.bin (40 MB, uint32)
- data/qwen25_coder/qwen25_coder_100m.bin (400 MB, uint32)
- data/qwen25_coder/metadata.json

Usage:
    .venv/bin/python3 scripts/prepare_lrh_tokenized_dataset.py --target-tokens 100000000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def compute_sha256(file_path: Path) -> str:
    """Computes SHA-256 hash of a file for integrity verification."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def stream_multi_domain_code() -> Iterator[str]:
    """Interleave Python (50%), Polyglot Code (30%), and Technical Text (20%)."""
    print("Initializing multi-domain streams from Hugging Face...")
    py_ds = iter(load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True))
    poly_ds = iter(load_dataset("code_search_net", "all", split="train", streaming=True))
    text_ds = iter(load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True))

    while True:
        # 1. 5 Python documents (50%)
        for _ in range(5):
            try:
                item = next(py_ds)
                txt = item.get("content", "")
                if txt.strip():
                    yield txt
            except StopIteration:
                pass

        # 2. 3 Polyglot code documents with docstrings (30%)
        for _ in range(3):
            try:
                item = next(poly_ds)
                doc = item.get("func_documentation_string", "")
                code = item.get("whole_func_string") or item.get("func_code_string", "")
                txt = f"{doc}\n{code}".strip()
                if txt:
                    yield txt
            except StopIteration:
                pass

        # 3. 2 Technical educational prose & math documents (20%)
        for _ in range(2):
            try:
                item = next(text_ds)
                txt = item.get("text", "")
                if txt.strip():
                    yield txt
            except StopIteration:
                pass


def main():
    parser = argparse.ArgumentParser(description="Pre-tokenize balanced multi-domain code corpus into compact uint32 binary")
    parser.add_argument("--tokenizer-id", type=str, default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    parser.add_argument("--target-tokens", type=int, default=100_000_000, help="Target total tokens (default: 100M)")
    parser.add_argument("--output-dir", type=str, default="data/qwen25_coder")
    parser.add_argument("--snapshot-10m", action="store_true", default=True, help="Save explicit 10M milestone file")
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("M5 PRO PRE-TOKENIZATION PIPELINE FOR QWEN2.5-CODER LRH")
    print("=" * 80)
    print(f"Tokenizer: {args.tokenizer_id}")
    print("Data Recipe: 50% Python (CodeParrot) | 30% Polyglot (CodeSearchNet) | 20% Technical (FineWeb-Edu)")
    print(f"Target Tokens: {args.target_tokens:,}")
    print(f"Output Directory: {out_dir}")

    # 1. Load Tokenizer
    t0_tok = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id, trust_remote_code=True)
    vocab_size = len(tokenizer)
    print(f"Loaded tokenizer with vocabulary size: {vocab_size:,} (fits in uint32).")

    # 2. Setup Binary Files
    file_100m = out_dir / "qwen25_coder_100m.bin"
    file_10m = out_dir / "qwen25_coder_10m.bin"

    f_100m = open(file_100m, "wb")
    f_10m = open(file_10m, "wb") if args.snapshot_10m else None

    # 3. Stream Multi-Domain Dataset
    stream = stream_multi_domain_code()

    tokens_total = 0
    t_start = time.perf_counter()
    log_interval = 2_000_000  # Report every 2M tokens
    next_log = log_interval

    buffer: List[int] = []
    chunk_flush_size = 500_000  # Flush in 500k token chunks to keep RAM < 50MB

    saved_10m = False

    try:
        for content in stream:
            ids = tokenizer.encode(content, add_special_tokens=False)
            buffer.extend(ids)
            tokens_total += len(ids)

            # Flush chunk to disk
            if len(buffer) >= chunk_flush_size:
                arr = np.array(buffer, dtype=np.uint32)
                f_100m.write(arr.tobytes())

                # Write to 10M file if within first 10M tokens
                if f_10m is not None and not saved_10m:
                    tokens_in_10m = min(len(buffer), 10_000_000 - (tokens_total - len(buffer)))
                    if tokens_in_10m > 0:
                        f_10m.write(arr[:tokens_in_10m].tobytes())
                    if tokens_total >= 10_000_000:
                        f_10m.close()
                        saved_10m = True
                        print(f"\n>>> MILESTONE 10M COMPLETED: {file_10m.name} ({os.path.getsize(file_10m) / (1024*1024):.1f} MB) <<<\n")

                buffer.clear()

            # Progress Logging
            if tokens_total >= next_log:
                elapsed = time.perf_counter() - t_start
                tok_per_sec = tokens_total / max(1e-6, elapsed)
                pct = (tokens_total / args.target_tokens) * 100
                remaining_sec = (args.target_tokens - tokens_total) / max(1e-6, tok_per_sec)
                print(
                    f"Processed: {tokens_total:>10,d} / {args.target_tokens:,} tokens ({pct:>5.1f}%) | "
                    f"Speed: {tok_per_sec:,.0f} tok/s | "
                    f"ETA: {remaining_sec / 60:>4.1f} min"
                )
                next_log += log_interval

            if tokens_total >= args.target_tokens:
                break

    finally:
        # Final flush
        if buffer:
            arr = np.array(buffer, dtype=np.uint32)
            f_100m.write(arr.tobytes())
            if f_10m is not None and not saved_10m:
                tokens_in_10m = min(len(buffer), 10_000_000 - (tokens_total - len(buffer)))
                if tokens_in_10m > 0:
                    f_10m.write(arr[:tokens_in_10m].tobytes())
                f_10m.close()

        f_100m.close()

    total_time = time.perf_counter() - t_start
    print("-" * 80)
    print(f"Tokenization complete in {total_time / 60:.2f} minutes!")
    print(f"Total tokens generated: {tokens_total:,}")
    print(f"Average speed: {tokens_total / total_time:,.0f} tokens/second.")

    size_100m = os.path.getsize(file_100m) / (1024 * 1024)
    size_10m = os.path.getsize(file_10m) / (1024 * 1024) if file_10m.exists() else 0.0

    print(f"\nFiles generated:")
    print(f"  - 10M File:  {file_10m.name} ({size_10m:.2f} MB)")
    print(f"  - 100M File: {file_100m.name} ({size_100m:.2f} MB)")

    # Metadata record
    meta = {
        "tokenizer_id": args.tokenizer_id,
        "vocab_size": vocab_size,
        "dtype": "uint32",
        "bytes_per_token": 4,
        "recipe": {
            "python_pct": 50,
            "polyglot_pct": 30,
            "technical_text_pct": 20,
            "sources": [
                "codeparrot/codeparrot-clean",
                "code_search_net (all)",
                "HuggingFaceFW/fineweb-edu (sample-10BT)",
            ],
        },
        "tokens_10m_file": file_10m.name,
        "tokens_100m_file": file_100m.name,
        "actual_tokens_100m": tokens_total,
        "generation_time_minutes": total_time / 60,
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved metadata to {out_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
