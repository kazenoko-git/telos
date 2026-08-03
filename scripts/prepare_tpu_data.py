"""Pre-tokenizes Python code datasets locally into uint16 binary files for TPU and Kaggle runs.

Generates:
1. data/train.bin (500M tokens = ~1.0 GB) for TPU 125M, 250M, 500M 1:1 ratio scaling
2. data/train_2b.bin (2.0B tokens = ~4.0 GB) for Kaggle 50M ratio study (1:1 to 1:40)
"""

import sys
import os
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

from tokenizers import Tokenizer


def pretokenize_corpus_to_bin(
    output_bin_path: str,
    target_tokens: int = 500_000_000,
    tokenizer_path: str = "configs/tokenizer_mac.json",
    dataset_name: str = "codeparrot/codeparrot-clean"
):
    """Streams dataset and pre-tokenizes directly into uint16 binary file."""
    output_path = Path(output_bin_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Pre-tokenizing {target_tokens:,} tokens into '{output_bin_path}'...")
    tokenizer = Tokenizer.from_file(tokenizer_path)

    token_buffer = []
    total_written = 0

    pbar = tqdm(total=target_tokens, unit="tokens", unit_scale=True)

    ds = load_dataset(dataset_name, split="train", streaming=True)

    with open(output_path, "wb") as f:
        for sample in ds:
            code = sample.get("content") or sample.get("code") or ""
            if not code or len(code) < 30:
                continue

            token_ids = tokenizer.encode(code).ids
            token_buffer.extend(token_ids)
            pbar.update(len(token_ids))

            # Flush to disk every 10,000,000 tokens (20 MB binary chunk)
            if len(token_buffer) >= 10_000_000:
                chunk = np.array(token_buffer[:10_000_000], dtype=np.uint16)
                f.write(chunk.tobytes())
                total_written += len(chunk)
                token_buffer = token_buffer[10_000_000:]

            if total_written + len(token_buffer) >= target_tokens:
                break

        if token_buffer:
            remaining = target_tokens - total_written
            if remaining > 0:
                chunk = np.array(token_buffer[:remaining], dtype=np.uint16)
                f.write(chunk.tobytes())
                total_written += len(chunk)

    pbar.close()
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Successfully generated '{output_bin_path}' ({total_written:,} tokens, {file_size_mb:.1f} MB binary)!")


def main():
    parser = argparse.ArgumentParser(description="Pre-tokenize Python data locally for TPU & Kaggle")
    parser.add_argument("--tpu-tokens", type=int, default=500_000_000, help="Tokens for TPU 1:1 suite (500M)")
    parser.add_argument("--kaggle-tokens", type=int, default=2_000_000_000, help="Tokens for Kaggle 50M ratio suite (2B)")
    parser.add_argument("--tokenizer", type=str, default="configs/tokenizer_mac.json", help="Tokenizer path")
    args = parser.parse_args()

    print("=" * 80)
    print("LOCAL DATA PRE-TOKENIZATION FOR TPU & KAGGLE")
    print("=" * 80 + "\n")

    # 1. Pre-tokenize TPU 500M token dataset
    pretokenize_corpus_to_bin(
        output_bin_path="data/train.bin",
        target_tokens=args.tpu_tokens,
        tokenizer_path=args.tokenizer
    )

    # 2. Pre-tokenize Kaggle 2.0B token dataset (if requested)
    if args.kaggle_tokens > 0:
        pretokenize_corpus_to_bin(
            output_bin_path="data/train_2b.bin",
            target_tokens=args.kaggle_tokens,
            tokenizer_path=args.tokenizer
        )


if __name__ == "__main__":
    main()
