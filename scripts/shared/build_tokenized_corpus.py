"""
High-Throughput Streaming & Tokenization Pipeline for télos.

Delegates directly to telos.dataprep.prepare to maintain a single source of truth.
Streams Python code from Hugging Face (e.g. codeparrot/codeparrot-clean),
tokenizes batches in parallel using the HuggingFace Rust tokenizers engine,
and serializes token IDs directly into a packed binary file (uint16 or uint32).
"""

import argparse
from pathlib import Path
from telos.dataprep.prepare import prepare_dataset


def build_corpus(
    output_path: str = "data/python_corpus_2.5b.bin",
    target_tokens: int = 2_500_000_000,
    dataset_name: str = "codeparrot/codeparrot-clean",
    batch_size: int = 2000,
    dtype: str = "uint16"
):
    """Legacy entry point wrapper calling unified telos.dataprep.prepare."""
    return prepare_dataset(
        output_path=output_path,
        dataset_name=dataset_name,
        max_tokens=target_tokens,
        batch_size=batch_size,
        dtype=dtype,
    )


def main():
    parser = argparse.ArgumentParser(description="High-Throughput Binary Corpus Builder for télos")
    parser.add_argument("--output", type=str, default="data/python_corpus_2.5b.bin", help="Output binary file path")
    parser.add_argument("--tokens", type=int, default=2_500_000_000, help="Target total tokens (e.g. 2500000000 for 2.5B)")
    parser.add_argument("--dataset", type=str, default="codeparrot/codeparrot-clean", help="HuggingFace dataset to stream")
    parser.add_argument("--batch-size", type=int, default=2000, help="Batch size for parallel Rust tokenization")
    parser.add_argument("--dtype", type=str, default="uint16", choices=["uint16", "uint32"], help="Data type for token storage")
    args = parser.parse_args()

    build_corpus(
        output_path=args.output,
        target_tokens=args.tokens,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        dtype=args.dtype
    )


if __name__ == "__main__":
    main()

