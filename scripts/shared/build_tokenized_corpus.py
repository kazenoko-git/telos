"""
High-Throughput Streaming & Tokenization Pipeline for télos.

Streams Python code from Hugging Face (e.g. codeparrot/codeparrot-clean),
tokenizes batches in parallel using the HuggingFace Rust tokenizers engine,
and serializes token IDs directly into a packed binary file (uint16 or uint32).
"""

import os
import sys
import argparse
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm
from tokenizers import Tokenizer
from datasets import load_dataset


def find_tokenizer_path() -> Path:
    candidates = [
        Path("configs/shared/tokenizer_mac.json"),
        Path("configs/tokenizer_mac.json"),
        Path("configs/tokenizer_0.json"),
        Path("configs/tokenizer.json"),
        Path("tokenizer.json")
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("Could not find a valid tokenizer.json in configs/ or root.")


def build_corpus(
    output_path: str = "data/python_corpus_2.5b.bin",
    target_tokens: int = 2_500_000_000,
    dataset_name: str = "codeparrot/codeparrot-clean",
    batch_size: int = 2000,
    dtype: str = "uint16"
):
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    tok_path = find_tokenizer_path()
    print(f"Loading tokenizer from {tok_path}...")
    tokenizer = Tokenizer.from_file(str(tok_path))
    vocab_size = tokenizer.get_vocab_size()
    print(f"Tokenizer loaded (Vocab size: {vocab_size:,})")
    
    np_dtype = np.uint16 if (dtype == "uint16" and vocab_size <= 65535) else np.uint32
    bytes_per_token = 2 if np_dtype == np.uint16 else 4
    target_gb = target_tokens * bytes_per_token / (1024**3)
    
    print(f"\nStarting streaming tokenization from '{dataset_name}'")
    print(f"Target: {target_tokens:,} tokens (~{target_gb:.2f} GB as {np_dtype.__name__})")
    print(f"Output File: {out_file.resolve()}\n")
    
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    ds_kwargs = {"split": "train", "streaming": True}
    if token:
        ds_kwargs["token"] = token
        
    ds = load_dataset(dataset_name, **ds_kwargs)
    
    total_tokens = 0
    buffer_text = []
    start_time = time.time()
    pbar = tqdm(total=target_tokens, unit="tokens", unit_scale=True)
    
    with open(out_file, "wb") as f_out:
        for sample in ds:
            code = sample.get("content") or sample.get("code") or ""
            if len(code.strip()) >= 20:
                buffer_text.append(code)
                
            if len(buffer_text) >= batch_size:
                encodings = tokenizer.encode_batch(buffer_text)
                batch_ids = []
                for enc in encodings:
                    batch_ids.extend(enc.ids)
                    
                if batch_ids:
                    arr = np.array(batch_ids, dtype=np_dtype)
                    f_out.write(arr.tobytes())
                    num_tok = len(batch_ids)
                    total_tokens += num_tok
                    pbar.update(num_tok)
                    
                buffer_text.clear()
                
                if total_tokens >= target_tokens:
                    break
                    
        # Flush remaining
        if buffer_text and total_tokens < target_tokens:
            encodings = tokenizer.encode_batch(buffer_text)
            batch_ids = []
            for enc in encodings:
                batch_ids.extend(enc.ids)
            if batch_ids:
                arr = np.array(batch_ids, dtype=np_dtype)
                f_out.write(arr.tobytes())
                total_tokens += len(batch_ids)
                pbar.update(len(batch_ids))

    pbar.close()
    elapsed = time.time() - start_time
    file_size_mb = out_file.stat().st_size / (1024**2)
    tok_per_sec = total_tokens / max(elapsed, 1e-3)
    
    print(f"\n=======================================================")
    print(f"Tokenization Complete in {elapsed/60.0:.2f} minutes ({tok_per_sec:,.0f} tok/s)!")
    print(f"Total Tokens Saved: {total_tokens:,}")
    print(f"Binary File Size:   {file_size_mb:,.1f} MB ({out_file})")
    print(f"=======================================================\n")


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
