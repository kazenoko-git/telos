"""
télos — Full Master Dataset Preparation Script for 85M Ratio Study
===================================================================
Streams & prepares up to 1.7 Billion tokens (3,320,312 sequences of length 512)
of high-quality Python code, tokenizes with 8,192 BPE vocab, and saves 
memmapped binary files: data/python_corpus_1.7b.bin.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import time
import numpy as np
from telos.data.tokenizer import load_tokenizer, train_bpe_tokenizer, PAD_TOKEN_ID
from telos.data.prepare import prepare_online_corpus


def prepare_dataset(target_tokens: int = 1_700_000_000, seq_len: int = 512):
    corpus_path = Path("data/python_corpus_1.7b.txt")
    bin_path = Path("data/python_corpus_1.7b.bin")
    meta_path = Path("data/python_corpus_1.7b.meta")
    tokenizer_path = Path("configs/tokenizer_mac.json")

    # Step 1: Download & prepare text corpus if not fully prepared
    if not corpus_path.exists() or corpus_path.stat().st_size < target_tokens * 3:
        print(f"==================================================================")
        print(f"  Step 1: Downloading & preparing text corpus ({target_tokens:,} tokens)")
        print(f"==================================================================")
        prepare_online_corpus(
            output_path=str(corpus_path),
            target_tokens=target_tokens,
            dataset_name="codeparrot/codeparrot-clean",
            raw_mode=True,
            fast_mode=False
        )

    # Step 2: Ensure BPE Tokenizer exists
    if not tokenizer_path.exists():
        print(f"==================================================================")
        print(f"  Step 2: Training 8,192 BPE Tokenizer -> {tokenizer_path}")
        print(f"==================================================================")
        train_bpe_tokenizer([str(corpus_path)], vocab_size=8192, save_path=str(tokenizer_path))

    tokenizer = load_tokenizer(str(tokenizer_path))

    # Step 3: Stream and tokenize corpus into binary int32 array
    if not bin_path.exists() or (meta_path.exists() and int(meta_path.read_text().strip()) * seq_len < target_tokens):
        print(f"==================================================================")
        print(f"  Step 3: Tokenizing corpus -> {bin_path}")
        print(f"==================================================================")
        start_t = time.time()
        batch_snippets = []
        total_samples = 0
        batch_limit = 32000
        block_lines = []

        with open(bin_path, "wb") as out_bin, \
             open(corpus_path, "r", encoding="utf-8") as text_in:

            def flush_batch():
                nonlocal total_samples
                if not batch_snippets:
                    return
                encoded = tokenizer.encode_batch(batch_snippets)
                chunk = np.full((len(encoded), seq_len), PAD_TOKEN_ID, dtype=np.int32)
                for i, enc in enumerate(encoded):
                    ids = enc.ids[:seq_len]
                    chunk[i, :len(ids)] = ids
                out_bin.write(chunk.tobytes())
                total_samples += len(encoded)
                batch_snippets.clear()
                tokens_done = total_samples * seq_len
                print(f"  Tokenized {total_samples:,} samples ({tokens_done:,} tokens)...", flush=True)

            consecutive_blanks = 0
            for line in text_in:
                if line == "\n":
                    consecutive_blanks += 1
                    if consecutive_blanks >= 2 and block_lines:
                        snippet = "".join(block_lines).strip()
                        if len(snippet) >= 30:
                            batch_snippets.append(snippet)
                        block_lines.clear()

                        if len(batch_snippets) >= batch_limit:
                            flush_batch()
                else:
                    if consecutive_blanks == 1:
                        block_lines.append("\n")
                    consecutive_blanks = 0
                    block_lines.append(line)

            if block_lines:
                snippet = "".join(block_lines).strip()
                if len(snippet) >= 30:
                    batch_snippets.append(snippet)
                block_lines.clear()

            flush_batch()

        with open(meta_path, "w") as mf:
            mf.write(str(total_samples))

        elapsed = time.time() - start_t
        print(f"==================================================================")
        print(f"  SUCCESS: Tokenized {total_samples:,} samples ({total_samples * seq_len:,} tokens) in {elapsed/60:.1f}m!")
        print(f"  Binary file: {bin_path} ({bin_path.stat().st_size / 1e9:.2f} GB)")
        print(f"==================================================================")
    else:
        num_samples = int(meta_path.read_text().strip())
        print(f"Binary dataset ready: {bin_path} ({num_samples:,} samples = {num_samples * seq_len:,} tokens).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare 1.7B Token Master Dataset for Ratio Study")
    parser.add_argument("--tokens", type=int, default=1_700_000_000, help="Target token count (default: 1.7B)")
    args = parser.parse_args()
    prepare_dataset(target_tokens=args.tokens)
