"""Overnight Mac M5 Pro Training Script for télos MDLM.

Runs 7.5 hours of continuous native PyTorch training on Apple Silicon M5 Pro (MPS).
Automates dataset preparation, 44-core parallel tokenization, memory mapping,
and 15,000 steps of model training with periodic checkpointing.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import yaml
import torch
import numpy as np
from telos.model.transformer import TelosTransformer, TelosConfig
from telos.data.tokenizer import load_tokenizer, train_bpe_tokenizer, PAD_TOKEN_ID
from telos.data.dataset import create_dataloader
from telos.training.trainer import TelosTrainer
from telos.data.prepare import prepare_online_corpus


def main():
    config_path = "configs/phase_b_mac.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    corpus_path = "data/python_corpus_mac.txt"
    bin_path = Path(corpus_path).with_suffix(".bin")
    meta_path = Path(corpus_path).with_suffix(".meta")
    tokenizer_path = "configs/tokenizer_mac.json"

    seq_len = cfg["model"].get("seq_len", 512)
    batch_size = cfg["training"].get("batch_size", 32)
    target_tokens = cfg["data"].get("corpus_size_tokens", 500_000_000)
    vocab_size = cfg["model"].get("vocab_size", 8192)

    # Step 1: Prepare raw Python corpus if missing
    if not Path(corpus_path).exists():
        print(f"Step 1: Preparing Python corpus ({target_tokens:,} tokens) -> {corpus_path}")
        prepare_online_corpus(
            output_path=corpus_path,
            target_tokens=target_tokens,
            dataset_name="codeparrot/codeparrot-clean",
            raw_mode=True,
            fast_mode=False
        )

    # Step 2: Train BPE tokenizer if missing
    if not Path(tokenizer_path).exists():
        print(f"Step 2: Training {vocab_size}-vocab BPE Tokenizer -> {tokenizer_path}")
        train_bpe_tokenizer([corpus_path], vocab_size=vocab_size, save_path=tokenizer_path)

    tokenizer = load_tokenizer(tokenizer_path)

    # Step 3: Stream and tokenize corpus into binary int32 array if missing
    if not bin_path.exists():
        print(f"Step 3: Streaming {corpus_path} -> {bin_path}...")
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
                            print(f"  Tokenized {total_samples:,} samples...")
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

        print(f"Tokenized {total_samples:,} samples -> {bin_path}")

    # Step 4: Load sample count from metadata
    with open(meta_path, "r") as mf:
        num_samples = int(mf.read().strip())

    print(f"Step 4: Memory-mapping {bin_path} ({num_samples:,} samples) in 0.00s...")
    arr = np.memmap(bin_path, dtype=np.int32, mode="r", shape=(num_samples, seq_len))

    # Step 5: Launch PyTorch training on M5 Pro MPS GPU
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Step 5: Launching overnight training on {device.upper()} GPU for 15,000 steps...")

    train_loader = create_dataloader(
        arr,
        batch_size=batch_size,
        max_seq_len=seq_len,
        shuffle=True,
        num_workers=4
    )

    model_cfg = TelosConfig(**cfg["model"])
    model = TelosTransformer(model_cfg)

    trainer = TelosTrainer(
        model=model,
        train_loader=train_loader,
        config=cfg,
        device=device
    )

    print("\n" + "="*60)
    print("  télos OVERNIGHT TRAINING LAUNCHED SUCCESSFULLY!")
    print(f"  Model: 85.4M params (12 layers, d=768, GQA 4:1)")
    print(f"  Hardware: Apple Silicon M5 Pro ({device.upper()})")
    print(f"  Target: 15,000 steps (~7.5 hours continuous training)")
    print(f"  Checkpoints: checkpoints/phase_b_mac/ (saved every 10 min)")
    print("="*60 + "\n")

    trainer.train()


if __name__ == "__main__":
    main()
