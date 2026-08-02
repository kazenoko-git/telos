import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import gc
import yaml
import torch
import numpy as np
from telos.model.transformer import TelosTransformer, TelosConfig
from telos.data.tokenizer import load_tokenizer, PAD_TOKEN_ID
from telos.data.dataset import create_dataloader
from telos.training.trainer import TelosTrainer


def main():
    parser = argparse.ArgumentParser(description="Train télos MDLM model")
    parser.add_argument("--config", type=str, default="configs/phase_a.yaml", help="Path to config YAML")
    parser.add_argument("--resume", type=str, default=None, help="Optional checkpoint path to resume from")
    parser.add_argument("--device", type=str, default=None, help="Device target ('auto', 'tpu', 'cuda', 'mps', 'cpu')")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    tokenizer_path = "configs/tokenizer.json"
    if not Path(tokenizer_path).exists():
        if Path("configs/tokenizer_mac.json").exists():
            tokenizer_path = "configs/tokenizer_mac.json"
        elif Path("configs/tokenizer_0.json").exists():
            tokenizer_path = "configs/tokenizer_0.json"
    tokenizer = load_tokenizer(tokenizer_path)

    corpus_path = Path("data/python_corpus.txt")
    if Path("data/python_corpus_mac.bin").exists():
        bin_path = Path("data/python_corpus_mac.bin")
    elif Path("data/python_corpus.bin").exists():
        bin_path = Path("data/python_corpus.bin")
    else:
        bin_path = Path("data/python_corpus.bin")

    bin_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = bin_path.with_suffix(".meta")

    seq_len = cfg["model"].get("seq_len", 512)
    batch_size = cfg["training"].get("batch_size", 32)
    device = cfg["training"].get("device", "auto")

    if not bin_path.exists():
        if not corpus_path.exists():
            print(f"Data file {bin_path} not found. Running online dataset downloader...")
            from telos.data.prepare import prepare_online_corpus
            target_tokens = cfg.get("data", {}).get("corpus_size_tokens", 500_000_000)
            prepare_online_corpus(output_path=str(corpus_path), target_tokens=target_tokens)
        print(f"Streaming {corpus_path} -> {bin_path} with <500MB RAM...")

        # Stream text line-by-line, accumulate blocks between blank lines,
        # batch-tokenize in chunks of 32k, and append int32 rows to disk.
        batch_snippets = []
        total_samples = 0
        batch_limit = 32000
        block_lines = []

        with open(bin_path, "wb") as out_bin, \
             open(corpus_path, "r", encoding="utf-8") as text_in:

            def flush_batch():
                """Tokenize current batch and write to disk."""
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
                    # File boundary = 2+ consecutive blank lines
                    if consecutive_blanks >= 2 and block_lines:
                        snippet = "".join(block_lines).strip()
                        if len(snippet) >= 30:
                            batch_snippets.append(snippet)
                        block_lines.clear()

                        if len(batch_snippets) >= batch_limit:
                            flush_batch()
                            if total_samples % 500000 < batch_limit:
                                print(f"  Tokenized {total_samples:,} samples...")
                else:
                    # If we had exactly 1 blank line, it's an intra-file blank
                    # so preserve it in the block
                    if consecutive_blanks == 1:
                        block_lines.append("\n")
                    consecutive_blanks = 0
                    block_lines.append(line)

            # Handle last block
            if block_lines:
                snippet = "".join(block_lines).strip()
                if len(snippet) >= 30:
                    batch_snippets.append(snippet)
                block_lines.clear()

            flush_batch()

        # Save metadata (sample count)
        with open(meta_path, "w") as mf:
            mf.write(str(total_samples))

        print(f"Tokenized {total_samples:,} samples -> {bin_path} "
              f"({bin_path.stat().st_size / (1024**3):.2f} GB)")

    # Read sample count from metadata or compute directly from binary file size
    if meta_path.exists():
        with open(meta_path, "r") as mf:
            num_samples = int(mf.read().strip())
    else:
        num_samples = bin_path.stat().st_size // (seq_len * 4)

    # Memory-map binary file (instant load, zero RAM)
    print(f"Memory-mapping {bin_path} ({num_samples:,} samples) in 0.00s...")
    arr = np.memmap(bin_path, dtype=np.int32, mode="r", shape=(num_samples, seq_len))

    if args.device:
        device = args.device
    elif device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    train_loader = create_dataloader(arr, batch_size=batch_size, max_seq_len=seq_len, shuffle=True)

    model_cfg = TelosConfig(**cfg["model"])
    model = TelosTransformer(model_cfg)

    trainer = TelosTrainer(model=model, train_loader=train_loader, config=cfg, device=device)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
