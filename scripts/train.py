import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import yaml
import torch
from telos.model.transformer import TelosTransformer, TelosConfig
from telos.data.tokenizer import load_tokenizer
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
    tokenizer = load_tokenizer(tokenizer_path)

    corpus_path = "data/python_corpus.txt"
    cache_npy_path = Path(corpus_path).with_suffix(".npy")

    try:
        import gc
        import numpy as np
        from telos.data.tokenizer import PAD_TOKEN_ID

        seq_len = cfg["model"].get("seq_len", 256)
        batch_size = cfg["training"].get("batch_size", 32)
        device = cfg["training"].get("device", "auto")

        if cache_npy_path.exists():
            print(f"Loading pre-tokenized binary dataset from {cache_npy_path} (0.05s instant load)...")
            arr = np.load(cache_npy_path, mmap_mode="r")
        else:
            print(f"Loading and encoding corpus from {corpus_path} into binary memory layout...")

            # Read corpus in streaming blocks to prevent CPython heap RAM spike
            token_sequences = []
            with open(corpus_path, "r", encoding="utf-8") as f:
                block = []
                for line in f:
                    if line == "\n" and block:
                        snippet = "".join(block).strip()
                        if snippet:
                            token_sequences.append(tokenizer.encode(snippet).ids[:seq_len])
                        block = []
                    else:
                        block.append(line)
                if block:
                    snippet = "".join(block).strip()
                    if snippet:
                        token_sequences.append(tokenizer.encode(snippet).ids[:seq_len])

            num_samples = len(token_sequences)
            print(f"Encoded {num_samples:,} function sequences. Converting to 2D NumPy array...")

            # Pack into contiguous 2D int32 array
            arr = np.full((num_samples, seq_len), PAD_TOKEN_ID, dtype=np.int32)
            for i, seq in enumerate(token_sequences):
                arr[i, :len(seq)] = seq

            del token_sequences
            gc.collect()

            print(f"Caching binary tokenized dataset to {cache_npy_path} for instant future loads...")
            np.save(cache_npy_path, arr)

        print(f"NumPy dataset memory footprint: {arr.nbytes / (1024 * 1024):.1f} MB ({len(arr):,} samples)!")

    except FileNotFoundError:
        print("Corpus file not found! Please run python scripts/prepare_data.py first.")
        return

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
