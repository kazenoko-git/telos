import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import yaml
import torch
import numpy as np
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
    bin_path = Path(corpus_path).with_suffix(".bin")

    seq_len = cfg["model"].get("seq_len", 256)
    batch_size = cfg["training"].get("batch_size", 32)
    device = cfg["training"].get("device", "auto")
    from telos.data.tokenizer import PAD_TOKEN_ID

    if not bin_path.exists():
        print(f"Tokenizing {corpus_path} directly into binary disk format with <500MB RAM footprint...")

        batch_size_snippets = 32000
        current_batch = []
        total_samples = 0

        with open(bin_path, "wb") as bin_file, open(corpus_path, "r", encoding="utf-8") as text_file:
            block = []
            for line in text_file:
                if line == "\n\n" or (line == "\n" and len(block) > 10):
                    snippet = "".join(block).strip()
                    if len(snippet) >= 30:
                        current_batch.append(snippet)
                    block = []
                else:
                    block.append(line)

                if len(current_batch) >= batch_size_snippets:
                    # Tokenize batch in parallel across 44 CPU cores
                    encoded = tokenizer.encode_batch(current_batch)
                    arr_batch = np.full((len(encoded), seq_len), PAD_TOKEN_ID, dtype=np.int32)
                    for i, enc in enumerate(encoded):
                        ids = enc.ids[:seq_len]
                        arr_batch[i, :len(ids)] = ids

                    bin_file.write(arr_batch.tobytes())
                    total_samples += len(encoded)
                    current_batch.clear()

            if block:
                snippet = "".join(block).strip()
                if len(snippet) >= 30:
                    current_batch.append(snippet)

            if current_batch:
                encoded = tokenizer.encode_batch(current_batch)
                arr_batch = np.full((len(encoded), seq_len), PAD_TOKEN_ID, dtype=np.int32)
                for i, enc in enumerate(encoded):
                    ids = enc.ids[:seq_len]
                    arr_batch[i, :len(ids)] = ids

                bin_file.write(arr_batch.tobytes())
                total_samples += len(encoded)
                current_batch.clear()

        print(f"Tokenized {total_samples:,} samples into binary disk file {bin_path}!")

    # Memory-map binary file (0.00s instant load, zero RAM footprint)
    file_bytes = bin_path.stat().st_size
    num_samples = file_bytes // (seq_len * 4)
    print(f"Memory-mapping {bin_path} ({num_samples:,} samples, {file_bytes / (1024*1024):.1f} MB) in 0.00s...")

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
