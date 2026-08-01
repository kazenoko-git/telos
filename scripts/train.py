"""main training script for télos MDLM."""

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
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    tokenizer_path = "configs/tokenizer.json"
    tokenizer = load_tokenizer(tokenizer_path)

    corpus_path = "data/python_corpus.txt"
    try:
        with open(corpus_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
    except FileNotFoundError:
        print("Corpus file not found! Please run python scripts/prepare_data.py first.")
        return

    snippets = [s for s in raw_text.split("\n\n") if len(s.strip()) > 0]
    sequences = [tokenizer.encode(s).ids for s in snippets]

    seq_len = cfg["model"].get("seq_len", 256)
    batch_size = cfg["training"].get("batch_size", 32)
    device = cfg["training"].get("device", "auto")

    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    train_loader = create_dataloader(sequences, batch_size=batch_size, max_seq_len=seq_len, shuffle=True)

    model_cfg = TelosConfig(**cfg["model"])
    model = TelosTransformer(model_cfg)

    trainer = TelosTrainer(model=model, train_loader=train_loader, config=cfg, device=device)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
