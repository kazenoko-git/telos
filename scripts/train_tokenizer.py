import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import yaml
from telos.data.tokenizer import train_bpe_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train ByteLevel BPE Tokenizer")
    parser.add_argument("--corpus", type=str, default="data/python_corpus.txt", help="Input corpus file path")
    parser.add_argument("--config", type=str, default="configs/phase_a.yaml", help="Path to YAML config file")
    parser.add_argument("--output", type=str, default="configs/tokenizer.json", help="Output tokenizer JSON path")
    args = parser.parse_args()

    vocab_size = 4096
    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
            vocab_size = cfg.get("model", {}).get("vocab_size", 4096)

    print(f"Training BPE Tokenizer (vocab_size={vocab_size}) from {args.corpus}...")
    train_bpe_tokenizer([args.corpus], vocab_size=vocab_size, save_path=args.output)
    print(f"Tokenizer trained and saved to {args.output}")


if __name__ == "__main__":
    main()
