"""Script to stream online Python dataset and prepare training corpus."""

import argparse
import yaml
from telos.data.prepare import prepare_online_corpus


def main():
    parser = argparse.ArgumentParser(description="Prepare Python code corpus from online datasets")
    parser.add_argument("--config", type=str, default="configs/phase_a.yaml", help="Path to config YAML")
    parser.add_argument("--output", type=str, default="data/python_corpus.txt", help="Output text file path")
    parser.add_argument("--dataset", type=str, default="codeparrot/codeparrot-clean", help="HuggingFace dataset to stream")
    args = parser.parse_args()

    # Load target token count from config YAML
    target_tokens = 30_000_000
    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
            target_tokens = cfg.get("data", {}).get("corpus_size_tokens", 30_000_000)

    prepare_online_corpus(
        output_path=args.output,
        target_tokens=target_tokens,
        dataset_name=args.dataset
    )


if __name__ == "__main__":
    main()
