import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import yaml
from mdiff.data.prepare import prepare_online_corpus


def main():
    parser = argparse.ArgumentParser(description="Prepare Python code corpus from online datasets")
    parser.add_argument("--config", type=str, default="configs/phase_a.yaml", help="Path to config YAML")
    parser.add_argument("--output", type=str, default="data/python_corpus.txt", help="Output text file path")
    parser.add_argument("--dataset", type=str, default="codeparrot/codeparrot-clean", help="HuggingFace dataset to stream")
    parser.add_argument("--raw", action="store_true", help="Skip AST parsing, write raw Python code (max throughput)")
    parser.add_argument("--fast", action="store_true", help="Download full dataset in parallel first (faster iteration)")
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
        dataset_name=args.dataset,
        raw_mode=args.raw,
        fast_mode=args.fast,
    )


if __name__ == "__main__":
    main()
