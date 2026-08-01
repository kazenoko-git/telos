"""Script to prepare initial Python corpus for tokenization and model training."""

import argparse
from telos.data.prepare import prepare_synthetic_corpus


def main():
    parser = argparse.ArgumentParser(description="Prepare Python code corpus")
    parser.add_argument("--output", type=str, default="data/python_corpus.txt", help="Output text file path")
    args = parser.parse_args()

    print(f"Preparing dataset corpus at {args.output}...")
    prepare_synthetic_corpus(args.output)
    print("Done!")


if __name__ == "__main__":
    main()
