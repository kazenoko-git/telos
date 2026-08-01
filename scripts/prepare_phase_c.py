"""Phase C Multi-Domain Data Preparation Script.

Downloads and prepares a multi-domain corpus for the 1.08B parameter model:
  - Python Code (60%): codeparrot/codeparrot-clean
  - English Technical Text (25%): HuggingFaceFW/fineweb-edu (sample-10BT)
  - Shell Commands (15%): andstor/the_stack_smol (shell subset)

Streams all three domains into a single interleaved text file,
then trains a 32,768-vocab BPE tokenizer on the mixed corpus.

All CPU-bound work — designed to run overnight on Mac M5 Pro.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import yaml
from tqdm import tqdm
from datasets import load_dataset
from telos.data.tokenizer import train_bpe_tokenizer


# Domain configs: (dataset_name, text_key, split, streaming, proportion)
DOMAIN_CONFIGS = {
    "python": {
        "dataset": "codeparrot/codeparrot-clean",
        "text_key": "content",
        "split": "train",
        "proportion": 0.60,
    },
    "english": {
        "dataset": "HuggingFaceFW/fineweb-edu",
        "name": "sample-10BT",
        "text_key": "text",
        "split": "train",
        "proportion": 0.25,
    },
    "shell": {
        "dataset": "bigcode/the-stack-smol",
        "name": "data/shell",
        "text_key": "content",
        "split": "train",
        "proportion": 0.15,
    },
}


def prepare_phase_c_corpus(
    output_path: str = "data/phase_c_corpus.txt",
    target_tokens: int = 60_000_000_000,
):
    """Streams multi-domain data into a single interleaved text file.

    Uses round-robin interleaving: for every 20 samples written,
    12 are Python, 5 are English, 3 are Shell (60/25/15 ratio).
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # ~4 chars per token estimate
    target_chars = target_tokens * 4

    print(f"Phase C Multi-Domain Corpus Preparation")
    print(f"Target: {target_tokens:,} tokens (~{target_chars / 1e9:.1f} GB text)")
    print(f"Output: {output_path}")
    print(f"Domains: Python (60%), English (25%), Shell (15%)")
    print()

    # Load all three domain iterators as streaming datasets
    print("Loading domain iterators (streaming mode)...")
    iterators = {}
    for domain, cfg in DOMAIN_CONFIGS.items():
        ds_kwargs = {"split": cfg["split"], "streaming": True}
        if "name" in cfg:
            ds_kwargs["name"] = cfg["name"]
        ds = load_dataset(cfg["dataset"], **ds_kwargs)
        iterators[domain] = iter(ds)
        print(f"  ✓ {domain}: {cfg['dataset']}")

    # Round-robin schedule: 12 python, 5 english, 3 shell per 20-sample cycle
    schedule = (
        ["python"] * 12 +
        ["english"] * 5 +
        ["shell"] * 3
    )

    written_chars = 0
    item_count = 0
    exhausted = set()
    buffer = []
    buffer_chars = 0

    pbar = tqdm(total=target_chars, unit="char", unit_scale=True, desc="Streaming")

    with open(output_file, "w", encoding="utf-8") as f:
        cycle_idx = 0

        while written_chars < target_chars:
            domain = schedule[cycle_idx % len(schedule)]
            cycle_idx += 1

            # Skip exhausted domains
            if domain in exhausted:
                if len(exhausted) == len(iterators):
                    print(f"\nAll domains exhausted at {written_chars / 1e9:.2f} GB")
                    break
                continue

            # Get next sample from this domain
            try:
                sample = next(iterators[domain])
            except StopIteration:
                exhausted.add(domain)
                print(f"\n  ⚠ {domain} domain exhausted at {written_chars / 1e9:.2f} GB")
                continue

            text_key = DOMAIN_CONFIGS[domain]["text_key"]
            text = sample.get(text_key, "")
            if not text or len(text) < 30:
                continue

            block = text.strip() + "\n\n"
            block_len = len(block)

            buffer.append(block)
            buffer_chars += block_len
            written_chars += block_len
            item_count += 1
            pbar.update(block_len)

            # Flush buffer to disk in 512KB chunks for fast sequential writes
            if buffer_chars >= 524288:
                f.write("".join(buffer))
                buffer.clear()
                buffer_chars = 0

        # Flush remaining buffer
        if buffer:
            f.write("".join(buffer))
            buffer.clear()

    pbar.close()
    approx_tokens = written_chars // 4
    print(f"\nPhase C corpus complete!")
    print(f"  Items: {item_count:,}")
    print(f"  Size: {written_chars / 1e9:.2f} GB ({approx_tokens:,} tokens)")
    print(f"  Saved: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Prepare Phase C multi-domain corpus")
    parser.add_argument("--config", type=str, default="configs/phase_c.yaml",
                        help="Path to Phase C config YAML")
    parser.add_argument("--output", type=str, default="data/phase_c_corpus.txt",
                        help="Output text file path")
    parser.add_argument("--tokenizer-output", type=str, default="configs/tokenizer_32k.json",
                        help="Output path for trained 32K tokenizer")
    parser.add_argument("--skip-tokenizer", action="store_true",
                        help="Skip tokenizer training (if already trained)")
    args = parser.parse_args()

    # Load target token count from config
    target_tokens = 60_000_000_000
    vocab_size = 32768
    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
            target_tokens = cfg.get("data", {}).get("corpus_size_tokens", 60_000_000_000)
            vocab_size = cfg.get("model", {}).get("vocab_size", 32768)

    # Step 1: Download and stream multi-domain corpus to disk
    corpus_path = prepare_phase_c_corpus(
        output_path=args.output,
        target_tokens=target_tokens,
    )

    # Step 2: Train 32K BPE tokenizer on the mixed corpus
    if not args.skip_tokenizer:
        print(f"\nTraining 32K BPE Tokenizer (vocab_size={vocab_size})...")
        print(f"  Corpus: {corpus_path}")
        print(f"  Output: {args.tokenizer_output}")
        train_bpe_tokenizer(
            [corpus_path],
            vocab_size=vocab_size,
            save_path=args.tokenizer_output,
        )
        print(f"  ✓ Tokenizer saved to {args.tokenizer_output}")
    else:
        print("Skipping tokenizer training (--skip-tokenizer flag set)")


if __name__ == "__main__":
    main()
