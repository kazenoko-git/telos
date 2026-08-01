import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import yaml
import torch
from telos.model.transformer import TelosTransformer, TelosConfig
from telos.data.tokenizer import load_tokenizer
from telos.eval.sample import run_qualitative_evaluation


def main():
    parser = argparse.ArgumentParser(description="Sample code completions from télos model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_final.pt", help="Path to checkpoint")
    parser.add_argument("--config", type=str, default="configs/phase_a.yaml", help="Path to config file")
    parser.add_argument("--steps", type=int, default=64, help="Number of denoising steps")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Load tokenizer
    tokenizer = load_tokenizer("configs/tokenizer.json")

    # Instantiate model & load checkpoint
    model_cfg = TelosConfig(**cfg["model"])
    model = TelosTransformer(model_cfg)

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    # Run sampling evaluation
    run_qualitative_evaluation(model, tokenizer, num_steps=args.steps, device=device)


if __name__ == "__main__":
    main()
