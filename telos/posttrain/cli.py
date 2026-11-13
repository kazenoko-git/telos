"""Command-line interface for SFT and DPO post-training."""

import argparse

from .trainer import posttrain


def main():
    parser = argparse.ArgumentParser(description="Post-train a causal Télos checkpoint")
    parser.add_argument("--method", choices=["sft", "dpo"], default="sft")
    parser.add_argument("--checkpoint", required=True, help="Native Telos .pt/.safetensors checkpoint or directory")
    parser.add_argument("--data", required=True, help="JSONL: prompt/response for SFT; prompt/chosen/rejected for DPO")
    parser.add_argument("--output", required=True, help="Output directory for post-trained checkpoints")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer JSON path (defaults to checkpoint setting or bundled tokenizer)")
    parser.add_argument("--hardware", "--device", dest="hardware", choices=["auto", "mlx", "cuda", "xla", "cpu"], default="auto",
                        help="Training backend; Apple Silicon uses MLX (not PyTorch MPS)")
    parser.add_argument("--devices", default="auto", help="XLA device count to require, or auto for the full runtime")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO divergence scale")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=250, help="Checkpoint cadence; 0 disables intermediate checkpoints")
    parser.add_argument("--resume", default=None, help="Full .resume.pt or .resume.npz checkpoint; repeat original run options")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    posttrain(
        args.checkpoint, args.data, args.output,
        method=args.method, tokenizer_path=args.tokenizer, hardware=args.hardware, devices=args.devices,
        max_steps=args.max_steps, batch_size=args.batch_size,
        learning_rate=args.learning_rate, beta=args.beta,
        max_length=args.max_length, save_every=args.save_every, seed=args.seed, resume=args.resume,
    )


if __name__ == "__main__":
    main()
