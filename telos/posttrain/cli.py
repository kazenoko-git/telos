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
    parser.add_argument("--finetune", choices=["full", "lora", "qlora"], default="full")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-targets", default=None, help="Comma-separated projection suffixes; default all transformer projections")
    parser.add_argument("--precision", choices=["auto", "float32", "bf16", "fp16"], default="auto")
    parser.add_argument("--compile", dest="compile_step", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--padding", choices=["auto", "batch", "fixed"], default="auto")
    parser.add_argument("--cache-reference", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reference-cache-size", type=int, default=128, help="Maximum cached DPO reference batches (MLX uses an equivalent per-response budget)")
    args = parser.parse_args()
    posttrain(
        args.checkpoint, args.data, args.output,
        method=args.method, tokenizer_path=args.tokenizer, hardware=args.hardware, devices=args.devices,
        max_steps=args.max_steps, batch_size=args.batch_size,
        learning_rate=args.learning_rate, beta=args.beta,
        max_length=args.max_length, save_every=args.save_every, seed=args.seed, resume=args.resume,
        finetune=args.finetune, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, lora_targets=args.lora_targets, precision=args.precision,
        compile_step=args.compile_step, padding=args.padding, cache_reference=args.cache_reference,
        reference_cache_size=args.reference_cache_size,
    )


if __name__ == "__main__":
    main()
