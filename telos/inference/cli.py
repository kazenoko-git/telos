"""Command-line interface for native text generation."""

import argparse
import sys
from pathlib import Path

from .generation import TextGenerator


def main():
    parser = argparse.ArgumentParser(prog="telos generate", description="Generate text with a native causal Telos checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Native checkpoint file or directory")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt", help="Prompt text")
    source.add_argument("--prompt-file", type=Path, help="Read the prompt from a UTF-8 file, or - for stdin")
    parser.add_argument("--tokenizer", type=Path, help="Override the checkpoint tokenizer JSON")
    parser.add_argument("--config", type=Path, help="Override checkpoint architecture metadata (JSON or YAML)")
    parser.add_argument("--device", default="auto", help="auto, mlx, cpu, cuda, or cuda:N")
    parser.add_argument("--context-length", type=int, help="Limit the context to at most this many tokens")
    parser.add_argument("--memory-budget-mib", type=int, help="Maximum RAM budget for native MLX MoE")
    parser.add_argument("--window-size", type=int, help="Pin decode MoE experts across W-token windows")
    parser.add_argument("--window-adapter", type=Path, help="Optional distilled window-routing Safetensors adapter")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--stop", action="append", default=[], help="Stop string. Repeat to supply more than one.")
    parser.add_argument("--bos", action=argparse.BooleanOptionalAction, default=None,
                        help="Override BOS insertion (default: on for post-trained checkpoints)")
    parser.add_argument("--stream", action="store_true", help="Flush output as text becomes available")
    parser.add_argument("--cache", choices=("auto", "dynamic", "static", "off"), default="auto")
    args = parser.parse_args()

    try:
        config = None
        if args.config is not None:
            import yaml

            try:
                config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
            except yaml.YAMLError as exc:
                raise ValueError(f"Invalid model config: {args.config}") from exc
            if not isinstance(config, dict):
                raise ValueError("--config must contain a mapping.")
        prompt = args.prompt
        if args.prompt_file is not None:
            prompt = sys.stdin.read() if args.prompt_file == Path("-") else args.prompt_file.read_text(encoding="utf-8")
        generator = TextGenerator.from_checkpoint(
            args.checkpoint, tokenizer=args.tokenizer, config=config, device=args.device,
            context_length=args.context_length, memory_budget_mib=args.memory_budget_mib,
            window_size=args.window_size, window_adapter=args.window_adapter,
        )
        if getattr(generator.model, "weight_storage", None) == "ssd_experts":
            print("Flash-MoE: streaming routed experts from SSD.", file=sys.stderr)
        options = {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "seed": args.seed,
            "stop": args.stop,
            "add_bos": args.bos,
            "cache": args.cache,
        }
        if args.stream:
            for chunk in generator.stream(prompt, **options):
                print(chunk, end="", flush=True)
        else:
            print(generator.generate(prompt, **options), end="")
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(2, f"telos generate: {exc}\n")
