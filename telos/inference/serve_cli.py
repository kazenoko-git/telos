"""Command-line interface for serving native Télos checkpoints over HTTP."""

import argparse
from pathlib import Path

from .chat import ChatTemplateRenderer
from .generation import TextGenerator
from .server import create_app


def _default_model_id(checkpoint: str) -> str:
    path = Path(checkpoint)
    return path.parent.name if path.is_file() else path.name


def main():
    parser = argparse.ArgumentParser(
        prog="telos serve",
        description="Serve a native causal checkpoint behind an OpenAI-compatible HTTP/SSE endpoint.",
    )
    parser.add_argument("--checkpoint", required=True, help="Native checkpoint file or directory")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="TCP port to listen on (default: 8000)")
    parser.add_argument("--model-name", help="Model ID advertised by /v1/models (default: checkpoint directory name)")
    parser.add_argument("--tokenizer", type=Path, help="Override the checkpoint tokenizer JSON")
    parser.add_argument("--config", type=Path, help="Override checkpoint architecture metadata (JSON or YAML)")
    parser.add_argument("--device", default="auto", help="auto, mlx, cpu, cuda, or cuda:N")
    parser.add_argument("--context-length", type=int, help="Limit the context to at most this many tokens")
    parser.add_argument("--memory-budget-mib", type=int, help="Maximum RAM budget for native Qwen3-MoE")
    parser.add_argument("--cache", choices=("auto", "dynamic", "static", "off"), default="auto")
    parser.add_argument("--prefix-cache-entries", type=int, default=4,
                        help="Number of exact prompt KV prefixes to retain across requests (0 disables)")
    parser.add_argument("--chat-template", help="Override Jinja2 chat template string or path to a template file")
    parser.add_argument("--api-key", help="Require Authorization: Bearer <key> on /v1/* endpoints")
    parser.add_argument("--warmup", action="store_true", help="Run a short 4-token warmup before accepting traffic")
    parser.add_argument("--log-level", default="info", choices=("critical", "error", "warning", "info", "debug"))
    args = parser.parse_args()

    try:
        try:
            import uvicorn
        except ImportError as exc:
            raise ImportError(
                "telos serve requires uvicorn and fastapi. "
                "Install them via `pip install 'telos-ml[serve]'`."
            ) from exc

        config = None
        if args.config is not None:
            import yaml

            try:
                config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
            except yaml.YAMLError as exc:
                raise ValueError(f"Invalid model config: {args.config}") from exc
            if not isinstance(config, dict):
                raise ValueError("--config must contain a mapping.")

        generator = TextGenerator.from_checkpoint(
            args.checkpoint,
            tokenizer=args.tokenizer,
            config=config,
            device=args.device,
            context_length=args.context_length,
            memory_budget_mib=args.memory_budget_mib,
            prefix_cache_entries=args.prefix_cache_entries,
        )
        if args.warmup:
            generator.generate("Hello", max_new_tokens=4, cache=args.cache)

        model_id = args.model_name or _default_model_id(args.checkpoint)
        renderer = ChatTemplateRenderer.from_checkpoint_dir(
            generator.checkpoint_dir,
            override=args.chat_template,
        )
        app = create_app(
            generator,
            model_id=model_id,
            chat_renderer=renderer,
            default_cache=args.cache,
            api_key=args.api_key,
        )
        uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(2, f"telos serve: {exc}\n")
