"""Run one small HumanEval slice through Telos's own evaluator."""

import argparse
import json
from pathlib import Path
import time

import torch

from telos.data.tokenizer import load_tokenizer
from telos.eval.adapters.openai_api import OpenAIAPIAdapter
from telos.eval.runner import evaluate_functional, load_model_from_checkpoint


class AuditAdapter(OpenAIAPIAdapter):
    """Bound the external evaluator's 2,048-token minimum for this smoke test."""

    def generate(self, prompt, max_new_tokens=256, **options):
        return super().generate(prompt, max_new_tokens=min(max_new_tokens, 384),
                                request_attempts=1, **options)


def evaluate(model_path, *, api_base=None, tokenizer_path=None):
    start = time.perf_counter()
    tokenizer = load_tokenizer(tokenizer_path) if tokenizer_path else None
    if api_base:
        model = AuditAdapter(model_path, api_base=api_base, api_key="EMPTY", concurrency=1, timeout=45)
        backend = model.backend_name
    else:
        model, backend, _ = load_model_from_checkpoint(model_path)
        torch.set_num_threads(6)
    result = evaluate_functional(model=model, tokenizer=tokenizer, backend=backend, suite="humaneval",
                                 prompt_mode="base", max_tasks=4, max_new_tokens=192, timeout_seconds=2)
    return {"source": model_path, "backend": backend, "wall_seconds": time.perf_counter() - start,
            "evaluation": result}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("output", type=Path)
    parser.add_argument("--api-base")
    parser.add_argument("--tokenizer")
    args = parser.parse_args()
    result = evaluate(args.model, api_base=args.api_base, tokenizer_path=args.tokenizer)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
