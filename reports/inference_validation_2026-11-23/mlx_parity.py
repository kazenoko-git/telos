"""Compare full-prefix and cached MLX scores on identical prefixes."""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from telos.inference import TextGenerator


def parity(generator, prompt, positions):
    ids = generator.tokenizer.encode(prompt, add_special_tokens=False).ids
    cache = None
    errors = []
    first_mismatch = None
    for step in range(positions):
        full_input = mx.array([ids], dtype=mx.int32)
        cached_input = full_input if cache is None else full_input[:, -1:]
        model = generator.model
        uncached = model.logits_from_hidden(model.hidden_states(full_input, is_causal=True)[:, -1])
        cached_hidden, cache = model.hidden_states_with_cache(cached_input, kv_cache=cache, is_causal=True)
        cached = model.logits_from_hidden(cached_hidden[:, -1])
        full, incr = (np.asarray(x.astype(mx.float32)).reshape(-1) for x in (uncached, cached))
        expected = int(full.argmax())
        observed = int(incr.argmax())
        errors.append(float(np.max(np.abs(full - incr))))
        if expected != observed and first_mismatch is None:
            first_mismatch = {"completion_offset": step, "full_id": expected, "cached_id": observed,
                              "full_scores_for_both": [float(full[expected]), float(full[observed])],
                              "cached_scores_for_both": [float(incr[expected]), float(incr[observed])],
                              "tokens": generator.tokenizer.decode([expected, observed])}
        ids.append(expected)
    return {"prompt_tokens": len(ids) - positions, "compared_positions": positions,
            "max_logit_delta": max(errors), "median_per_step_max_logit_delta": float(np.median(errors)),
            "first_greedy_mismatch": first_mismatch}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tokenizer", type=Path, required=True)
    args = parser.parse_args()
    generator = TextGenerator.from_checkpoint(args.checkpoint, tokenizer=args.tokenizer)
    prompts = [
        "# A utility module.\n" * 10 + "def is_even(n):\n    \"\"\"Return True if n is even.\"\"\"\n",
        "def square(x):\n    \"\"\"Return x squared.\"\"\"\n",
        "The capital city of France is",
    ]
    results = {}
    for dtype in (mx.bfloat16, mx.float32):
        generator.model.set_dtype(dtype)
        mx.eval(generator.model.parameters())
        results[str(dtype)] = [parity(generator, prompt, 72) for prompt in prompts]
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
