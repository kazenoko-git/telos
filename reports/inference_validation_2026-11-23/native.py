"""Measure trained Telos checkpoints with the public streaming generation path."""

import argparse
import ast
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

import psutil
import torch

from telos.eval.executor import execute_code_sandboxed
from telos.inference import TextGenerator

MiB = 1024 ** 2
TASKS = [
    ("square", "def square(x):\n    \"\"\"Return x squared.\"\"\"\n", "assert square(-3) == 9 and square(0) == 0 and square(7) == 49"),
    ("is_even", "def is_even(n):\n    \"\"\"Return True if n is an even integer.\"\"\"\n", "assert is_even(0) is True and is_even(-4) is True and is_even(3) is False"),
    ("reverse", "def reverse_string(text):\n    \"\"\"Return text with its characters in reverse order.\"\"\"\n", "assert reverse_string('abc') == 'cba' and reverse_string('') == '' and reverse_string('é🙂') == '🙂é'"),
    ("product", "def product(values):\n    \"\"\"Return the product of all values, or 1 when empty.\"\"\"\n", "assert product([]) == 1 and product([2, -3, 4]) == -24 and product([0, 7]) == 0"),
    ("unique", "def unique_in_order(values):\n    \"\"\"Return a list of distinct items, keeping first-appearance order.\"\"\"\n", "assert unique_in_order([]) == [] and unique_in_order([3, 1, 3, 2, 1]) == [3, 1, 2]"),
    ("merge", "def merge_sorted(a, b):\n    \"\"\"Merge two ascending lists into one ascending list, retaining duplicates.\"\"\"\n", "assert merge_sorted([], [2]) == [2] and merge_sorted([1, 3, 3], [2, 3]) == [1, 2, 3, 3]"),
]
PROSE = [
    ("plain_fact", "The capital city of France is"),
    ("instruction_format", "Answer in exactly one word: What is the capital of France?\nAnswer:"),
    ("code_seed", "# A short Python example that reads a CSV file and sums the price column.\nimport csv\n"),
]


def code_grade(prompt, completion, checks):
    source = prompt + completion
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return "syntax_error"
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        return "unexpected_module_shape"
    if len(tree.body[0].body) < 2:
        return "docstring_only"
    outcome, _ = execute_code_sandboxed(source, checks, timeout_seconds=2)
    return outcome.value


def perf(gen, prompt, *, cache="auto", limit=144, **options):
    token_times = []
    emission_times = []
    text = []
    attribute = "_mlx_tokens" if gen.backend == "mlx" else "_torch_tokens"
    initial = getattr(gen, attribute)

    def tracked(*args):
        for token in initial(*args):
            if token not in gen.stop_ids:
                token_times.append(time.perf_counter())
            yield token

    if gen.backend == "mlx":
        import mlx.core as mx

        mx.synchronize()
        mlx_active_before = mx.get_active_memory() / MiB
        mx.reset_peak_memory()
    setattr(gen, attribute, tracked)
    start = time.perf_counter()
    try:
        for part in gen.stream(prompt, cache=cache, max_new_tokens=limit, **options):
            emission_times.append(time.perf_counter())
            text.append(part)
    finally:
        delattr(gen, attribute)
    end = time.perf_counter()
    n = len(token_times)
    result = {"prompt_tokens": len(gen.tokenizer.encode(prompt, add_special_tokens=False).ids),
              "output_tokens": n,
              "ttft_ms": (token_times[0] - start) * 1e3 if n else None,
              "first_text_ms": (emission_times[0] - start) * 1e3 if text else None,
              "decode_tok_s": (n - 1) / (token_times[-1] - token_times[0]) if n > 1 else None,
              "last_32_tok_s": 31 / (token_times[-1] - token_times[-32]) if n >= 32 else None,
              "total_ms": (end - start) * 1e3, "text": "".join(text)}
    if gen.backend == "mlx":
        result.update(mlx_peak_mib=mx.get_peak_memory() / MiB, mlx_active_before_mib=mlx_active_before)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tokenizer", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(6)
    process = psutil.Process()
    baseline = process.memory_info().rss / MiB
    start = time.perf_counter()
    gen = TextGenerator.from_checkpoint(args.checkpoint, tokenizer=args.tokenizer)
    if gen.backend == "mlx":
        import mlx.core as mx

        mx.eval(gen.model.parameters())
    result = {"checkpoint": str(args.checkpoint), "backend": gen.backend,
              "context": gen.max_seq_len, "tokenizer_override": str(args.tokenizer), "model_load_ms": (time.perf_counter() - start) * 1e3,
              "baseline_rss_mib": baseline, "loaded_rss_mib": process.memory_info().rss / MiB}
    result["cold"] = perf(gen, PROSE[-1][1], limit=64)
    result["quality"] = {}
    for name, prompt, checks in TASKS:
        entries = {}
        for label, options in [("greedy", {}), ("seeded", {"temperature": 0.65, "top_p": 0.95, "seed": 42})]:
            text = gen.generate(prompt, max_new_tokens=96, stop=("\ndef ", "\nclass ", "\nif __name__"), **options)
            entries[label] = {"text": text, "outcome": code_grade(prompt, text, checks)}
        result["quality"][name] = entries
    result["prose"] = {name: gen.generate(prompt, max_new_tokens=64) for name, prompt in PROSE}
    result["trials"] = {}
    for name, repeats in [("short", 2), ("long", 16)]:
        suffix = "def merge_sorted(left, right):\n    \"\"\"Merge two sorted lists.\"\"\"\n"
        prompts = [f"# Validation case {index}.\n" + ("# Sort records while preserving original order.\n" * repeats)
                   + suffix for index in range(1, 6)]
        limit = min(144, gen.max_seq_len - max(len(gen.tokenizer.encode(p).ids) for p in prompts))
        assert limit > 15
        result["trials"][name] = [perf(gen, p, limit=limit, temperature=0.65, top_p=0.95, seed=42) for p in prompts]
    result["cache_compare"] = {}
    prompt = "# A utility module.\n" * 10 + "def is_even(n):\n    \"\"\"Return True if n is even.\"\"\"\n"
    for mode in (["off", "dynamic", "static"] if gen.backend == "pytorch" else ["off", "dynamic"]):
        result["cache_compare"][mode] = perf(gen, prompt, cache=mode, limit=72)
    result["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / MiB
    start = time.perf_counter()
    with subprocess.Popen([sys.executable, "-m", "telos", "generate", "--checkpoint", str(args.checkpoint),
                           "--tokenizer", str(args.tokenizer), "--prompt", PROSE[-1][1], "--max-new-tokens", "64", "--stream"],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE) as child:
        first = child.stdout.read(1)
        first_time = time.perf_counter()
        text, error = child.communicate(timeout=60)
    result["cli_cold"] = {"first_byte_ms": (first_time - start) * 1e3 if first else None,
                          "exit_code": child.returncode, "stderr": error.decode(), "text": (first + text).decode()}
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
