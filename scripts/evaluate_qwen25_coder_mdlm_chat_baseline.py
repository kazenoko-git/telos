#!/usr/bin/env python3
"""Evaluate standalone Native MDLM baseline with documented chat template & decoding settings.

Model: dllm-hub/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1
Documented Native Settings:
- Tokenizer chat template:
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": prompt}
    ]
- Generation settings:
    max_new_tokens: 256
    steps: 256
    block_size: 64
    temperature: 0.0
    remasking: "low_confidence"

Outputs:
- Exact formatted inputs (string and token IDs).
- Exact generated raw and cleaned outputs.
- Unpacked boolean AST checks on assembled executable code.
- Sandboxed unit test execution.
- Complete per-task records saved to research/qwen25_coder_mdlm_chat_baseline_2026-10-19/.
"""
from __future__ import annotations

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from tokenizers import Tokenizer
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from telos.eval.runner import (
    clean_functional_completion,
    execute_code_sandboxed,
    ExecutionResult,
)
from telos.eval.syntax import check_ast_validity
from scripts.qwen25_coder_model import (
    load_qwen25_coder_model,
    get_diff_weights_path,
    get_ar_weights_path,
)
from scripts.validate_qwen25_coder_upstream_parity import native_generate


def main():
    print("=" * 80)
    print("STANDALONE MDLM BASELINE EVALUATION (CHAT TEMPLATE & NATIVE SETTINGS)")
    print("=" * 80)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Running evaluation on {device}...")

    # Load weights paths
    diff_weights = get_diff_weights_path()
    diff_dir = diff_weights.parent
    ar_weights = get_ar_weights_path()

    # Load tokenizers and set up documented chat template
    tok = AutoTokenizer.from_pretrained(str(diff_dir))
    ar_tok = AutoTokenizer.from_pretrained(str(ar_weights.parent))
    tok.chat_template = ar_tok.chat_template

    mask_id = tok.mask_token_id
    assert mask_id == 151665, f"Expected mask_id=151665, got {mask_id}"
    pad_id = tok.pad_token_id or tok.eos_token_id

    # Load benchmark suite
    dev_path = PROJECT_ROOT / "research/corosred_dev_retained_edits_30tasks_2026-10-18/per_task_results.json"
    with open(dev_path) as f:
        dev_cases = json.load(f)

    with open(PROJECT_ROOT / "evals/benchmarks/mbpp_suite.json") as f:
        mbpp_tests = {t["id"]: t["test_harness"] for t in json.load(f)}

    print(f"Loaded {len(dev_cases)} MBPP benchmark tasks.")

    out_dir = PROJECT_ROOT / "research/qwen25_coder_mdlm_chat_baseline_2026-10-19"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading MDLM model from {diff_weights.name}...")
    model = load_qwen25_coder_model(diff_weights, device=device)

    results: List[Dict[str, Any]] = []

    # Decoding parameters from official dLLM README
    max_new_tokens = 256
    steps = 256
    block_size = 64
    temperature = 0.0

    print(f"\nDecoding settings: max_new_tokens={max_new_tokens}, steps={steps}, block_size={block_size}, temperature={temperature}")
    print("-" * 80)

    for idx, case in enumerate(dev_cases, 1):
        tid = case["task_id"]
        prompt = case["prompt"]
        test_harness = mbpp_tests[tid]

        # 1. Format input using documented chat template
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt},
        ]
        formatted_input_str = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        formatted_input_ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)

        p_tensor = torch.tensor([formatted_input_ids], dtype=torch.long, device=device)
        p_lens = torch.tensor([len(formatted_input_ids)], dtype=torch.long, device=device)

        # 2. Native decoding
        t0 = time.monotonic()
        gen_tensor = native_generate(
            lambda x: model(x, is_causal=False)[0],
            p_tensor,
            p_lens,
            pad_id=pad_id,
            mask_id=mask_id,
            steps=steps,
            max_new_tokens=max_new_tokens,
            block_size=block_size,
            temperature=temperature,
        )
        latency = time.monotonic() - t0

        output_tokens = gen_tensor[0, len(formatted_input_ids):].tolist()
        raw_output_text = tok.decode(output_tokens)

        # 3. Clean and assemble executable code
        clean_code = clean_functional_completion(prompt, raw_output_text)

        # Assembled program for execution
        if (
            clean_code.strip().startswith("def ")
            or clean_code.strip().startswith("import ")
            or clean_code.strip().startswith("from ")
        ):
            prog = clean_code
        else:
            prog = prompt + ("\n" if not prompt.endswith("\n") else "") + clean_code

        # 4. AST validity check
        is_ast_valid, ast_error = check_ast_validity(prog)
        assert isinstance(is_ast_valid, bool), "AST check must return a boolean"

        # 5. Sandboxed execution against unit tests
        outcome, details = execute_code_sandboxed(prog, test_harness, timeout_seconds=3.0)
        passed = (outcome == ExecutionResult.PASSED)

        results.append({
            "task_id": tid,
            "original_prompt": prompt,
            "chat_messages": messages,
            "formatted_input_string": formatted_input_str,
            "formatted_input_token_ids": formatted_input_ids,
            "raw_output_tokens": output_tokens,
            "raw_output_text": raw_output_text,
            "clean_code": clean_code,
            "assembled_program": prog,
            "ast_valid": is_ast_valid,
            "ast_error": ast_error,
            "passed": passed,
            "outcome": outcome.name,
            "details": details,
            "latency_seconds": latency,
        })

        status_str = "PASS" if passed else f"FAIL ({outcome.name})"
        print(f"[{idx:02d}/30] {tid}: MDLM Chat {status_str} in {latency:.2f}s (AST: {is_ast_valid})")

    # Summary metrics
    pass_count = sum(1 for r in results if r["passed"])
    ast_count = sum(1 for r in results if r["ast_valid"])
    mean_latency = sum(r["latency_seconds"] for r in results) / len(results)

    print("\n" + "=" * 60)
    print("STANDALONE MDLM CHAT BASELINE SUMMARY")
    print("=" * 60)
    print(f"Total Tasks:          {len(results)}")
    print(f"Pass@1:               {pass_count}/{len(results)} ({pass_count / len(results) * 100:.1f}%)")
    print(f"AST Validity:         {ast_count}/{len(results)} ({ast_count / len(results) * 100:.1f}%)")
    print(f"Mean Latency:         {mean_latency:.2f}s per task")
    print(f"Passing Tasks:        {[r['task_id'] for r in results if r['passed']]}")

    # Save exact formatted inputs and outputs
    out_records_path = out_dir / "exact_formatted_inputs_and_outputs.json"
    out_records_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved exact formatted records to {out_records_path}")

    # Save summary
    summary = {
        "model": "dllm-hub/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1",
        "benchmark": "MBPP (30 Tasks)",
        "timestamp": int(time.time()),
        "decoding_settings": {
            "apply_chat_template": True,
            "max_new_tokens": max_new_tokens,
            "steps": steps,
            "block_size": block_size,
            "temperature": temperature,
            "remasking": "low_confidence",
        },
        "metrics": {
            "pass_at_1_pct": round(pass_count / len(results) * 100, 1),
            "pass_at_1_count": pass_count,
            "ast_validity_pct": round(ast_count / len(results) * 100, 1),
            "ast_validity_count": ast_count,
            "mean_latency_seconds": round(mean_latency, 2),
            "passing_tasks": [r["task_id"] for r in results if r["passed"]],
        },
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
