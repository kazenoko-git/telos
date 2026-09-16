"""
Official OpenAI HumanEval (164 Tasks) Head-to-Head Evaluation
100M AR (5B tokens) vs 100M COROSred Unified (5B tokens)
"""

import os
import sys
import json
import time
import urllib.request
import gzip
import torch
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from telos.eval.runner import load_model_from_checkpoint, clean_functional_completion, bootstrap_confidence_interval
from telos.eval.syntax import check_ast_validity
from telos.eval.executor import execute_code_sandboxed, ExecutionResult
from telos.data.tokenizer import load_tokenizer

HUMANEVAL_URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"

def fetch_humaneval():
    print("Fetching OpenAI HumanEval dataset (164 tasks)...")
    req = urllib.request.urlopen(HUMANEVAL_URL)
    with gzip.GzipFile(fileobj=req) as f:
        tasks = [json.loads(line) for line in f]
    print(f"Successfully fetched {len(tasks)} HumanEval tasks.")
    return tasks

def generate_greedy_completion_fast(
    model,
    tokenizer,
    backend: str,
    device,
    prompt: str,
    max_new_tokens: int = 256,
    stop_tokens: list | None = None
) -> str:
    p_ids = tokenizer.encode(prompt).ids
    max_seq_len = 512
    if hasattr(model, "config") and hasattr(model.config, "max_seq_len"):
        max_seq_len = model.config.max_seq_len

    if len(p_ids) >= max_seq_len - 16:
        p_ids = p_ids[-(max_seq_len - 16):]

    curr_ids = list(p_ids)
    stop_set = set(stop_tokens or [0, 3])
    stop_words = ["\ndef ", "\nclass ", "\nif __name__"]

    actual_max_new = min(max_new_tokens, max_seq_len - len(curr_ids))

    for _ in range(actual_max_new):
        if len(curr_ids) >= max_seq_len:
            break

        if backend == "mlx":
            import mlx.core as mx
            x = mx.array([curr_ids], dtype=mx.int32)
            logits = model(x)
            next_tok = int(np.argmax(np.array(logits[0, -1].astype(mx.float32))))
        else:
            x = torch.tensor([curr_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                logits = model(x)
            next_tok = int(torch.argmax(logits[0, -1]).item())

        if next_tok in stop_set:
            break
        curr_ids.append(next_tok)

        cur_text = tokenizer.decode(curr_ids[len(p_ids):])
        if any(sw in cur_text for sw in stop_words):
            break

    continuation_ids = curr_ids[len(p_ids):]
    return tokenizer.decode(continuation_ids)

def evaluate_model_humaneval(model, tokenizer, backend: str, tasks: list, model_name: str, max_new_tokens: int = 256, timeout_seconds: float = 3.0):
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")) if backend == "pytorch" else None
    if backend == "pytorch":
        model = model.to(device)
        model.eval()

    print("\n" + "=" * 85)
    print(f"  HUMANEVAL 164 BENCHMARK: {model_name.upper()}")
    print(f"  Backend: {backend} (Device: {device}) | Max Tokens: {max_new_tokens} | Timeout: {timeout_seconds}s | Temp: 0.0")
    print("=" * 85)

    results = []
    passed_count = 0
    ast_valid_count = 0
    outcome_counts = {k.value: 0 for k in ExecutionResult}
    start_time = time.time()

    for idx, task in enumerate(tasks, 1):
        task_id = task["task_id"]
        prompt = task["prompt"]
        entry_point = task["entry_point"]
        test_code = task["test"]
        test_harness = f"{test_code}\n\ncheck({entry_point})\n"

        raw_comp = generate_greedy_completion_fast(
            model=model,
            tokenizer=tokenizer,
            backend=backend,
            device=device,
            prompt=prompt,
            max_new_tokens=max_new_tokens
        )
        completion = clean_functional_completion(prompt, raw_comp)
        full_code = prompt + completion

        is_ast_valid, ast_err = check_ast_validity(full_code)
        if is_ast_valid:
            ast_valid_count += 1

        outcome, details = execute_code_sandboxed(
            code=full_code,
            test_harness=test_harness,
            timeout_seconds=timeout_seconds
        )
        outcome_counts[outcome.value] += 1
        is_passed = (outcome == ExecutionResult.PASSED)
        if is_passed:
            passed_count += 1

        results.append({
            "task_id": task_id,
            "outcome": outcome.value,
            "ast_valid": is_ast_valid,
            "completion": completion,
            "details": details
        })

        if idx % 10 == 0 or idx == len(tasks) or is_passed:
            pass_str = f"PASSED [✓]" if is_passed else outcome.value
            print(f"  [{idx:3d}/{len(tasks)}] {task_id:<15} | Status: {pass_str:<18} | Pass@1: {passed_count/idx*100:5.2f}% ({passed_count}/{idx})")

    elapsed = time.time() - start_time
    total = len(tasks)
    pass1_pct = (passed_count / total) * 100.0
    ast_pct = (ast_valid_count / total) * 100.0
    ci_pass1 = bootstrap_confidence_interval([1.0 if r["outcome"] == ExecutionResult.PASSED.value else 0.0 for r in results])

    print("\n" + "-" * 85)
    print(f"  FINAL RESULTS FOR {model_name}:")
    print(f"  Pass@1:          {pass1_pct:.2f}% ({passed_count}/{total}) [95% CI: {ci_pass1[0]}% - {ci_pass1[1]}%]")
    print(f"  AST Validity:    {ast_pct:.2f}% ({ast_valid_count}/{total})")
    print(f"  Outcomes:        {outcome_counts}")
    print(f"  Time Elapsed:    {elapsed:.1f}s")
    print("=" * 85 + "\n")

    return {
        "model_name": model_name,
        "total_tasks": total,
        "passed_count": passed_count,
        "pass_at_1_pct": round(pass1_pct, 2),
        "pass_at_1_95ci": ci_pass1,
        "ast_validity_pct": round(ast_pct, 2),
        "execution_outcomes": outcome_counts,
        "elapsed_seconds": round(elapsed, 1),
        "task_results": results
    }

def main():
    tasks = fetch_humaneval()
    tokenizer_path = PROJECT_ROOT / "configs" / "tokenizer_mac.json"
    if not tokenizer_path.exists():
        tokenizer_path = PROJECT_ROOT / "configs" / "shared" / "tokenizer_0.json"
    tokenizer = load_tokenizer(str(tokenizer_path))

    models_to_eval = [
        ("100M COROSred Unified (5B)", PROJECT_ROOT / "checkpoints/corosred/unified/100m_5b/checkpoint_final.pt"),
        ("100M AR (5B)", PROJECT_ROOT / "checkpoints/ar/100m_5b/checkpoint_final.pt")
    ]

    all_reports = {}
    for model_name, ckpt_path in models_to_eval:
        if not ckpt_path.exists():
            print(f"Skipping {model_name}, checkpoint not found at {ckpt_path}")
            continue

        print(f"\nLoading model: {model_name} from {ckpt_path}")
        model, backend, vocab_size = load_model_from_checkpoint(ckpt_path)
        report = evaluate_model_humaneval(model, tokenizer, backend, tasks, model_name)
        all_reports[model_name] = report

    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    report_file = logs_dir / f"humaneval_164_100m_report_{int(time.time())}.json"
    with open(report_file, "w") as f:
        json.dump(all_reports, f, indent=2)
    print(f"\nSaved full HumanEval evaluation report to {report_file}")

if __name__ == "__main__":
    main()
