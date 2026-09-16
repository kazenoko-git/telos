"""
Iterative Mask-Infilling Decoding Evaluation for 100M COROSred Unified
Evaluates 100M COROSred using MDLMSampler (32-step iterative confidence unmasking)
against 100M AR on HumanEval (164 tasks) and Private Unseen Suite (512 tasks).
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
from telos.diffusion.sampler import MDLMSampler

HUMANEVAL_URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"

def fetch_humaneval():
    req = urllib.request.urlopen(HUMANEVAL_URL)
    with gzip.GzipFile(fileobj=req) as f:
        return [json.loads(line) for line in f]

def generate_iterative_infill_completion(
    model,
    tokenizer,
    backend: str,
    device,
    prompt: str,
    target_gen_len: int = 128,
    num_steps: int = 32,
    temperature: float = 0.1,
    mask_token_id: int = 1
) -> str:
    """
    Generates completion using MDLMSampler confidence-based iterative unmasking.
    Appends [MASK] tokens to prompt and progressively unmasks over num_steps.
    """
    p_ids = tokenizer.encode(prompt).ids
    max_seq_len = 512
    if hasattr(model, "config") and hasattr(model.config, "max_seq_len"):
        max_seq_len = model.config.max_seq_len

    if len(p_ids) >= max_seq_len - 32:
        p_ids = p_ids[-(max_seq_len - 32):]

    actual_gen_len = min(target_gen_len, max_seq_len - len(p_ids))
    total_len = len(p_ids) + actual_gen_len

    if backend == "mlx":
        import mlx.core as mx
        from telos.diffusion.sampler import MLXMDLMSampler
        sampler = MLXMDLMSampler(model=model, mask_token_id=mask_token_id, num_steps=num_steps, temperature=temperature)
        sample_ids = sampler.sample(seq_len=total_len, prompt_ids=p_ids)
        mx.eval(sample_ids)
        full_ids = np.array(sample_ids[0]).tolist()
    else:
        sampler = MDLMSampler(model=model, mask_token_id=mask_token_id, num_steps=num_steps, temperature=temperature)
        p_tensor = torch.tensor([p_ids], dtype=torch.long, device=device)
        sample_tensor = sampler.sample(seq_len=total_len, prompt_ids=p_tensor, device=device)
        full_ids = sample_tensor[0].cpu().numpy().tolist()

    gen_ids = full_ids[len(p_ids):]
    # Filter out residual mask tokens if any
    gen_ids = [tok for tok in gen_ids if tok != mask_token_id]
    return tokenizer.decode(gen_ids)

def evaluate_model_iterative(model, tokenizer, backend: str, tasks: list, model_name: str, num_steps: int = 32, timeout_seconds: float = 3.0):
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")) if backend == "pytorch" else None
    if backend == "pytorch":
        model = model.to(device)
        model.eval()

    print("\n" + "=" * 85)
    print(f"  ITERATIVE MASK-INFILLING DECODING ({num_steps} STEPS): {model_name.upper()}")
    print(f"  Backend: {backend} (Device: {device}) | Steps: {num_steps} | Timeout: {timeout_seconds}s")
    print("=" * 85)

    results = []
    passed_count = 0
    ast_valid_count = 0
    outcome_counts = {k.value: 0 for k in ExecutionResult}
    start_time = time.time()

    for idx, task in enumerate(tasks, 1):
        task_id = task.get("task_id", task.get("id", f"task_{idx}"))
        prompt = task["prompt"]
        entry_point = task.get("entry_point")
        test_code = task.get("test", "")
        test_harness = task.get("test_harness", "")
        if not test_harness and test_code and entry_point:
            test_harness = f"{test_code}\n\ncheck({entry_point})\n"

        raw_comp = generate_iterative_infill_completion(
            model=model,
            tokenizer=tokenizer,
            backend=backend,
            device=device,
            prompt=prompt,
            num_steps=num_steps
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
    print(f"  FINAL RESULTS FOR {model_name} (Iterative Mask-Infilling {num_steps} Steps):")
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

    ckpt_path = PROJECT_ROOT / "checkpoints/corosred/unified/100m_5b/checkpoint_final.pt"
    if not ckpt_path.exists():
        print(f"Checkpoint not found at {ckpt_path}")
        return

    print(f"\nLoading 100M COROSred model from {ckpt_path}...")
    model, backend, vocab_size = load_model_from_checkpoint(ckpt_path)
    evaluate_model_iterative(model, tokenizer, backend, tasks, "100M COROSred (Iterative Infill 32 Steps)", num_steps=32)

if __name__ == "__main__":
    main()
