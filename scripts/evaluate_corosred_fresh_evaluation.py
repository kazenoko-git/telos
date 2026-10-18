#!/usr/bin/env python3
"""Evaluate Untouched AR baseline, Working MDLM baseline, and Frozen COROSred on fresh tasks.

Tasks: 40 tasks from HumanEval (fresh evaluation set)
Protocol:
1. Untouched AR Draft Baseline:
   - Generate each AR draft ONCE.
   - Save exact AR draft.
   - Execute exact AR draft before repair.
2. Working MDLM Baseline:
   - Use documented chat template (system prompt, user prompt, enable_thinking=False).
   - Use documented block diffusion (block_size=64, steps=128, remasking='low_confidence').
   - Extract code via clean_functional_completion and execute in sandbox.
3. Frozen COROSred Policy (Single-Token Repair):
   - Repair exact saved AR draft by masking only the single most uncertain token.
   - Execute after repair.
4. Report all regressions, drafts fixed, latency, and routing stats.
"""
from __future__ import annotations

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import ast
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TOKENIZER_FILE = (
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/tokenizer.json"
)

from telos.eval.runner import (
    clean_functional_completion,
    check_ast_validity,
    execute_code_sandboxed,
    ExecutionResult,
)
from scripts.qwen3_06b_model import (
    load_qwen3_06b_model,
    AR_WEIGHTS,
    DIFF_WEIGHTS,
)

EOS_TOKENS = {151645, 151643}
MASK_TOKEN_ID = 151669
DEFAULT_STOP_STRINGS = ["\nclass ", "\ndef ", "\nif __name__"]


def generate_ar_draft(
    model,
    tok: Tokenizer,
    prompt: str,
    device: str,
    max_tokens: int = 128,
) -> Tuple[str, List[int], List[Dict[str, float]], float]:
    """Generate AR draft with KV cache and track per-token confidence."""
    t0 = time.monotonic()
    input_ids = tok.encode(prompt, add_special_tokens=False).ids
    prompt_len = len(input_ids)
    ar_tokens: List[int] = []
    token_confidences: List[Dict[str, float]] = []

    t_in = torch.tensor([input_ids], dtype=torch.long, device=device)
    logits, past_kvs = model(t_in, is_causal=True, use_cache=True)
    l_last = logits[0, -1, :]
    probs = F.softmax(l_last.float(), dim=-1)
    top2 = torch.topk(probs, 2)
    p1 = float(top2.values[0].item())
    p2 = float(top2.values[1].item())
    entropy = -float(torch.sum(probs * torch.log(probs.clamp_min(1e-12))).item())
    pred_token = int(top2.indices[0].item())

    for _ in range(max_tokens):
        if pred_token in EOS_TOKENS:
            break
        ar_tokens.append(pred_token)
        token_confidences.append({"margin": p1 - p2, "entropy": entropy})

        text_so_far = tok.decode(ar_tokens)
        if any(s in text_so_far for s in DEFAULT_STOP_STRINGS):
            break

        t_curr = torch.tensor([[pred_token]], dtype=torch.long, device=device)
        logits, past_kvs = model(
            t_curr,
            is_causal=True,
            past_key_values=past_kvs,
            start_pos=prompt_len + len(ar_tokens) - 1,
            use_cache=True,
        )
        l_last = logits[0, -1, :]
        probs = F.softmax(l_last.float(), dim=-1)
        top2 = torch.topk(probs, 2)
        p1 = float(top2.values[0].item())
        p2 = float(top2.values[1].item())
        entropy = -float(torch.sum(probs * torch.log(probs.clamp_min(1e-12))).item())
        pred_token = int(top2.indices[0].item())

    latency = time.monotonic() - t0
    draft_text = tok.decode(ar_tokens)
    return draft_text, ar_tokens, token_confidences, latency


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """Calculate transfer tokens schedule for MDLM block diffusion."""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer[i, :remainder[i]] += 1
    return num_transfer


@torch.no_grad()
def generate_working_mdlm(
    model,
    tok: Tokenizer,
    prompt: str,
    device: str,
    steps: int = 128,
    max_new_tokens: int = 128,
    block_size: int = 64,
) -> Tuple[str, float]:
    """Generate completion using MDLM's documented input format and decoding procedure."""
    t0 = time.monotonic()
    chat_text = (
        f"<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n"
        f"<|im_start|>user\nComplete the following Python function:\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    prompt_ids = tok.encode(chat_text, add_special_tokens=False).ids
    prompt_len = len(prompt_ids)
    total_len = prompt_len + max_new_tokens
    pad_id = 151643  # <|endoftext|>

    x = torch.full((1, total_len), pad_id, dtype=torch.long, device=device)
    x[0, :prompt_len] = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    x[0, prompt_len:] = MASK_TOKEN_ID

    positions = torch.arange(total_len, device=device)
    num_blocks = max_new_tokens // block_size
    steps_per_block = steps // num_blocks

    for num_block in range(num_blocks):
        block_start = prompt_len + num_block * block_size
        block_end = block_start + block_size
        init_block_mask = (positions >= block_start) & (positions < block_end) & (x[0] == MASK_TOKEN_ID)
        num_transfer = get_num_transfer_tokens(init_block_mask.unsqueeze(0), steps_per_block)

        for s in range(steps_per_block):
            block_mask = (positions >= block_start) & (positions < block_end) & (x[0] == MASK_TOKEN_ID)
            if not block_mask.any():
                break
            logits = model(x, is_causal=False)
            probs = F.softmax(logits.float(), dim=-1)
            x0 = torch.argmax(probs, dim=-1)
            x0_p = torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)

            confidence = torch.where(block_mask, x0_p[0], torch.tensor(-float("inf"), device=device))
            k = int(num_transfer[0, s].item())
            if k > 0:
                _, select_index = torch.topk(confidence, k=min(k, int(block_mask.sum().item())))
                x[0, select_index] = x0[0, select_index]

    new_ids = x[0, prompt_len:].tolist()
    # Truncate at EOS
    for i, tid in enumerate(new_ids):
        if tid in EOS_TOKENS:
            new_ids = new_ids[:i]
            break
    latency = time.monotonic() - t0
    raw_completion = tok.decode(new_ids)
    return raw_completion, latency


def repair_frozen_single_token(
    diff_model,
    tok,
    prompt_ids: List[int],
    ar_tokens: List[int],
    confidences: List[Dict[str, float]],
    device: str,
    margin_threshold: float = 0.5,
    entropy_threshold: float = 1.0,
) -> Tuple[str, bool, int, float]:
    """Frozen Policy: Single-token repair on most uncertain token."""
    t0 = time.monotonic()
    candidates = [
        (i, c["margin"], c["entropy"])
        for i, c in enumerate(confidences)
        if c["margin"] < margin_threshold or c["entropy"] > entropy_threshold
    ]
    if not candidates:
        return tok.decode(ar_tokens), False, 0, 0.0

    candidates.sort(key=lambda x: -x[2])
    target_idx = candidates[0][0]

    repaired = list(ar_tokens)
    repaired[target_idx] = MASK_TOKEN_ID
    full_seq = prompt_ids + repaired
    t_full = torch.tensor([full_seq], dtype=torch.long, device=device)

    with torch.no_grad():
        diff_logits = diff_model(t_full, is_causal=False)
        pos = len(prompt_ids) + target_idx
        pred = int(diff_logits[0, pos].argmax().item())
        repaired[target_idx] = pred

    latency = time.monotonic() - t0
    return tok.decode(repaired), True, 1, latency


def evaluate_candidate(
    prompt: str,
    raw_completion: str,
    test_harness: str,
    timeout_seconds: float = 3.0,
) -> Tuple[bool, bool, str, str]:
    """Clean, check AST, and execute completion in sandbox."""
    completion = clean_functional_completion(prompt, raw_completion)
    try:
        tree = ast.parse(completion)
        if any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in ast.walk(tree)):
            full_code = completion
        else:
            full_code = prompt + ("\n" if not prompt.endswith("\n") else "") + completion
    except SyntaxError:
        full_code = prompt + ("\n" if not prompt.endswith("\n") else "") + completion

    is_ast_valid, _ = check_ast_validity(full_code)
    outcome, details = execute_code_sandboxed(
        code=full_code,
        test_harness=test_harness,
        timeout_seconds=timeout_seconds,
    )
    is_passed = (outcome == ExecutionResult.PASSED)
    return is_passed, is_ast_valid, outcome.value, full_code


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate on fresh HumanEval tasks.")
    parser.add_argument("--max-tasks", type=int, default=40, help="Max tasks to evaluate")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    print("=" * 80)
    print(f"EVALUATION ON FRESH TEST SET: HUMANEVAL ({args.max_tasks} TASKS)")
    print("Comparing: 1) Untouched AR Draft Baseline")
    print("           2) Working MDLM Baseline (Documented Chat + Block Diffusion)")
    print("           3) Frozen COROSred (Single-Token Repair)")
    print("=" * 80)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = Tokenizer.from_file(str(TOKENIZER_FILE))

    print(f"Loading AR Model onto {device}...")
    ar_model = load_qwen3_06b_model(AR_WEIGHTS, device=device)
    print(f"Loading MDLM Model onto {device}...")
    diff_model = load_qwen3_06b_model(DIFF_WEIGHTS, device=device)

    # Load fresh tasks from HumanEval
    he_file = PROJECT_ROOT / "evals" / "benchmarks" / "humaneval_suite.json"
    with open(he_file, "r", encoding="utf-8") as f:
        all_he = json.load(f)
    fresh_tasks = all_he[:args.max_tasks]
    print(f"Loaded {len(fresh_tasks)} tasks from HumanEval fresh test set.\n")

    out_dir_path = args.output_dir or str(PROJECT_ROOT / "research" / f"corosred_fresh_evaluation_{args.max_tasks}tasks_2026-10-18")
    out_dir = Path(out_dir_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    systems = ["untouched_ar", "working_mdlm", "frozen_corosred"]
    stats = {
        s: {
            "passed": 0,
            "ast_valid": 0,
            "outcomes": {k.value: 0 for k in ExecutionResult},
            "total_time": 0.0,
            "drafts_fixed": 0,
            "drafts_broken": 0,
            "broken_task_ids": [],
            "fixed_task_ids": [],
        }
        for s in systems
    }
    total_routing_triggers = 0

    for idx, task in enumerate(fresh_tasks, 1):
        task_id = task.get("id", f"HumanEval/{idx-1}")
        prompt = task["prompt"]
        test_harness = task["test_harness"]
        prompt_ids = tok.encode(prompt, add_special_tokens=False).ids

        print(f"[{idx:02d}/{len(fresh_tasks)}] Task {task_id}...", end="", flush=True)

        # 1. System 1: Generate AR draft ONCE and evaluate untouched baseline
        ar_draft, ar_tokens, confs, ar_lat = generate_ar_draft(
            ar_model, tok, prompt, device=device, max_tokens=128
        )
        ar_passed, ar_ast_valid, ar_outcome, ar_code = evaluate_candidate(
            prompt, ar_draft, test_harness
        )
        stats["untouched_ar"]["passed"] += int(ar_passed)
        stats["untouched_ar"]["ast_valid"] += int(ar_ast_valid)
        stats["untouched_ar"]["outcomes"][ar_outcome] += 1
        stats["untouched_ar"]["total_time"] += ar_lat

        # 2. System 2: Working MDLM Baseline (Documented Chat + Block Diffusion)
        mdlm_raw, mdlm_lat = generate_working_mdlm(
            diff_model, tok, prompt, device=device, steps=128, max_new_tokens=128, block_size=64
        )
        mdlm_passed, mdlm_ast_valid, mdlm_outcome, mdlm_code = evaluate_candidate(
            prompt, mdlm_raw, test_harness
        )
        stats["working_mdlm"]["passed"] += int(mdlm_passed)
        stats["working_mdlm"]["ast_valid"] += int(mdlm_ast_valid)
        stats["working_mdlm"]["outcomes"][mdlm_outcome] += 1
        stats["working_mdlm"]["total_time"] += mdlm_lat

        # 3. System 3: Frozen COROSred (Single-Token Repair on exact saved AR draft)
        coros_text, coros_routed, coros_masks, coros_repair_lat = repair_frozen_single_token(
            diff_model, tok, prompt_ids, ar_tokens, confs, device=device
        )
        coros_total_lat = ar_lat + coros_repair_lat
        coros_passed, coros_ast_valid, coros_outcome, coros_code = evaluate_candidate(
            prompt, coros_text, test_harness
        )
        stats["frozen_corosred"]["passed"] += int(coros_passed)
        stats["frozen_corosred"]["ast_valid"] += int(coros_ast_valid)
        stats["frozen_corosred"]["outcomes"][coros_outcome] += 1
        stats["frozen_corosred"]["total_time"] += coros_total_lat
        if coros_routed:
            total_routing_triggers += 1

        # Check regressions and fixes relative to untouched AR
        if not ar_passed and coros_passed:
            stats["frozen_corosred"]["drafts_fixed"] += 1
            stats["frozen_corosred"]["fixed_task_ids"].append(task_id)
        elif ar_passed and not coros_passed:
            stats["frozen_corosred"]["drafts_broken"] += 1
            stats["frozen_corosred"]["broken_task_ids"].append(task_id)

        results.append({
            "task_id": task_id,
            "prompt": prompt,
            "ar_draft": ar_draft,
            "ar_passed": ar_passed,
            "ar_outcome": ar_outcome,
            "ar_latency": ar_lat,
            "mdlm_completion": mdlm_raw,
            "mdlm_passed": mdlm_passed,
            "mdlm_outcome": mdlm_outcome,
            "mdlm_latency": mdlm_lat,
            "coros_completion": coros_text,
            "coros_passed": coros_passed,
            "coros_outcome": coros_outcome,
            "coros_routed": coros_routed,
            "coros_repair_latency": coros_repair_lat,
            "coros_total_latency": coros_total_lat,
        })

        print(f" AR: {'PASS' if ar_passed else 'FAIL'} | MDLM: {'PASS' if mdlm_passed else 'FAIL'} | COROS: {'PASS' if coros_passed else 'FAIL'}")

    n_tasks = len(fresh_tasks)
    master_summary = {
        "suite": "HumanEval (Fresh Test Set)",
        "total_tasks": n_tasks,
        "timestamp": int(time.time()),
        "pinned_revisions": {
            "ar_model": "Qwen/Qwen3-0.6B @ c1899de289a04d12100db370d81485cdf75e47ca",
            "diff_model": "dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1 @ c8d24a3f4adaeef46881b450e1bf7d1005203bd7",
        },
        "systems": {
            s: {
                "pass_at_1_pct": round(stats[s]["passed"] / n_tasks * 100, 1),
                "ast_validity_pct": round(stats[s]["ast_valid"] / n_tasks * 100, 1),
                "total_time_seconds": round(stats[s]["total_time"], 2),
                "mean_latency_per_task": round(stats[s]["total_time"] / n_tasks, 3),
                "outcomes": stats[s]["outcomes"],
                "drafts_fixed_count": stats[s]["drafts_fixed"],
                "drafts_fixed_ids": stats[s]["fixed_task_ids"],
                "drafts_broken_count": stats[s]["drafts_broken"],
                "drafts_broken_ids": stats[s]["broken_task_ids"],
            }
            for s in systems
        },
        "corosred_routing": {
            "routing_triggers": f"{total_routing_triggers}/{n_tasks} ({total_routing_triggers/n_tasks*100:.1f}%)",
            "policy": "single_token (frozen from MBPP dev set)",
        },
    }

    (out_dir / "master_summary.json").write_text(json.dumps(master_summary, indent=2))
    (out_dir / "per_task_results.json").write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 80)
    print("FRESH TEST EVALUATION MASTER SUMMARY (40 HUMANEVAL TASKS):")
    print("=" * 80)
    for s in systems:
        sm = master_summary["systems"][s]
        print(f"\nSystem: {s.upper()}")
        print(f"  Pass@1:          {sm['pass_at_1_pct']}% ({stats[s]['passed']}/{n_tasks})")
        print(f"  AST Validity:    {sm['ast_validity_pct']}% ({stats[s]['ast_valid']}/{n_tasks})")
        print(f"  Mean Latency:    {sm['mean_latency_per_task']:.3f} s/task")
        print(f"  Drafts Broken:   {sm['drafts_broken_count']} (IDs: {sm['drafts_broken_ids']})")
        print(f"  Drafts Fixed:    {sm['drafts_fixed_count']} (IDs: {sm['drafts_fixed_ids']})")
        print(f"  Outcomes:        {sm['outcomes']}")
    print("=" * 80)
    print(f"Artifacts saved to {out_dir}")


if __name__ == "__main__":
    main()
