#!/usr/bin/env python3
"""Evaluate three COROSred repair policies on a separate development set (MBPP).

Policies compared:
1. Single-token repair: mask only the single most uncertain token.
2. Small fixed mask budget (k=3): mask top-3 uncertain tokens simultaneously.
3. Iterative confidence-based unmasking (k=3): mask top-3, unmask highest confidence first over 3 steps.

Protocol:
- Generate each AR draft ONCE.
- Save exact AR draft.
- Execute exact AR draft before repair (untouched-draft baseline).
- Apply each policy to the saved AR draft and execute after repair.
- Record regressions (correct drafts broken) and fixes (incorrect drafts fixed).
"""
from __future__ import annotations

import ast
import json
import os
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

TOKENIZER_FILE = (
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/tokenizer.json"
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


def repair_single_token(
    diff_model,
    tok: Tokenizer,
    prompt_ids: List[int],
    ar_tokens: List[int],
    confidences: List[Dict[str, float]],
    device: str,
    margin_threshold: float = 0.5,
    entropy_threshold: float = 1.0,
) -> Tuple[str, bool, int, float]:
    """Policy 1: Mask only the single most uncertain token."""
    t0 = time.monotonic()
    candidates = [
        (i, c["margin"], c["entropy"])
        for i, c in enumerate(confidences)
        if c["margin"] < margin_threshold or c["entropy"] > entropy_threshold
    ]
    if not candidates:
        return tok.decode(ar_tokens), False, 0, 0.0

    # Pick single token with highest entropy
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


def repair_budget_k3(
    diff_model,
    tok: Tokenizer,
    prompt_ids: List[int],
    ar_tokens: List[int],
    confidences: List[Dict[str, float]],
    device: str,
    margin_threshold: float = 0.5,
    entropy_threshold: float = 1.0,
    k_budget: int = 3,
) -> Tuple[str, bool, int, float]:
    """Policy 2: Small fixed mask budget (k=3), simultaneous repair."""
    t0 = time.monotonic()
    candidates = [
        (i, c["margin"], c["entropy"])
        for i, c in enumerate(confidences)
        if c["margin"] < margin_threshold or c["entropy"] > entropy_threshold
    ]
    if not candidates:
        return tok.decode(ar_tokens), False, 0, 0.0

    # Sort by highest entropy and take top-k
    candidates.sort(key=lambda x: -x[2])
    selected_indices = [c[0] for c in candidates[:k_budget]]

    repaired = list(ar_tokens)
    for idx in selected_indices:
        repaired[idx] = MASK_TOKEN_ID
    full_seq = prompt_ids + repaired
    t_full = torch.tensor([full_seq], dtype=torch.long, device=device)

    with torch.no_grad():
        diff_logits = diff_model(t_full, is_causal=False)
        for idx in selected_indices:
            pos = len(prompt_ids) + idx
            repaired[idx] = int(diff_logits[0, pos].argmax().item())

    latency = time.monotonic() - t0
    return tok.decode(repaired), True, len(selected_indices), latency


def repair_iterative_unmask(
    diff_model,
    tok: Tokenizer,
    prompt_ids: List[int],
    ar_tokens: List[int],
    confidences: List[Dict[str, float]],
    device: str,
    margin_threshold: float = 0.5,
    entropy_threshold: float = 1.0,
    k_budget: int = 3,
) -> Tuple[str, bool, int, int, float]:
    """Policy 3: Iterative confidence-based unmasking over k steps."""
    t0 = time.monotonic()
    candidates = [
        (i, c["margin"], c["entropy"])
        for i, c in enumerate(confidences)
        if c["margin"] < margin_threshold or c["entropy"] > entropy_threshold
    ]
    if not candidates:
        return tok.decode(ar_tokens), False, 0, 0, 0.0

    candidates.sort(key=lambda x: -x[2])
    masked_set = set(c[0] for c in candidates[:k_budget])
    num_masked = len(masked_set)

    repaired = list(ar_tokens)
    for idx in masked_set:
        repaired[idx] = MASK_TOKEN_ID

    prompt_len = len(prompt_ids)
    forward_passes = 0

    with torch.no_grad():
        while masked_set:
            forward_passes += 1
            full_seq = prompt_ids + repaired
            t_full = torch.tensor([full_seq], dtype=torch.long, device=device)
            diff_logits = diff_model(t_full, is_causal=False)

            best_idx = None
            best_token = None
            best_prob = -1.0

            for idx in masked_set:
                pos = prompt_len + idx
                l = diff_logits[0, pos].float()
                p = F.softmax(l, dim=-1)
                top_p, top_tok = torch.topk(p, 1)
                prob_val = float(top_p.item())
                if prob_val > best_prob:
                    best_prob = prob_val
                    best_token = int(top_tok.item())
                    best_idx = idx

            repaired[best_idx] = best_token
            masked_set.remove(best_idx)

    latency = time.monotonic() - t0
    return tok.decode(repaired), True, num_masked, forward_passes, latency


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
    print("=" * 80)
    print("DEVELOPMENT SET EVALUATION: COMPARING 3 COROSRED REPAIR POLICIES (MBPP)")
    print("=" * 80)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = Tokenizer.from_file(str(TOKENIZER_FILE))

    print(f"Loading AR Model onto {device}...")
    ar_model = load_qwen3_06b_model(AR_WEIGHTS, device=device)
    print(f"Loading MDLM Model onto {device}...")
    diff_model = load_qwen3_06b_model(DIFF_WEIGHTS, device=device)

    # Load 20 tasks from MBPP
    mbpp_file = PROJECT_ROOT / "evals" / "benchmarks" / "mbpp_suite.json"
    with open(mbpp_file, "r", encoding="utf-8") as f:
        all_mbpp = json.load(f)
    dev_tasks = all_mbpp[:20]
    print(f"Loaded {len(dev_tasks)} tasks from MBPP development set.\n")

    out_dir = PROJECT_ROOT / "research" / "corosred_dev_mbpp_policies_2026-10-18"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    policy_names = ["untouched_ar", "single_token", "budget_k3", "iterative_unmask"]
    stats = {p: {"passed": 0, "ast_valid": 0, "fixed": 0, "broken": 0, "total_time": 0.0} for p in policy_names}

    for idx, task in enumerate(dev_tasks, 1):
        task_id = task.get("id", f"MBPP/{idx}")
        prompt = task["prompt"]
        test_harness = task["test_harness"]
        prompt_ids = tok.encode(prompt, add_special_tokens=False).ids

        print(f"[{idx:02d}/{len(dev_tasks)}] Task {task_id}...", end="", flush=True)

        # 1. Generate AR draft ONCE
        draft_text, ar_tokens, confidences, ar_latency = generate_ar_draft(
            ar_model, tok, prompt, device=device, max_tokens=128
        )
        ar_passed, ar_ast_valid, ar_outcome, ar_code = evaluate_candidate(
            prompt, draft_text, test_harness
        )
        stats["untouched_ar"]["passed"] += int(ar_passed)
        stats["untouched_ar"]["ast_valid"] += int(ar_ast_valid)
        stats["untouched_ar"]["total_time"] += ar_latency

        # 2. Policy 1: Single Token Repair
        p1_text, p1_routed, p1_masks, p1_lat = repair_single_token(
            diff_model, tok, prompt_ids, ar_tokens, confidences, device=device
        )
        p1_passed, p1_ast_valid, p1_outcome, p1_code = evaluate_candidate(
            prompt, p1_text, test_harness
        )
        stats["single_token"]["passed"] += int(p1_passed)
        stats["single_token"]["ast_valid"] += int(p1_ast_valid)
        stats["single_token"]["total_time"] += (ar_latency + p1_lat)
        if not ar_passed and p1_passed:
            stats["single_token"]["fixed"] += 1
        elif ar_passed and not p1_passed:
            stats["single_token"]["broken"] += 1

        # 3. Policy 2: Budget k=3
        p2_text, p2_routed, p2_masks, p2_lat = repair_budget_k3(
            diff_model, tok, prompt_ids, ar_tokens, confidences, device=device, k_budget=3
        )
        p2_passed, p2_ast_valid, p2_outcome, p2_code = evaluate_candidate(
            prompt, p2_text, test_harness
        )
        stats["budget_k3"]["passed"] += int(p2_passed)
        stats["budget_k3"]["ast_valid"] += int(p2_ast_valid)
        stats["budget_k3"]["total_time"] += (ar_latency + p2_lat)
        if not ar_passed and p2_passed:
            stats["budget_k3"]["fixed"] += 1
        elif ar_passed and not p2_passed:
            stats["budget_k3"]["broken"] += 1

        # 4. Policy 3: Iterative Unmasking k=3
        p3_text, p3_routed, p3_masks, p3_passes, p3_lat = repair_iterative_unmask(
            diff_model, tok, prompt_ids, ar_tokens, confidences, device=device, k_budget=3
        )
        p3_passed, p3_ast_valid, p3_outcome, p3_code = evaluate_candidate(
            prompt, p3_text, test_harness
        )
        stats["iterative_unmask"]["passed"] += int(p3_passed)
        stats["iterative_unmask"]["ast_valid"] += int(p3_ast_valid)
        stats["iterative_unmask"]["total_time"] += (ar_latency + p3_lat)
        if not ar_passed and p3_passed:
            stats["iterative_unmask"]["fixed"] += 1
        elif ar_passed and not p3_passed:
            stats["iterative_unmask"]["broken"] += 1

        results.append({
            "task_id": task_id,
            "prompt": prompt,
            "ar_draft": draft_text,
            "ar_passed": ar_passed,
            "ar_outcome": ar_outcome,
            "policy1_single": {"text": p1_text, "passed": p1_passed, "outcome": p1_outcome, "latency": p1_lat},
            "policy2_budget_k3": {"text": p2_text, "passed": p2_passed, "outcome": p2_outcome, "latency": p2_lat},
            "policy3_iterative": {"text": p3_text, "passed": p3_passed, "outcome": p3_outcome, "passes": p3_passes, "latency": p3_lat},
        })

        print(f" AR: {'PASS' if ar_passed else 'FAIL'} | P1: {'PASS' if p1_passed else 'FAIL'} | P2: {'PASS' if p2_passed else 'FAIL'} | P3: {'PASS' if p3_passed else 'FAIL'}")

    n_tasks = len(dev_tasks)
    summary = {
        "development_set": "MBPP (20 tasks)",
        "timestamp": int(time.time()),
        "policies": {
            p: {
                "pass_at_1_pct": round(stats[p]["passed"] / n_tasks * 100, 1),
                "ast_validity_pct": round(stats[p]["ast_valid"] / n_tasks * 100, 1),
                "drafts_fixed": stats[p]["fixed"],
                "drafts_broken": stats[p]["broken"],
                "mean_latency": round(stats[p]["total_time"] / n_tasks, 3),
            }
            for p in policy_names
        },
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "per_task_results.json").write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 80)
    print("DEVELOPMENT SET POLICY COMPARISON SUMMARY:")
    print("=" * 80)
    print(f"{'Policy':<20} | {'Pass@1':<10} | {'AST Valid':<10} | {'Fixed':<6} | {'Broken':<6} | {'Mean Latency'}")
    print("-" * 80)
    for p in policy_names:
        s = summary["policies"][p]
        print(f"{p:<20} | {s['pass_at_1_pct']}%{'':<5} | {s['ast_validity_pct']}%{'':<5} | {s['drafts_fixed']:<6} | {s['drafts_broken']:<6} | {s['mean_latency']:.3f}s")
    print("=" * 80)
    print(f"Saved artifacts to {out_dir}")


if __name__ == "__main__":
    main()
