#!/usr/bin/env python3
"""Evaluate COROSred repair policies with edits strictly restricted to retained execution code.

Development Work:
- Executed on a separate development set (MBPP) to preserve fresh test tasks (HumanEval).
- Eligible tokens for repair are strictly filtered to the code retained for execution.
- Logs exact selected token (index, string, margin, entropy) and replacement (token, string, prob).
- Classifies and reports effective code edits, fixes, regressions, and neutral edits.
"""
from __future__ import annotations

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
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


def get_eligible_retained_token_indices(
    tok: Tokenizer,
    raw_text: str,
    prompt: str,
) -> Tuple[List[int], str]:
    """Find token indices whose character spans fall strictly within the retained execution code."""
    cleaned_code = clean_functional_completion(prompt, raw_text)
    
    # Locate where cleaned code starts and ends in the raw text
    start_char = raw_text.find(cleaned_code)
    if start_char == -1:
        cleaned_str = cleaned_code.strip()
        start_char = raw_text.find(cleaned_str) if cleaned_str else 0
        end_char = start_char + len(cleaned_str) if cleaned_str else len(raw_text)
    else:
        end_char = start_char + len(cleaned_code)

    enc = tok.encode(raw_text, add_special_tokens=False)
    eligible = []
    for i, (s, e) in enumerate(enc.offsets):
        # Must fall strictly within retained execution code
        if s >= start_char and e <= end_char and s < e:
            tok_text = tok.decode([enc.ids[i]])
            # Exclude pure whitespace / newlines to preserve structural indentation
            if tok_text.strip():
                eligible.append(i)
    return eligible, cleaned_code


def repair_single_token_retained(
    diff_model,
    tok: Tokenizer,
    prompt_ids: List[int],
    ar_tokens: List[int],
    confidences: List[Dict[str, float]],
    eligible_indices: List[int],
    device: str,
    margin_threshold: float = 0.5,
    entropy_threshold: float = 1.0,
) -> Tuple[str, bool, Optional[Dict[str, Any]], float]:
    """Policy 1: Mask only the single most uncertain token within retained code."""
    t0 = time.monotonic()
    eligible_set = set(eligible_indices)
    candidates = [
        (i, confidences[i]["margin"], confidences[i]["entropy"])
        for i in eligible_set
        if confidences[i]["margin"] < margin_threshold or confidences[i]["entropy"] > entropy_threshold
    ]
    if not candidates:
        return tok.decode(ar_tokens), False, None, 0.0

    # Pick single candidate with highest entropy
    candidates.sort(key=lambda x: -x[2])
    target_idx = candidates[0][0]
    orig_token = ar_tokens[target_idx]

    repaired = list(ar_tokens)
    repaired[target_idx] = MASK_TOKEN_ID
    full_seq = prompt_ids + repaired
    t_full = torch.tensor([full_seq], dtype=torch.long, device=device)

    with torch.no_grad():
        diff_logits = diff_model(t_full, is_causal=False)
        pos = len(prompt_ids) + target_idx
        l = diff_logits[0, pos].float()
        p = F.softmax(l, dim=-1)
        pred_token = int(l.argmax().item())
        pred_prob = float(p[pred_token].item())
        repaired[target_idx] = pred_token

    latency = time.monotonic() - t0
    repaired_text = tok.decode(repaired)

    edit_log = {
        "position": target_idx,
        "original_token_id": orig_token,
        "original_token_str": tok.decode([orig_token]),
        "original_margin": candidates[0][1],
        "original_entropy": candidates[0][2],
        "replacement_token_id": pred_token,
        "replacement_token_str": tok.decode([pred_token]),
        "replacement_prob": pred_prob,
        "is_different_token": (orig_token != pred_token),
    }
    return repaired_text, True, edit_log, latency


def repair_budget_k3_retained(
    diff_model,
    tok: Tokenizer,
    prompt_ids: List[int],
    ar_tokens: List[int],
    confidences: List[Dict[str, float]],
    eligible_indices: List[int],
    device: str,
    margin_threshold: float = 0.5,
    entropy_threshold: float = 1.0,
    k_budget: int = 3,
) -> Tuple[str, bool, List[Dict[str, Any]], float]:
    """Policy 2: Small fixed mask budget (up to k=3) within retained code."""
    t0 = time.monotonic()
    eligible_set = set(eligible_indices)
    candidates = [
        (i, confidences[i]["margin"], confidences[i]["entropy"])
        for i in eligible_set
        if confidences[i]["margin"] < margin_threshold or confidences[i]["entropy"] > entropy_threshold
    ]
    if not candidates:
        return tok.decode(ar_tokens), False, [], 0.0

    candidates.sort(key=lambda x: -x[2])
    selected = candidates[:k_budget]
    selected_indices = [c[0] for c in selected]

    repaired = list(ar_tokens)
    for idx in selected_indices:
        repaired[idx] = MASK_TOKEN_ID
    full_seq = prompt_ids + repaired
    t_full = torch.tensor([full_seq], dtype=torch.long, device=device)

    edits_log = []
    with torch.no_grad():
        diff_logits = diff_model(t_full, is_causal=False)
        for cand in selected:
            idx = cand[0]
            pos = len(prompt_ids) + idx
            l = diff_logits[0, pos].float()
            p = F.softmax(l, dim=-1)
            pred = int(l.argmax().item())
            prob = float(p[pred].item())
            orig = ar_tokens[idx]
            repaired[idx] = pred

            edits_log.append({
                "position": idx,
                "original_token_id": orig,
                "original_token_str": tok.decode([orig]),
                "original_margin": cand[1],
                "original_entropy": cand[2],
                "replacement_token_id": pred,
                "replacement_token_str": tok.decode([pred]),
                "replacement_prob": prob,
                "is_different_token": (orig != pred),
            })

    latency = time.monotonic() - t0
    repaired_text = tok.decode(repaired)
    return repaired_text, True, edits_log, latency


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
    parser = argparse.ArgumentParser(description="Evaluate COROSred with edits restricted to retained execution code.")
    parser.add_argument("--max-tasks", type=int, default=30, help="Number of development tasks to evaluate")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory path")
    args = parser.parse_args()

    print("=" * 80)
    print(f"DEVELOPMENT WORK: EVALUATING COROSRED RESTRICTED TO RETAINED EXECUTION CODE")
    print(f"Dataset: MBPP Development Set ({args.max_tasks} Tasks) | Fresh Tasks Reserved")
    print("=" * 80)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = Tokenizer.from_file(str(TOKENIZER_FILE))

    print(f"Loading AR Model onto {device}...")
    ar_model = load_qwen3_06b_model(AR_WEIGHTS, device=device)
    print(f"Loading MDLM Model onto {device}...")
    diff_model = load_qwen3_06b_model(DIFF_WEIGHTS, device=device)

    # Load MBPP development tasks
    mbpp_file = PROJECT_ROOT / "evals" / "benchmarks" / "mbpp_suite.json"
    with open(mbpp_file, "r", encoding="utf-8") as f:
        all_mbpp = json.load(f)
    dev_tasks = all_mbpp[:args.max_tasks]
    print(f"Loaded {len(dev_tasks)} tasks from MBPP development set.\n")

    out_dir_path = args.output_dir or str(PROJECT_ROOT / "research" / f"corosred_dev_retained_edits_{args.max_tasks}tasks_2026-10-18")
    out_dir = Path(out_dir_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    policies = ["untouched_ar", "single_token_retained", "budget_k3_retained"]
    stats = {
        p: {
            "passed": 0,
            "ast_valid": 0,
            "total_routing_triggers": 0,
            "effective_code_edits": 0,
            "noop_edits": 0,
            "drafts_fixed": 0,
            "drafts_broken": 0,
            "fixed_ids": [],
            "broken_ids": [],
            "total_latency": 0.0,
        }
        for p in policies
    }

    edits_log_master = []

    for idx, task in enumerate(dev_tasks, 1):
        task_id = task.get("id", f"MBPP/{idx}")
        prompt = task["prompt"]
        test_harness = task["test_harness"]
        prompt_ids = tok.encode(prompt, add_special_tokens=False).ids

        print(f"[{idx:02d}/{len(dev_tasks)}] Task {task_id}...", end="", flush=True)

        # 1. Generate AR draft ONCE and save
        ar_draft, ar_tokens, confs, ar_lat = generate_ar_draft(
            ar_model, tok, prompt, device=device, max_tokens=128
        )
        ar_passed, ar_ast_valid, ar_outcome, ar_code = evaluate_candidate(
            prompt, ar_draft, test_harness
        )
        stats["untouched_ar"]["passed"] += int(ar_passed)
        stats["untouched_ar"]["ast_valid"] += int(ar_ast_valid)
        stats["untouched_ar"]["total_latency"] += ar_lat

        # Compute eligible indices strictly within retained execution code
        eligible_indices, clean_ar_code = get_eligible_retained_token_indices(tok, ar_draft, prompt)

        # 2. Policy 1: Single-Token Repair within Retained Code
        p1_text, p1_routed, p1_log, p1_lat = repair_single_token_retained(
            diff_model, tok, prompt_ids, ar_tokens, confs, eligible_indices, device=device
        )
        p1_passed, p1_ast_valid, p1_outcome, p1_code = evaluate_candidate(
            prompt, p1_text, test_harness
        )
        stats["single_token_retained"]["passed"] += int(p1_passed)
        stats["single_token_retained"]["ast_valid"] += int(p1_ast_valid)
        stats["single_token_retained"]["total_latency"] += (ar_lat + p1_lat)

        p1_effective_code_edit = False
        if p1_routed and p1_log:
            stats["single_token_retained"]["total_routing_triggers"] += 1
            # Check if cleaned code actually changed
            clean_p1_code = clean_functional_completion(prompt, p1_text)
            p1_effective_code_edit = (clean_p1_code != clean_ar_code)
            if p1_effective_code_edit:
                stats["single_token_retained"]["effective_code_edits"] += 1
            else:
                stats["single_token_retained"]["noop_edits"] += 1

            if not ar_passed and p1_passed:
                stats["single_token_retained"]["drafts_fixed"] += 1
                stats["single_token_retained"]["fixed_ids"].append(task_id)
            elif ar_passed and not p1_passed:
                stats["single_token_retained"]["drafts_broken"] += 1
                stats["single_token_retained"]["broken_ids"].append(task_id)

            p1_log["task_id"] = task_id
            p1_log["policy"] = "single_token_retained"
            p1_log["effective_code_edit"] = p1_effective_code_edit
            p1_log["ar_passed"] = ar_passed
            p1_log["repaired_passed"] = p1_passed
            edits_log_master.append(p1_log)

        # 3. Policy 2: Small Fixed Budget k=3 within Retained Code
        p2_text, p2_routed, p2_logs, p2_lat = repair_budget_k3_retained(
            diff_model, tok, prompt_ids, ar_tokens, confs, eligible_indices, device=device, k_budget=3
        )
        p2_passed, p2_ast_valid, p2_outcome, p2_code = evaluate_candidate(
            prompt, p2_text, test_harness
        )
        stats["budget_k3_retained"]["passed"] += int(p2_passed)
        stats["budget_k3_retained"]["ast_valid"] += int(p2_ast_valid)
        stats["budget_k3_retained"]["total_latency"] += (ar_lat + p2_lat)

        p2_effective_code_edit = False
        if p2_routed and p2_logs:
            stats["budget_k3_retained"]["total_routing_triggers"] += 1
            clean_p2_code = clean_functional_completion(prompt, p2_text)
            p2_effective_code_edit = (clean_p2_code != clean_ar_code)
            if p2_effective_code_edit:
                stats["budget_k3_retained"]["effective_code_edits"] += 1
            else:
                stats["budget_k3_retained"]["noop_edits"] += 1

            if not ar_passed and p2_passed:
                stats["budget_k3_retained"]["drafts_fixed"] += 1
                stats["budget_k3_retained"]["fixed_ids"].append(task_id)
            elif ar_passed and not p2_passed:
                stats["budget_k3_retained"]["drafts_broken"] += 1
                stats["budget_k3_retained"]["broken_ids"].append(task_id)

            for l in p2_logs:
                l["task_id"] = task_id
                l["policy"] = "budget_k3_retained"
                l["effective_code_edit"] = p2_effective_code_edit
                l["ar_passed"] = ar_passed
                l["repaired_passed"] = p2_passed
                edits_log_master.append(l)

        results.append({
            "task_id": task_id,
            "prompt": prompt,
            "ar_draft": ar_draft,
            "clean_ar_code": clean_ar_code,
            "ar_passed": ar_passed,
            "ar_outcome": ar_outcome,
            "eligible_tokens_count": len(eligible_indices),
            "p1_single_retained": {
                "text": p1_text,
                "clean_code": clean_functional_completion(prompt, p1_text),
                "routed": p1_routed,
                "passed": p1_passed,
                "outcome": p1_outcome,
                "effective_edit": p1_effective_code_edit,
                "edit_log": p1_log,
            },
            "p2_budget_retained": {
                "text": p2_text,
                "clean_code": clean_functional_completion(prompt, p2_text),
                "routed": p2_routed,
                "passed": p2_passed,
                "outcome": p2_outcome,
                "effective_edit": p2_effective_code_edit,
                "edit_logs": p2_logs,
            },
        })

        p1_mark = 'PASS' if p1_passed else 'FAIL'
        p2_mark = 'PASS' if p2_passed else 'FAIL'
        print(f" AR: {'PASS' if ar_passed else 'FAIL'} | P1: {p1_mark} (eff={p1_effective_code_edit}) | P2: {p2_mark} (eff={p2_effective_code_edit})")

    n_tasks = len(dev_tasks)
    summary = {
        "development_dataset": f"MBPP ({n_tasks} Tasks)",
        "timestamp": int(time.time()),
        "policies": {
            p: {
                "pass_at_1_pct": round(stats[p]["passed"] / n_tasks * 100, 1),
                "ast_validity_pct": round(stats[p]["ast_valid"] / n_tasks * 100, 1),
                "total_routing_triggers": stats[p]["total_routing_triggers"],
                "effective_code_edits": stats[p]["effective_code_edits"],
                "noop_edits": stats[p]["noop_edits"],
                "drafts_fixed": stats[p]["drafts_fixed"],
                "drafts_fixed_ids": stats[p]["fixed_ids"],
                "drafts_broken": stats[p]["drafts_broken"],
                "drafts_broken_ids": stats[p]["broken_ids"],
                "mean_latency_seconds": round(stats[p]["total_latency"] / n_tasks, 3),
            }
            for p in policies
        },
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "edits_log.json").write_text(json.dumps(edits_log_master, indent=2))
    (out_dir / "per_task_results.json").write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 80)
    print("DEVELOPMENT SET SUMMARY (EDITS RESTRICTED TO RETAINED EXECUTION CODE):")
    print("=" * 80)
    for p in policies:
        sm = summary["policies"][p]
        print(f"\nPolicy: {p.upper()}")
        print(f"  Pass@1:               {sm['pass_at_1_pct']}% ({stats[p]['passed']}/{n_tasks})")
        print(f"  AST Validity:         {sm['ast_validity_pct']}%")
        print(f"  Routing Triggers:     {sm['total_routing_triggers']}/{n_tasks}")
        print(f"  Effective Code Edits: {sm['effective_code_edits']}")
        print(f"  No-Op Edits:          {sm['noop_edits']}")
        print(f"  Drafts Fixed:         {sm['drafts_fixed']} (IDs: {sm['drafts_fixed_ids']})")
        print(f"  Drafts Broken:        {sm['drafts_broken']} (IDs: {sm['drafts_broken_ids']})")
        print(f"  Mean Latency:         {sm['mean_latency_seconds']:.3f} s/task")
    print("=" * 80)
    print(f"Saved artifacts to {out_dir}")


if __name__ == "__main__":
    main()
