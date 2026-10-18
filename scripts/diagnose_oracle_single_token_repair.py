#!/usr/bin/env python3
"""Offline Oracle Diagnostic: Exhaustive Single-Token MDLM Repairability Scan.

Evaluates a bounded sample of 15 failed MBPP development drafts.
Exhaustively masks each eligible executable-code token individually (excluding comments,
docstrings, whitespace, and discarded text).
Records whether any single-token MDLM replacement makes the exact saved draft pass unit tests.
Measures:
- How many drafts have ANY successful repair?
- Which positions work?
- Where do those positions rank under AR uncertainty (entropy and margin ranks)?
"""
from __future__ import annotations

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import ast
import json
import re
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

MASK_TOKEN_ID = 151669


def get_eligible_executable_tokens(
    tok: Tokenizer,
    raw_text: str,
    prompt: str,
) -> Tuple[List[int], str, List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Find token indices strictly corresponding to executable code within retained text.
    
    Excludes:
    - Discarded text (outside clean_functional_completion)
    - Pure whitespace/newlines
    - Comments (#.*$)
    - Docstrings (\"\"\"...\"\"\" or '''...''')
    """
    cleaned_code = clean_functional_completion(prompt, raw_text)

    start_char = raw_text.find(cleaned_code)
    if start_char == -1:
        cleaned_str = cleaned_code.strip()
        start_char = raw_text.find(cleaned_str) if cleaned_str else 0
        end_char = start_char + len(cleaned_str) if cleaned_str else len(raw_text)
    else:
        end_char = start_char + len(cleaned_code)

    comment_spans = [(m.start(), m.end()) for m in re.finditer(r"#.*$", raw_text, re.MULTILINE)]
    docstring_spans = [(m.start(), m.end()) for m in re.finditer(r'("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')', raw_text)]

    enc = tok.encode(raw_text, add_special_tokens=False)
    eligible = []
    for i, (s, e) in enumerate(enc.offsets):
        # Must fall strictly inside retained code
        if s < start_char or e > end_char or s >= e:
            continue
        tok_str = tok.decode([enc.ids[i]])
        if not tok_str.strip():
            continue
        # Exclude comment overlap
        if any(max(s, cs) < min(e, ce) for cs, ce in comment_spans):
            continue
        # Exclude docstring overlap
        if any(max(s, ds) < min(e, de) for ds, de in docstring_spans):
            continue
        eligible.append(i)

    return eligible, cleaned_code, comment_spans, docstring_spans


def main():
    print("=" * 80)
    print("OFFLINE ORACLE DIAGNOSTIC: EXHAUSTIVE SINGLE-TOKEN MDLM REPAIR SCAN")
    print("Dataset: Bounded Sample of 15 Failed MBPP Development Drafts")
    print("Executable Code Only: Excluding Comments, Docstrings, Whitespace, Dead Code")
    print("=" * 80)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = Tokenizer.from_file(str(TOKENIZER_FILE))

    print(f"Loading MDLM Model onto {device}...")
    diff_model = load_qwen3_06b_model(DIFF_WEIGHTS, device=device)

    # Load previously evaluated MBPP dev results
    prev_results_path = PROJECT_ROOT / "research" / "corosred_dev_retained_edits_30tasks_2026-10-18" / "per_task_results.json"
    with open(prev_results_path, "r", encoding="utf-8") as f:
        all_dev = json.load(f)

    # Select 15 failed tasks (prioritizing tasks with code lines, plus 2 comment-only tasks)
    failed_tasks = [r for r in all_dev if not r["ar_passed"]]
    tasks_with_code = [
        r for r in failed_tasks
        if any(l.strip() and not l.strip().startswith("#") for l in r["clean_ar_code"].splitlines())
    ]
    tasks_without_code = [
        r for r in failed_tasks
        if not any(l.strip() and not l.strip().startswith("#") for l in r["clean_ar_code"].splitlines())
    ]

    selected_sample = tasks_with_code[:13] + tasks_without_code[:2]
    print(f"Sample Selected: {len(selected_sample)} failed tasks ({len(tasks_with_code[:13])} with code, {len(tasks_without_code[:2])} comment-only).\n")

    out_dir = PROJECT_ROOT / "research" / "corosred_oracle_diagnostic_2026-10-18"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load MBPP test harnesses from dataset
    with open(PROJECT_ROOT / "evals" / "benchmarks" / "mbpp_suite.json") as f:
        mbpp_full = {t["id"]: t["test_harness"] for t in json.load(f)}

    # Track metrics
    total_drafts = len(selected_sample)
    drafts_with_any_successful_repair = 0
    tasks_diagnostic_records = []
    all_successful_repairs = []

    for t_idx, task_record in enumerate(selected_sample, 1):
        task_id = task_record["task_id"]
        prompt = task_record["prompt"]
        ar_draft = task_record["ar_draft"]
        test_harness = mbpp_full[task_id]

        prompt_ids = tok.encode(prompt, add_special_tokens=False).ids
        prompt_len = len(prompt_ids)
        enc = tok.encode(ar_draft, add_special_tokens=False)
        ar_tokens = list(enc.ids)

        eligible_indices, clean_code, _, _ = get_eligible_executable_tokens(tok, ar_draft, prompt)
        n_eligible = len(eligible_indices)

        print(f"[{t_idx:02d}/{total_drafts}] Task {task_id}: {n_eligible} eligible executable tokens...", flush=True)

        if n_eligible == 0:
            tasks_diagnostic_records.append({
                "task_id": task_id,
                "eligible_executable_tokens": 0,
                "has_successful_repair": False,
                "repairs": [],
                "note": "No executable code tokens (draft consists entirely of comments or whitespace)",
            })
            continue

        # Load AR per-token confidences
        # In per_task_results.json, we can reconstruct or load confidences
        # If not present, we can recompute AR logits at each draft position with ar_model
        # Or load from edit logs. Let's compute them accurately:
        # Re-run AR forward pass to get exact per-token entropy and margin
        # We can do this with the cached tokens
        t_ar_seq = torch.tensor([prompt_ids + ar_tokens], device=device)
        # We need AR model to compute exact margin and entropy
        # Let's load AR model once if needed
        ar_model = load_qwen3_06b_model(AR_WEIGHTS, device=device)
        with torch.no_grad():
            ar_logits = ar_model(t_ar_seq, is_causal=True)
            # Position i in ar_tokens was predicted by prompt_len + i - 1
            pos_logits = ar_logits[0, prompt_len - 1 : prompt_len + len(ar_tokens) - 1, :].float()
            probs = F.softmax(pos_logits, dim=-1)
            top2 = torch.topk(probs, 2, dim=-1)
            margins = (top2.values[:, 0] - top2.values[:, 1]).tolist()
            entropies = (-torch.sum(probs * torch.log(probs.clamp_min(1e-12)), dim=-1)).tolist()

        # Compute ranks among eligible tokens
        eligible_entropies = [(i, entropies[i]) for i in eligible_indices]
        eligible_margins = [(i, margins[i]) for i in eligible_indices]
        eligible_entropies.sort(key=lambda x: -x[1])  # highest entropy = Rank 1
        eligible_margins.sort(key=lambda x: x[1])    # lowest margin = Rank 1

        rank_H_map = {item[0]: rank + 1 for rank, item in enumerate(eligible_entropies)}
        rank_m_map = {item[0]: rank + 1 for rank, item in enumerate(eligible_margins)}

        task_repairs = []
        task_effective_edits = 0

        # Scan each eligible token position
        for p in eligible_indices:
            orig_token_id = ar_tokens[p]
            orig_token_str = tok.decode([orig_token_id])

            repaired_seq = list(ar_tokens)
            repaired_seq[p] = MASK_TOKEN_ID
            full_seq = prompt_ids + repaired_seq
            t_full = torch.tensor([full_seq], device=device)

            with torch.no_grad():
                diff_logits = diff_model(t_full, is_causal=False)
                pos = prompt_len + p
                l = diff_logits[0, pos].float()
                p_dist = F.softmax(l, dim=-1)
                pred_token_id = int(l.argmax().item())
                pred_prob = float(p_dist[pred_token_id].item())
                pred_token_str = tok.decode([pred_token_id])

            if pred_token_id == orig_token_id:
                # MDLM regenerates the same token
                continue

            task_effective_edits += 1

            # Test repaired completion in sandbox
            repaired_seq[p] = pred_token_id
            repaired_code_clean = clean_functional_completion(prompt, tok.decode(repaired_seq))
            candidate_code = prompt + ("\n" if not prompt.endswith("\n") else "") + repaired_code_clean

            outcome, details = execute_code_sandboxed(candidate_code, test_harness, timeout_seconds=3.0)
            is_passed = (outcome == ExecutionResult.PASSED)

            if is_passed:
                repair_info = {
                    "task_id": task_id,
                    "position": p,
                    "original_token_id": orig_token_id,
                    "original_token_str": orig_token_str,
                    "replacement_token_id": pred_token_id,
                    "replacement_token_str": pred_token_str,
                    "replacement_prob": pred_prob,
                    "ar_entropy": entropies[p],
                    "entropy_rank": f"{rank_H_map[p]}/{n_eligible}",
                    "ar_margin": margins[p],
                    "margin_rank": f"{rank_m_map[p]}/{n_eligible}",
                    "original_code": clean_code,
                    "repaired_code": repaired_code_clean,
                }
                task_repairs.append(repair_info)
                all_successful_repairs.append(repair_info)
                print(f"  >>> SUCCESSFUL REPAIR at pos {p}: {repr(orig_token_str)} -> {repr(pred_token_str)} (H-Rank: {rank_H_map[p]}/{n_eligible}) <<<")

        has_repair = len(task_repairs) > 0
        if has_repair:
            drafts_with_any_successful_repair += 1

        tasks_diagnostic_records.append({
            "task_id": task_id,
            "eligible_executable_tokens": n_eligible,
            "effective_edits_count": task_effective_edits,
            "has_successful_repair": has_repair,
            "successful_repairs_count": len(task_repairs),
            "repairs": task_repairs,
        })
        print(f"  Result: {len(task_repairs)} successful repairs (out of {task_effective_edits} effective edits tested).")

    summary = {
        "diagnostic_suite": "MBPP Offline Oracle Single-Token Diagnostic",
        "sample_size": total_drafts,
        "timestamp": int(time.time()),
        "drafts_with_executable_code": len(tasks_with_code[:13]),
        "comment_only_drafts": len(tasks_without_code[:2]),
        "total_eligible_tokens_scanned": sum(r["eligible_executable_tokens"] for r in tasks_diagnostic_records),
        "total_effective_edits_tested": sum(r.get("effective_edits_count", 0) for r in tasks_diagnostic_records),
        "drafts_with_any_successful_repair": drafts_with_any_successful_repair,
        "repairable_percentage": round(drafts_with_any_successful_repair / total_drafts * 100, 1),
        "all_successful_repairs": all_successful_repairs,
    }

    (out_dir / "results.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "tasks_diagnostic.json").write_text(json.dumps(tasks_diagnostic_records, indent=2))

    print("\n" + "=" * 80)
    print("ORACLE DIAGNOSTIC SUMMARY:")
    print("=" * 80)
    print(f"Total Failed Drafts Scanned:       {total_drafts}")
    print(f"Drafts with Executable Code:       {len(tasks_with_code[:13])}")
    print(f"Total Eligible Executable Tokens:  {summary['total_eligible_tokens_scanned']}")
    print(f"Total Effective Edits Tested:      {summary['total_effective_edits_tested']}")
    print(f"Drafts with ANY Successful Repair: {drafts_with_any_successful_repair} / {total_drafts} ({summary['repairable_percentage']}%)")
    print(f"Total Successful Single-Token Fixes: {len(all_successful_repairs)}")
    if all_successful_repairs:
        print("\nSUCCESSFUL REPAIRS DETAILS:")
        for r in all_successful_repairs:
            print(f"  Task {r['task_id']} | Pos {r['position']} | {repr(r['original_token_str'])} -> {repr(r['replacement_token_str'])} | H-Rank: {r['entropy_rank']} | Margin-Rank: {r['margin_rank']}")
    else:
        print("\nNo single-token MDLM replacement converted any failed draft into a passing program.")
    print("=" * 80)
    print(f"Saved artifacts to {out_dir}")


if __name__ == "__main__":
    main()
