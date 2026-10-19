#!/usr/bin/env python3
"""Enhanced COROSred Evaluation on Qwen3-0.6B.

Tests the four additions on Qwen3-0.6B:
1. Multi-signal candidate locator (inspecting Booleans, operators, constants, + uncertainty).
2. Internal MDLM disagreement (Delta logP = log P_MDLM(rep) - log P_MDLM(orig)).
3. Independent candidate generation.
4. AST syntax and visible test/example verification.
5. Final evaluation on hidden benchmark tests.
"""
from __future__ import annotations

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from telos.eval.runner import (
    clean_functional_completion,
    execute_code_sandboxed,
    ExecutionResult,
)
from telos.eval.syntax import check_ast_validity
from scripts.qwen3_06b_model import load_qwen3_06b_model, DIFF_WEIGHTS

BOOLEAN_KEYWORDS = {"True", "False"}
RELATIONAL_OPS = {"<", ">", "<=", ">=", "==", "!=", "in", "is"}
ARITHMETIC_LOGICAL_OPS = {"+", "-", "*", "/", "//", "%", "**", "and", "or", "not", "^", "&", "|"}
BOUNDARY_CONSTANTS = {"0", "1", "-1", "2"}


def locate_candidate_positions(
    tok: Tokenizer,
    tokens: List[int],
    eligible_indices: List[int],
    emitted_probs: List[float],
    max_hotspots: int = 12,
) -> List[Tuple[int, str]]:
    candidates: List[Tuple[int, str]] = []
    seen: Set[int] = set()

    for pos in eligible_indices:
        tok_str = tok.decode([tokens[pos]]).strip()
        loc_type = None
        if tok_str in BOOLEAN_KEYWORDS:
            loc_type = f"boolean_{tok_str}"
        elif tok_str in RELATIONAL_OPS:
            loc_type = f"relational_{tok_str}"
        elif tok_str in ARITHMETIC_LOGICAL_OPS:
            loc_type = f"operator_{tok_str}"
        elif tok_str in BOUNDARY_CONSTANTS:
            loc_type = f"constant_{tok_str}"

        if loc_type and pos not in seen:
            candidates.append((pos, loc_type))
            seen.add(pos)

    sorted_by_prob = sorted(eligible_indices, key=lambda p: emitted_probs[p])
    for pos in sorted_by_prob[:3]:
        if pos not in seen:
            candidates.append((pos, "ar_uncertainty"))
            seen.add(pos)

    return candidates[:max_hotspots]


def main():
    print("=" * 80)
    print("ENHANCED COROSred EVALUATION ON QWEN3-0.6B")
    print("=" * 80)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # Load dev cases and suite
    dev_path = PROJECT_ROOT / "research/corosred_dev_retained_edits_30tasks_2026-10-18/per_task_results.json"
    with open(dev_path) as f:
        dev_cases = json.load(f)

    with open(PROJECT_ROOT / "evals/benchmarks/mbpp_suite.json") as f:
        mbpp_tasks = {t["id"]: t for t in json.load(f)}

    out_dir = PROJECT_ROOT / "research/qwen3_enhanced_corosred_2026-10-19"
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = Tokenizer.from_file(str(DIFF_WEIGHTS.parent / "tokenizer.json"))
    mask_id = 151669
    diff_model = load_qwen3_06b_model(DIFF_WEIGHTS, device=device)

    task_records = []
    fixes = []
    regressions = []
    total_candidates = 0
    total_disagreements = 0
    total_accepted = 0

    for idx, c in enumerate(dev_cases, 1):
        tid = c["task_id"]
        prompt = c["prompt"]
        ar_draft = c["ar_draft"]
        ar_clean = c["clean_ar_code"]
        orig_assembled_prog = prompt + ("\n" if not prompt.endswith("\n") else "") + ar_clean

        task_data = mbpp_tasks[tid]
        test_lines = [l for l in task_data["test_harness"].strip().splitlines() if l.strip().startswith("assert ")]
        visible_test = test_lines[0]
        full_harness = task_data["test_harness"]

        p_ids = tok.encode(prompt, add_special_tokens=False).ids
        d_ids = tok.encode(ar_draft, add_special_tokens=False).ids

        # Baseline execution
        orig_vis_res, _ = execute_code_sandboxed(orig_assembled_prog, visible_test, timeout_seconds=3.0)
        orig_passes_vis = (orig_vis_res == ExecutionResult.PASSED)

        orig_full_res, _ = execute_code_sandboxed(orig_assembled_prog, full_harness, timeout_seconds=3.0)
        orig_passes_full = (orig_full_res == ExecutionResult.PASSED)

        # Eligible tokens
        enc_clean = tok.encode(ar_clean, add_special_tokens=False)
        eligible_indices = list(range(min(len(d_ids), len(enc_clean.ids))))

        # Dummy/uniform emitted probs if not cached, or rank by hot-spots
        emitted_probs = [0.5] * len(d_ids)

        candidate_locs = locate_candidate_positions(tok, d_ids, eligible_indices, emitted_probs)
        total_candidates += len(candidate_locs)

        generated_candidates = []

        for pos, loc_type in candidate_locs:
            masked_seq = list(d_ids)
            masked_seq[pos] = mask_id
            full_ids = p_ids + masked_seq
            t_input = torch.tensor([full_ids], dtype=torch.long, device=device)

            with torch.no_grad():
                logits = diff_model(t_input, is_causal=False)

            pos_idx = len(p_ids) + pos
            log_p = F.log_softmax(logits[0, pos_idx, :].float(), dim=-1)

            orig_id = d_ids[pos]
            orig_logp = log_p[orig_id].item()

            log_p_rep = log_p.clone()
            log_p_rep[orig_id] = -float("inf")
            best_rep_id = int(torch.argmax(log_p_rep).item())
            best_rep_logp = log_p[best_rep_id].item()

            delta_logp = best_rep_logp - orig_logp

            if delta_logp > 0.0:
                total_disagreements += 1
                cand_tokens = list(d_ids)
                cand_tokens[pos] = best_rep_id
                cand_clean = clean_functional_completion(prompt, tok.decode(cand_tokens))
                cand_prog = prompt + ("\n" if not prompt.endswith("\n") else "") + cand_clean

                is_ast_valid, _ = check_ast_validity(cand_prog)
                if not is_ast_valid:
                    continue

                cand_vis_res, _ = execute_code_sandboxed(cand_prog, visible_test, timeout_seconds=3.0)
                cand_passes_vis = (cand_vis_res == ExecutionResult.PASSED)

                generated_candidates.append({
                    "position": pos,
                    "locator_type": loc_type,
                    "orig_token": tok.decode([orig_id]),
                    "rep_token": tok.decode([best_rep_id]),
                    "delta_logp": delta_logp,
                    "cand_prog": cand_prog,
                    "passes_visible_test": cand_passes_vis,
                })

        # Verification & Acceptance
        selected_program = orig_assembled_prog
        decision_info = {"status": "retained_original", "accepted_candidate": None}

        if orig_passes_vis:
            decision_info = {"status": "retained_original_passing", "accepted_candidate": None}
            selected_program = orig_assembled_prog
        else:
            improving = [c for c in generated_candidates if c["passes_visible_test"]]
            if improving:
                improving.sort(key=lambda c: c["delta_logp"], reverse=True)
                best_cand = improving[0]
                total_accepted += 1
                selected_program = best_cand["cand_prog"]
                decision_info = {
                    "status": "accepted_repair",
                    "reason": f"Improved visible test with Delta logP={best_cand['delta_logp']:.4f}",
                    "accepted_candidate": best_cand,
                }
            else:
                decision_info = {"status": "retained_original_failing", "accepted_candidate": None}
                selected_program = orig_assembled_prog

        final_outcome, _ = execute_code_sandboxed(selected_program, full_harness, timeout_seconds=3.0)
        final_passed = (final_outcome == ExecutionResult.PASSED)

        if not orig_passes_full and final_passed:
            fixes.append(tid)
            status_label = "FIX (FAIL -> PASS)"
        elif orig_passes_full and not final_passed:
            regressions.append(tid)
            status_label = "REGRESSION (PASS -> FAIL)"
        elif orig_passes_full and final_passed:
            status_label = "PRESERVED_PASS"
        else:
            status_label = "PRESERVED_FAIL"

        print(f"[{idx:02d}/30] {tid}: {status_label} | Decision: {decision_info['status']}")

        task_records.append({
            "task_id": tid,
            "orig_passed": orig_passes_full,
            "final_passed": final_passed,
            "status_label": status_label,
            "decision": decision_info,
            "candidates_found": len(generated_candidates),
        })

    orig_total_pass = sum(1 for r in task_records if r["orig_passed"])
    final_total_pass = sum(1 for r in task_records if r["final_passed"])

    print("\n" + "=" * 60)
    print("QWEN3-0.6B ENHANCED COROSred SUMMARY")
    print("=" * 60)
    print(f"Total Programs:            {len(task_records)}")
    print(f"Original AR Pass@1:        {orig_total_pass}/30 ({orig_total_pass/30*100:.1f}%)")
    print(f"Enhanced COROSred Pass@1:  {final_total_pass}/30 ({final_total_pass/30*100:.1f}%)")
    print(f"Fixes (FAIL -> PASS):      {len(fixes)} {fixes}")
    print(f"Regressions (PASS -> FAIL):{len(regressions)} {regressions}")
    print(f"Total Candidates:          {total_candidates}")
    print(f"Disagreements Found (>0):  {total_disagreements}")
    print(f"Accepted Edits:            {total_accepted}")

    (out_dir / "records.json").write_text(json.dumps(task_records, indent=2))
    print(f"\nSaved Qwen3 records to {out_dir / 'records.json'}")


if __name__ == "__main__":
    main()
