#!/usr/bin/env python3
"""Evaluate COROSred with CoRe Context Sensitivity Routing & T2M Iterative Remasking.

Hypotheses tested:
1. CoRe (Context-Robust Sensitivity Probing, arXiv:2602.04096):
   - Probes token stability under bidirectional context:
     Sensitivity(x_i) = max_{v != x_i} log P_MDLM(v | x_{\\i}) - log P_MDLM(x_i | x_{\\i})
   - Routes by context instability rather than stale AR generation confidence.
2. T2M (Token-to-Mask Refinement, arXiv:2604.18738):
   - Returns suspect positions to [MASK] state and runs iterative denoising trajectory
     (unmasking highest-confidence positions first, allowing remaining masks to condition on updated context).
3. Verifier Gating:
   - Immediate syntax error rejection.
   - Visible specification test verification (preserves passing drafts; requires failing drafts to improve).
   - Hidden benchmark tests held out exclusively for final reporting.
"""
from __future__ import annotations

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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
from scripts.qwen25_coder_model import (
    load_qwen25_coder_model,
    get_diff_weights_path,
)


def get_mask_token_id(tokenizer_path: Path) -> int:
    tok = Tokenizer.from_file(str(tokenizer_path))
    mask_id = tok.token_to_id("<|mask|>")
    assert mask_id == 151665
    return mask_id


def compute_core_sensitivities(
    model,
    tok: Tokenizer,
    prompt_ids: List[int],
    tokens: List[int],
    eligible_indices: List[int],
    mask_id: int,
    device: str,
) -> List[Dict[str, Any]]:
    """Compute CoRe context sensitivity for each eligible token under MDLM context."""
    sensitivities = []

    for p in eligible_indices:
        masked_seq = list(tokens)
        masked_seq[p] = mask_id
        full_ids = prompt_ids + masked_seq
        t_input = torch.tensor([full_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            logits, _ = model(t_input, is_causal=False)

        pos_idx = len(prompt_ids) + p
        log_p = F.log_softmax(logits[0, pos_idx, :].float(), dim=-1)

        orig_id = tokens[p]
        orig_logp = log_p[orig_id].item()

        log_p_rep = log_p.clone()
        log_p_rep[orig_id] = -float("inf")
        best_rep_id = int(torch.argmax(log_p_rep).item())
        best_rep_logp = log_p[best_rep_id].item()

        delta_logp = best_rep_logp - orig_logp

        sensitivities.append({
            "position": p,
            "orig_id": orig_id,
            "orig_token": tok.decode([orig_id]),
            "best_id": best_rep_id,
            "best_token": tok.decode([best_rep_id]),
            "orig_prob": math.exp(orig_logp),
            "best_prob": math.exp(best_rep_logp),
            "delta_logp": delta_logp,
        })

    # Sort descending by sensitivity / instability
    sensitivities.sort(key=lambda s: s["delta_logp"], reverse=True)
    return sensitivities


def t2m_refine(
    model,
    tok: Tokenizer,
    prompt_ids: List[int],
    tokens: List[int],
    positions_to_remask: List[int],
    mask_id: int,
    device: str,
) -> List[int]:
    """Execute Token-to-Mask (T2M) iterative denoising on suspect positions."""
    current_tokens = list(tokens)
    for p in positions_to_remask:
        current_tokens[p] = mask_id

    remaining = list(positions_to_remask)

    while remaining:
        full_ids = prompt_ids + current_tokens
        t_input = torch.tensor([full_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            logits, _ = model(t_input, is_causal=False)

        # Find most confident prediction among remaining masked positions
        best_pos = None
        best_id = None
        best_conf = -1.0

        for p in remaining:
            pos_idx = len(prompt_ids) + p
            p_dist = F.softmax(logits[0, pos_idx, :].float(), dim=-1)
            pred_id = int(torch.argmax(p_dist).item())
            conf = float(p_dist[pred_id].item())
            if conf > best_conf:
                best_conf = conf
                best_pos = p
                best_id = pred_id

        current_tokens[best_pos] = best_id
        remaining.remove(best_pos)

    return current_tokens


def main():
    print("=" * 80)
    print("COROSred EVALUATION: CoRe CONTEXT SENSITIVITY + T2M DYNAMIC REMASKING")
    print("=" * 80)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # Load paths
    diff_weights = get_diff_weights_path()
    tok = Tokenizer.from_file(str(diff_weights.parent / "tokenizer.json"))
    mask_id = get_mask_token_id(diff_weights.parent / "tokenizer.json")

    # Load benchmark suite
    dev_path = PROJECT_ROOT / "research/corosred_dev_retained_edits_30tasks_2026-10-18/per_task_results.json"
    with open(dev_path) as f:
        dev_cases = json.load(f)

    with open(PROJECT_ROOT / "evals/benchmarks/mbpp_suite.json") as f:
        mbpp_tasks = {t["id"]: t for t in json.load(f)}

    ar_results_path = PROJECT_ROOT / "research/qwen25_coder_corosred_eval_corrected_2026-10-18/ar_eval_results.json"
    with open(ar_results_path) as f:
        ar_results = json.load(f)

    out_dir = PROJECT_ROOT / "research/qwen25_coder_core_t2m_2026-10-19"
    out_dir.mkdir(parents=True, exist_ok=True)

    diff_model = load_qwen25_coder_model(diff_weights, device=device)

    task_records: List[Dict[str, Any]] = []
    fixes = []
    regressions = []
    total_unstable_positions_found = 0
    total_edits_accepted = 0

    print("\nStarting CoRe + T2M Evaluation across 30 tasks...")
    print("-" * 80)

    for idx, ar_r in enumerate(ar_results, 1):
        tid = ar_r["task_id"]
        prompt = ar_r["prompt"]
        ar_tokens = list(ar_r["ar_tokens"])
        eligible = ar_r["eligible_positions"]
        orig_assembled_prog = ar_r["assembled_program"]

        task_data = mbpp_tasks[tid]
        test_lines = [l for l in task_data["test_harness"].strip().splitlines() if l.strip().startswith("assert ")]
        visible_test = test_lines[0]
        full_harness = task_data["test_harness"]

        prompt_ids = tok.encode(prompt, add_special_tokens=False).ids

        # 1. Baseline status
        orig_vis_res, _ = execute_code_sandboxed(orig_assembled_prog, visible_test, timeout_seconds=3.0)
        orig_passes_vis = (orig_vis_res == ExecutionResult.PASSED)
        ar_passed_benchmark = ar_r["passed"]

        # 2. CoRe Probing: Measure context sensitivity across all eligible tokens
        sensitivities = compute_core_sensitivities(diff_model, tok, prompt_ids, ar_tokens, eligible, mask_id, device)
        unstable_positions = [s for s in sensitivities if s["delta_logp"] > 0.0]
        total_unstable_positions_found += len(unstable_positions)

        # 3. T2M Candidate Generation
        generated_candidates = []

        # Candidate Strategy A: Single-token T2M refinement on top unstable positions
        for s in unstable_positions[:3]:
            pos = s["position"]
            refined_tokens = t2m_refine(diff_model, tok, prompt_ids, ar_tokens, [pos], mask_id, device)
            refined_raw = tok.decode(refined_tokens)
            refined_clean = clean_functional_completion(prompt, refined_raw)

            if (
                refined_clean.strip().startswith("def ")
                or refined_clean.strip().startswith("import ")
                or refined_clean.strip().startswith("from ")
            ):
                cand_prog = refined_clean
            else:
                cand_prog = prompt + ("\n" if not prompt.endswith("\n") else "") + refined_clean

            is_ast_valid, _ = check_ast_validity(cand_prog)
            if is_ast_valid:
                cand_vis_res, _ = execute_code_sandboxed(cand_prog, visible_test, timeout_seconds=3.0)
                cand_passes_vis = (cand_vis_res == ExecutionResult.PASSED)
                generated_candidates.append({
                    "strategy": "t2m_single",
                    "positions": [pos],
                    "tokens_changed": f"{s['orig_token']} -> {s['best_token']}",
                    "delta_logp": s["delta_logp"],
                    "cand_prog": cand_prog,
                    "passes_visible_test": cand_passes_vis,
                })

        # Candidate Strategy B: Multi-token T2M joint remasking trajectory on top-2/3 unstable positions
        if len(unstable_positions) >= 2:
            top_positions = [s["position"] for s in unstable_positions[:3]]
            joint_refined = t2m_refine(diff_model, tok, prompt_ids, ar_tokens, top_positions, mask_id, device)
            joint_raw = tok.decode(joint_refined)
            joint_clean = clean_functional_completion(prompt, joint_raw)

            if (
                joint_clean.strip().startswith("def ")
                or joint_clean.strip().startswith("import ")
                or joint_clean.strip().startswith("from ")
            ):
                joint_prog = joint_clean
            else:
                joint_prog = prompt + ("\n" if not prompt.endswith("\n") else "") + joint_clean

            is_ast_valid, _ = check_ast_validity(joint_prog)
            if is_ast_valid:
                joint_vis_res, _ = execute_code_sandboxed(joint_prog, visible_test, timeout_seconds=3.0)
                joint_passes_vis = (joint_vis_res == ExecutionResult.PASSED)
                mean_delta = sum(s["delta_logp"] for s in unstable_positions[:3]) / len(top_positions)
                generated_candidates.append({
                    "strategy": "t2m_multi_step",
                    "positions": top_positions,
                    "tokens_changed": f"{len(top_positions)} positions joint T2M",
                    "delta_logp": mean_delta,
                    "cand_prog": joint_prog,
                    "passes_visible_test": joint_passes_vis,
                })

        # 4. Verifier Gating
        selected_program = orig_assembled_prog
        decision_info = {"status": "retained_original", "accepted_candidate": None}

        if orig_passes_vis:
            # Preserving passing drafts
            decision_info = {"status": "retained_original_passing", "accepted_candidate": None}
            selected_program = orig_assembled_prog
        else:
            # Failing drafts: require candidate that improves visible specification test
            improving = [c for c in generated_candidates if c["passes_visible_test"]]
            if improving:
                improving.sort(key=lambda c: c["delta_logp"], reverse=True)
                best_cand = improving[0]
                total_edits_accepted += 1
                selected_program = best_cand["cand_prog"]
                decision_info = {
                    "status": "accepted_t2m_repair",
                    "reason": f"T2M candidate passed visible test with Delta logP={best_cand['delta_logp']:.4f}",
                    "accepted_candidate": best_cand,
                }
            else:
                decision_info = {"status": "retained_original_failing", "accepted_candidate": None}
                selected_program = orig_assembled_prog

        # 5. Final Evaluation on Hidden Benchmark Tests
        final_outcome, _ = execute_code_sandboxed(selected_program, full_harness, timeout_seconds=3.0)
        final_passed = (final_outcome == ExecutionResult.PASSED)

        if not ar_passed_benchmark and final_passed:
            fixes.append(tid)
            status_label = "FIX (FAIL -> PASS)"
        elif ar_passed_benchmark and not final_passed:
            regressions.append(tid)
            status_label = "REGRESSION (PASS -> FAIL)"
        elif ar_passed_benchmark and final_passed:
            status_label = "PRESERVED_PASS"
        else:
            status_label = "PRESERVED_FAIL"

        print(f"[{idx:02d}/30] {tid}: {status_label} | Unstable positions: {len(unstable_positions)}/{len(eligible)} | Decision: {decision_info['status']}")

        task_records.append({
            "task_id": tid,
            "ar_baseline_passed": ar_passed_benchmark,
            "unstable_positions_count": len(unstable_positions),
            "top_unstable": unstable_positions[:3],
            "decision": decision_info,
            "final_passed": final_passed,
            "status_label": status_label,
        })

    final_pass_count = sum(1 for r in task_records if r["final_passed"])
    print("\n" + "=" * 60)
    print("CoRe + T2M COROSred EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total Programs:               {len(task_records)}")
    print(f"Native AR Baseline Pass@1:    15/30 (50.0%)")
    print(f"CoRe + T2M Pass@1:            {final_pass_count}/30 ({final_pass_count / len(task_records) * 100:.1f}%)")
    print(f"Fixes (FAIL -> PASS):         {len(fixes)} {fixes}")
    print(f"Regressions (PASS -> FAIL):   {len(regressions)} {regressions}")
    print(f"Total Unstable Tokens Found:  {total_unstable_positions_found}")
    print(f"Total Edits Accepted:         {total_edits_accepted}")

    out_records_path = out_dir / "records.json"
    out_records_path.write_text(json.dumps(task_records, indent=2))
    print(f"\nSaved CoRe + T2M records to {out_records_path}")

    summary = {
        "model_family": "Qwen2.5-Coder-0.5B-Instruct",
        "benchmark": "MBPP (30 Tasks)",
        "timestamp": int(time.time()),
        "ar_baseline": {"pass_at_1_pct": 50.0, "pass_count": 15},
        "core_t2m_corosred": {
            "pass_at_1_pct": round(final_pass_count / len(task_records) * 100, 1),
            "pass_count": final_pass_count,
            "fixes": fixes,
            "regressions": regressions,
            "unstable_tokens_found": total_unstable_positions_found,
            "accepted_edits": total_edits_accepted,
        },
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
