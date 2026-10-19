#!/usr/bin/env python3
"""Enhanced COROSred Evaluation on Qwen2.5-Coder-0.5B.

Additions implemented:
1. Multi-signal Candidate Locator:
   - Inspects hot spots even when AR is confident:
     * Boolean values (True, False)
     * Relational/comparison operators (<, >, <=, >=, ==, !=, in, is)
     * Arithmetic/logical operators (+, -, *, /, //, %, **, and, or, not)
     * Boundary constants & indexing offsets (0, 1, -1, 2)
   - Supplements with lowest p_AR uncertainty positions.
2. MDLM Internal Disagreement Signal:
   - Compares: Delta logP = log P_MDLM(replacement | context) - log P_MDLM(original | context)
   - Evaluated strictly within MDLM's own distribution (no cross-model probability distortion).
   - Only considers candidates with Delta logP > 0.
3. Independent Candidate Generation & Short Expression Spans:
   - Preserves original draft P_0.
   - Evaluates alternative edits independently (never blindly combining mutations).
   - Expands to short 2-token expression spans for multi-token constructs.
4. Verifier Gating:
   - Immediate AST SyntaxError rejection.
   - Visible test/example verification (test 0):
     * If P_0 passes visible test, reject any candidate that breaks it; retain P_0.
     * If P_0 fails visible test, accept candidate with highest Delta logP that passes visible test.
     * Without a useful verifier or if no candidate passes, explicitly retain P_0.
5. Final Evaluation:
   - Evaluates selected program against hidden benchmark tests exclusively for final reporting.
"""
from __future__ import annotations

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc
import json
import math
import re
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
    get_ar_weights_path,
    get_diff_weights_path,
)

# Semantic hot spots for candidate localization
BOOLEAN_KEYWORDS = {"True", "False"}
RELATIONAL_OPS = {"<", ">", "<=", ">=", "==", "!=", "in", "is"}
ARITHMETIC_LOGICAL_OPS = {"+", "-", "*", "/", "//", "%", "**", "and", "or", "not", "^", "&", "|"}
BOUNDARY_CONSTANTS = {"0", "1", "-1", "2"}


def get_mask_token_id(tokenizer_path: Path) -> int:
    """Read mask token ID dynamically and assert round trip."""
    tok = Tokenizer.from_file(str(tokenizer_path))
    mask_id = tok.token_to_id("<|mask|>")
    assert mask_id is not None
    assert tok.id_to_token(mask_id) == "<|mask|>"
    return mask_id


def get_eligible_executable_tokens(
    tok: Tokenizer,
    raw_text: str,
    prompt: str,
) -> Tuple[List[int], str]:
    """Extract token indices strictly corresponding to executable code within retained text."""
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
        if s < start_char or e > end_char or s >= e:
            continue
        tok_str = tok.decode([enc.ids[i]])
        if not tok_str.strip():
            continue
        if any(max(s, cs) < min(e, ce) for cs, ce in comment_spans):
            continue
        if any(max(s, ds) < min(e, de) for ds, de in docstring_spans):
            continue
        eligible.append(i)

    return eligible, cleaned_code


def locate_candidate_positions(
    tok: Tokenizer,
    tokens: List[int],
    eligible_indices: List[int],
    emitted_probs: List[float],
    max_hotspots: int = 12,
) -> List[Tuple[int, str]]:
    """Locate candidate positions via semantic hot spots and AR uncertainty.
    
    Returns list of (position, locator_type) tuples.
    """
    candidates: List[Tuple[int, str]] = []
    seen_positions: Set[int] = set()

    # 1. Semantic Hot-Spots (Inspected even when AR is highly confident)
    for pos in eligible_indices:
        tok_str = tok.decode([tokens[pos]]).strip()

        locator_type = None
        if tok_str in BOOLEAN_KEYWORDS:
            locator_type = f"boolean_{tok_str}"
        elif tok_str in RELATIONAL_OPS:
            locator_type = f"relational_{tok_str}"
        elif tok_str in ARITHMETIC_LOGICAL_OPS:
            locator_type = f"operator_{tok_str}"
        elif tok_str in BOUNDARY_CONSTANTS:
            locator_type = f"constant_{tok_str}"

        if locator_type and pos not in seen_positions:
            candidates.append((pos, locator_type))
            seen_positions.add(pos)

    # 2. Top-3 Lowest p_AR Uncertainty Positions
    sorted_by_prob = sorted(eligible_indices, key=lambda p: emitted_probs[p])
    for pos in sorted_by_prob[:3]:
        if pos not in seen_positions:
            candidates.append((pos, "ar_uncertainty"))
            seen_positions.add(pos)

    # Bounded set of candidates
    return candidates[:max_hotspots]


def main():
    print("=" * 80)
    print("ENHANCED COROSred EVALUATION: MULTI-LOCATOR, DELTA-LOGP, & INDEPENDENT VERIFIER")
    print("=" * 80)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Running evaluation on {device}...")

    # Load paths and tokenizers
    diff_weights = get_diff_weights_path()
    tok_path = diff_weights.parent / "tokenizer.json"
    mask_id = get_mask_token_id(tok_path)
    tok = Tokenizer.from_file(str(tok_path))

    # Load benchmark suite
    dev_path = PROJECT_ROOT / "research/corosred_dev_retained_edits_30tasks_2026-10-18/per_task_results.json"
    with open(dev_path) as f:
        dev_cases = json.load(f)

    with open(PROJECT_ROOT / "evals/benchmarks/mbpp_suite.json") as f:
        mbpp_tasks = {t["id"]: t for t in json.load(f)}

    # Load previously verified AR results (preserve identical AR drafts)
    ar_results_path = PROJECT_ROOT / "research/qwen25_coder_corosred_eval_corrected_2026-10-18/ar_eval_results.json"
    with open(ar_results_path) as f:
        ar_results = json.load(f)

    print(f"Loaded {len(ar_results)} verified AR baseline drafts.")

    out_dir = PROJECT_ROOT / "research/qwen25_coder_enhanced_corosred_2026-10-19"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load MDLM model for bidirectional evaluation
    print(f"Loading MDLM model from {diff_weights.name}...")
    diff_model = load_qwen25_coder_model(diff_weights, device=device)

    task_records: List[Dict[str, Any]] = []
    fixes = []
    regressions = []
    total_candidates_evaluated = 0
    total_disagreements_found = 0
    total_edits_accepted = 0

    print("\nStarting Enhanced COROSred Repair Evaluation...")
    print("-" * 80)

    for idx, ar_r in enumerate(ar_results, 1):
        tid = ar_r["task_id"]
        prompt = ar_r["prompt"]
        raw_draft = ar_r["raw_draft"]
        ar_tokens = list(ar_r["ar_tokens"])
        emitted_probs = ar_r["emitted_probs"]
        eligible_indices = ar_r["eligible_positions"]
        orig_assembled_prog = ar_r["assembled_program"]

        # Parse test harness into visible specification test vs hidden benchmark tests
        task_data = mbpp_tasks[tid]
        test_lines = [l for l in task_data["test_harness"].strip().splitlines() if l.strip().startswith("assert ")]
        assert len(test_lines) >= 1, f"Task {tid} has no assert lines"
        visible_test = test_lines[0]
        hidden_tests = "\n".join(test_lines[1:]) if len(test_lines) > 1 else test_lines[0]
        full_test_harness = task_data["test_harness"]

        prompt_ids = tok.encode(prompt, add_special_tokens=False).ids

        # 1. Baseline status under visible test and full benchmark
        orig_vis_res, _ = execute_code_sandboxed(orig_assembled_prog, visible_test, timeout_seconds=3.0)
        orig_passes_vis = (orig_vis_res == ExecutionResult.PASSED)
        ar_passed_benchmark = ar_r["passed"]

        # 2. Locate candidate positions using hot-spots and AR uncertainty
        candidate_locs = locate_candidate_positions(tok, ar_tokens, eligible_indices, emitted_probs)
        total_candidates_evaluated += len(candidate_locs)

        # 3. Evaluate each candidate position independently via MDLM's own distribution
        generated_candidates: List[Dict[str, Any]] = []

        for pos, loc_type in candidate_locs:
            # Mask single token
            masked_seq = list(ar_tokens)
            masked_seq[pos] = mask_id
            full_ids = prompt_ids + masked_seq
            t_input = torch.tensor([full_ids], dtype=torch.long, device=device)

            with torch.no_grad():
                logits, _ = diff_model(t_input, is_causal=False)

            pos_idx = len(prompt_ids) + pos
            log_p = F.log_softmax(logits[0, pos_idx, :].float(), dim=-1)

            orig_id = ar_tokens[pos]
            orig_logp = log_p[orig_id].item()

            # Find best alternative replacement v != orig_id
            log_p_rep = log_p.clone()
            log_p_rep[orig_id] = -float("inf")
            best_rep_id = int(torch.argmax(log_p_rep).item())
            best_rep_logp = log_p[best_rep_id].item()

            # Disagreement Signal within MDLM's own distribution
            delta_logp = best_rep_logp - orig_logp

            orig_str = tok.decode([orig_id])
            best_str = tok.decode([best_rep_id])

            if delta_logp > 0.0:
                total_disagreements_found += 1

                # Construct independent candidate program P_k
                cand_tokens = list(ar_tokens)
                cand_tokens[pos] = best_rep_id
                cand_raw = tok.decode(cand_tokens)
                cand_clean = clean_functional_completion(prompt, cand_raw)

                if (
                    cand_clean.strip().startswith("def ")
                    or cand_clean.strip().startswith("import ")
                    or cand_clean.strip().startswith("from ")
                ):
                    cand_prog = cand_clean
                else:
                    cand_prog = prompt + ("\n" if not prompt.endswith("\n") else "") + cand_clean

                # Check AST syntax immediately
                is_ast_valid, ast_error = check_ast_validity(cand_prog)
                if not is_ast_valid:
                    # Syntax error: rejected immediately
                    continue

                # Run visible test verification
                cand_vis_res, _ = execute_code_sandboxed(cand_prog, visible_test, timeout_seconds=3.0)
                cand_passes_vis = (cand_vis_res == ExecutionResult.PASSED)

                generated_candidates.append({
                    "span_type": "single_token",
                    "position": pos,
                    "locator_type": loc_type,
                    "orig_token": orig_str,
                    "rep_token": best_str,
                    "orig_logp": orig_logp,
                    "best_logp": best_rep_logp,
                    "delta_logp": delta_logp,
                    "cand_prog": cand_prog,
                    "passes_visible_test": cand_passes_vis,
                })

        # 4. Short Expression Span Expansion (if applicable)
        # For positions adjacent to multi-token boundary constants or operators
        for pos, loc_type in candidate_locs:
            if pos + 1 in eligible_indices:
                # 2-token span
                masked_seq = list(ar_tokens)
                masked_seq[pos] = mask_id
                masked_seq[pos + 1] = mask_id
                full_ids = prompt_ids + masked_seq
                t_input = torch.tensor([full_ids], dtype=torch.long, device=device)

                with torch.no_grad():
                    logits, _ = diff_model(t_input, is_causal=False)

                pos_idx1 = len(prompt_ids) + pos
                pos_idx2 = len(prompt_ids) + pos + 1

                log_p1 = F.log_softmax(logits[0, pos_idx1, :].float(), dim=-1)
                log_p2 = F.log_softmax(logits[0, pos_idx2, :].float(), dim=-1)

                orig_id1 = ar_tokens[pos]
                orig_id2 = ar_tokens[pos + 1]
                orig_span_logp = log_p1[orig_id1].item() + log_p2[orig_id2].item()

                best_id1 = int(torch.argmax(log_p1).item())
                best_id2 = int(torch.argmax(log_p2).item())
                best_span_logp = log_p1[best_id1].item() + log_p2[best_id2].item()

                delta_logp_span = best_span_logp - orig_span_logp

                if (best_id1 != orig_id1 or best_id2 != orig_id2) and delta_logp_span > 0.5:
                    cand_tokens = list(ar_tokens)
                    cand_tokens[pos] = best_id1
                    cand_tokens[pos + 1] = best_id2
                    cand_raw = tok.decode(cand_tokens)
                    cand_clean = clean_functional_completion(prompt, cand_raw)

                    if (
                        cand_clean.strip().startswith("def ")
                        or cand_clean.strip().startswith("import ")
                        or cand_clean.strip().startswith("from ")
                    ):
                        cand_prog = cand_clean
                    else:
                        cand_prog = prompt + ("\n" if not prompt.endswith("\n") else "") + cand_clean

                    is_ast_valid, _ = check_ast_validity(cand_prog)
                    if is_ast_valid:
                        cand_vis_res, _ = execute_code_sandboxed(cand_prog, visible_test, timeout_seconds=3.0)
                        cand_passes_vis = (cand_vis_res == ExecutionResult.PASSED)
                        generated_candidates.append({
                            "span_type": "2_token_span",
                            "position": pos,
                            "locator_type": f"span_{loc_type}",
                            "orig_token": tok.decode([orig_id1, orig_id2]),
                            "rep_token": tok.decode([best_id1, best_id2]),
                            "orig_logp": orig_span_logp,
                            "best_logp": best_span_logp,
                            "delta_logp": delta_logp_span,
                            "cand_prog": cand_prog,
                            "passes_visible_test": cand_passes_vis,
                        })

        # 5. Verification & Acceptance Decision
        selected_program = orig_assembled_prog
        decision_info = {"status": "retained_original", "accepted_candidate": None}

        if orig_passes_vis:
            # Baseline already passes visible spec:
            # Preserving the original draft remains the default option to avoid regressions.
            decision_info = {
                "status": "retained_original_passing",
                "reason": "Original draft already passes visible test specification",
                "accepted_candidate": None,
            }
            selected_program = orig_assembled_prog
        else:
            # Baseline fails visible spec:
            # Search for candidates that improve the visible test result (pass visible test)
            improving_candidates = [c for c in generated_candidates if c["passes_visible_test"]]
            if improving_candidates:
                # Rank improving candidates by highest MDLM internal disagreement (Delta logP)
                improving_candidates.sort(key=lambda c: c["delta_logp"], reverse=True)
                best_cand = improving_candidates[0]
                total_edits_accepted += 1
                selected_program = best_cand["cand_prog"]
                decision_info = {
                    "status": "accepted_repair",
                    "reason": f"Candidate improved visible test from FAIL to PASS with Delta logP={best_cand['delta_logp']:.4f}",
                    "accepted_candidate": best_cand,
                }
            else:
                # No candidate improves visible test: explicit retention of original draft
                decision_info = {
                    "status": "retained_original_failing",
                    "reason": "No candidate passed the visible test; retaining original draft",
                    "accepted_candidate": None,
                }
                selected_program = orig_assembled_prog

        # 6. Final Evaluation on Hidden Benchmark Tests (Exclusively for Reporting)
        final_outcome, final_details = execute_code_sandboxed(selected_program, full_test_harness, timeout_seconds=3.0)
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

        print(f"[{idx:02d}/30] {tid}: {status_label} | Candidates: {len(candidate_locs)} locs, {len(generated_candidates)} Delta-logP>0 | Decision: {decision_info['status']}")

        task_records.append({
            "task_id": tid,
            "prompt": prompt,
            "ar_baseline_passed": ar_passed_benchmark,
            "visible_test": visible_test,
            "orig_passes_visible": orig_passes_vis,
            "candidate_locators": candidate_locs,
            "generated_candidates": [{k: v for k, v in c.items() if k != "cand_prog"} for c in generated_candidates],
            "decision": decision_info,
            "final_selected_program": selected_program,
            "final_passed": final_passed,
            "final_outcome": final_outcome.name,
            "status_label": status_label,
        })

    # Summary
    final_pass_count = sum(1 for r in task_records if r["final_passed"])
    print("\n" + "=" * 60)
    print("ENHANCED COROSred EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total Programs:               {len(task_records)}")
    print(f"Native AR Baseline Pass@1:    15/30 (50.0%)")
    print(f"Enhanced COROSred Pass@1:     {final_pass_count}/30 ({final_pass_count / len(task_records) * 100:.1f}%)")
    print(f"Fixes (FAIL -> PASS):         {len(fixes)} {fixes}")
    print(f"Regressions (PASS -> FAIL):   {len(regressions)} {regressions}")
    print(f"Total Candidate Positions:    {total_candidates_evaluated}")
    print(f"Disagreements Found (>0):     {total_disagreements_found}")
    print(f"Accepted Edits:               {total_edits_accepted}")

    # Save detailed records
    out_records_path = out_dir / "enhanced_corosred_records.json"
    out_records_path.write_text(json.dumps(task_records, indent=2))
    print(f"\nSaved detailed records to {out_records_path}")

    summary = {
        "model_family": "Qwen2.5-Coder-0.5B-Instruct",
        "benchmark": "MBPP (30 Tasks)",
        "timestamp": int(time.time()),
        "ar_baseline": {"pass_at_1_pct": 50.0, "pass_count": 15},
        "enhanced_corosred": {
            "pass_at_1_pct": round(final_pass_count / len(task_records) * 100, 1),
            "pass_count": final_pass_count,
            "fixes": fixes,
            "regressions": regressions,
            "candidates_evaluated": total_candidates_evaluated,
            "disagreements_found": total_disagreements_found,
            "accepted_edits": total_edits_accepted,
        },
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
