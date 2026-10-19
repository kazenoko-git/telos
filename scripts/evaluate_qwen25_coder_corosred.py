#!/usr/bin/env python3
"""Evaluate Qwen2.5-Coder-0.5B-Instruct: Native AR vs Native MDLM vs COROSred.

Benchmarks:
1. Native AR Baseline: Greedy causal generation with Qwen2.5-Coder-0.5B-Instruct.
2. Native MDLM Baseline: Documented native decoder from dLLM repository with
   dllm-hub/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1, preserving committed tokens
   and handling every output position.
3. COROSred Routed Repair:
   - AR draft generation with token probability tracking.
   - Retained executable code token identification (excluding comments, docstrings, whitespace).
   - Lowest p_AR uncertainty ranking.
   - Bidirectional MDLM candidate evaluation at candidate positions using the verified mask ID.
   - Acceptance Gate: p_MDLM >= 0.50 AND p_MDLM > p_AR AND y_MDLM != y_orig.
   - Sandboxed execution against unit tests.

Validations enforced:
- Mask ID read dynamically from diffusion tokenizer and asserted for round-trip.
- Exact assembled program validated by AST check and executed by sandbox.
- Unpacked boolean AST check outcomes.
- Complete per-task records saved to disk.
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
from typing import Any, Dict, List, Optional, Tuple

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

DEFAULT_STOP_STRINGS = [
    "\n\n\n",
    "\ndef ",
    "\nclass ",
    "\nif __name__",
    "\nprint(",
    "<|im_end|>",
    "<|endoftext|>",
]


def get_mask_token_id(tokenizer_path: Path) -> int:
    """Read mask token ID dynamically from the diffusion tokenizer and assert round-trip."""
    tok = Tokenizer.from_file(str(tokenizer_path))
    mask_id = tok.token_to_id("<|mask|>")
    assert mask_id is not None, "Failed to resolve <|mask|> token ID from tokenizer."
    recovered_token = tok.id_to_token(mask_id)
    assert recovered_token == "<|mask|>", (
        f"Mask token round-trip failed: token_to_id gives {mask_id}, but id_to_token gives {recovered_token}"
    )
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


def generate_ar_draft(
    model,
    tok: Tokenizer,
    prompt: str,
    max_new_tokens: int = 160,
    device: str = "cpu",
) -> Tuple[str, List[int], List[float], float]:
    """Generate greedy causal AR completion and record emitted token probabilities."""
    prompt_ids = tok.encode(prompt, add_special_tokens=False).ids
    input_ids = list(prompt_ids)

    generated_tokens: List[int] = []
    emitted_probs: List[float] = []

    eos_token_id = tok.token_to_id("<|im_end|>")
    endoftext_id = tok.token_to_id("<|endoftext|>")
    stop_ids = {eos_token_id, endoftext_id}

    t0 = time.monotonic()
    curr_input = torch.tensor([input_ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits, _ = model(curr_input, is_causal=True)

        next_logits = logits[0, -1, :].float()
        probs = F.softmax(next_logits, dim=-1)
        next_token = int(torch.argmax(next_logits).item())
        prob = float(probs[next_token].item())

        if next_token in stop_ids:
            break

        generated_tokens.append(next_token)
        emitted_probs.append(prob)
        curr_input = torch.cat(
            [curr_input, torch.tensor([[next_token]], dtype=torch.long, device=device)],
            dim=1,
        )

        # Early stop on newline block boundaries
        decoded_so_far = tok.decode(generated_tokens)
        if any(stop_s in decoded_so_far for stop_s in DEFAULT_STOP_STRINGS):
            break

    latency = time.monotonic() - t0
    raw_completion = tok.decode(generated_tokens)
    return raw_completion, generated_tokens, emitted_probs, latency


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """Documented native transfer schedule from official dLLM implementation."""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer[i, :remainder[i]] += 1
    return num_transfer


@torch.no_grad()
def generate_native_mdlm(
    model,
    tok: Tokenizer,
    prompt: str,
    mask_id: int,
    pad_id: int,
    max_new_tokens: int = 128,
    block_size: int = 64,
    steps: int = 128,
    device: str = "cpu",
) -> Tuple[str, float]:
    """Documented native MDLM decoder preserving committed tokens and handling every position."""
    prompt_ids = tok.encode(prompt, add_special_tokens=False).ids
    prompt_len = len(prompt_ids)

    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    prompt_lens = torch.tensor([prompt_len], dtype=torch.long, device=device)

    batch_size = 1
    total_length = prompt_len + max_new_tokens
    x = torch.full((batch_size, total_length), pad_id, dtype=torch.long, device=device)
    x[0, :prompt_len] = prompt_tensor[0, :prompt_len]
    x[0, prompt_len:total_length] = mask_id

    positions = torch.arange(total_length, device=device)
    assert max_new_tokens % block_size == 0, "max_new_tokens must be divisible by block_size"
    num_blocks = max_new_tokens // block_size
    assert steps % num_blocks == 0, "steps must be divisible by num_blocks"
    steps_per_block = steps // num_blocks

    t0 = time.monotonic()

    for num_block in range(num_blocks):
        block_start = prompt_lens + num_block * block_size
        block_end = block_start + block_size

        init_block_mask = (
            (positions.unsqueeze(0) >= block_start.unsqueeze(1))
            & (positions.unsqueeze(0) < block_end.unsqueeze(1))
            & (x == mask_id)
        )
        num_transfer = get_num_transfer_tokens(init_block_mask, steps_per_block)

        for s in range(steps_per_block):
            # Only uncommitted (masked) positions within current block are eligible
            block_mask = (
                (positions.unsqueeze(0) >= block_start.unsqueeze(1))
                & (positions.unsqueeze(0) < block_end.unsqueeze(1))
                & (x == mask_id)
            )

            logits, _ = model(x, is_causal=False)
            logits = logits.float()
            x0 = torch.argmax(logits, dim=-1)

            p = F.softmax(logits, dim=-1)
            x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)

            confidence = torch.full_like(x0_p, -float("inf"))
            confidence = torch.where(block_mask, x0_p, confidence)
            x0 = torch.where(block_mask, x0, x)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=device)
            k = int(num_transfer[0, s].item())
            if k > 0:
                _, select_index = torch.topk(confidence[0], k=k)
                transfer_index[0, select_index] = True
            x[transfer_index] = x0[transfer_index]

        # Handle every output position: fallback fill any remaining masked positions
        block_remain = (
            (positions.unsqueeze(0) >= block_start.unsqueeze(1))
            & (positions.unsqueeze(0) < block_end.unsqueeze(1))
            & (x == mask_id)
        )
        if block_remain.any():
            logits, _ = model(x, is_causal=False)
            x0 = torch.argmax(logits.float(), dim=-1)
            x = torch.where(block_remain, x0, x)

    # Assert no mask token remains in the output
    assert not (x[0, prompt_len:] == mask_id).any(), "Unfilled mask token found in generated output!"

    latency = time.monotonic() - t0
    generated_tokens = x[0, prompt_len:].tolist()
    raw_completion = tok.decode(generated_tokens)
    return raw_completion, latency


def main():
    print("=" * 80)
    print("CORRECTED EVALUATION: QWEN 2.5 CODER (AR vs MDLM vs COROSred)")
    print("=" * 80)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Running evaluation on {device}...")

    # File paths
    ar_weights = get_ar_weights_path()
    diff_weights = get_diff_weights_path()
    tokenizer_path = diff_weights.parent / "tokenizer.json"

    # 1. Resolve and validate mask token ID
    mask_id = get_mask_token_id(tokenizer_path)
    print(f"Validated Diffusion Mask Token ID: {mask_id}")

    tok = Tokenizer.from_file(str(tokenizer_path))
    pad_id = tok.token_to_id("<|endoftext|>")

    # Load benchmark suite
    dev_path = PROJECT_ROOT / "research/corosred_dev_retained_edits_30tasks_2026-10-18/per_task_results.json"
    with open(dev_path) as f:
        dev_cases = json.load(f)

    with open(PROJECT_ROOT / "evals/benchmarks/mbpp_suite.json") as f:
        mbpp_tests = {t["id"]: t["test_harness"] for t in json.load(f)}

    print(f"Loaded {len(dev_cases)} MBPP benchmark tasks.")

    out_dir = PROJECT_ROOT / "research/qwen25_coder_corosred_eval_corrected_2026-10-18"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Phase 1: Native AR Baseline
    # ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("1. RUNNING NATIVE AR BASELINE (Qwen2.5-Coder-0.5B-Instruct)")
    print("=" * 60)

    ar_model = load_qwen25_coder_model(ar_weights, device=device)
    ar_results = []

    for idx, case in enumerate(dev_cases, 1):
        tid = case["task_id"]
        prompt = case["prompt"]
        test_harness = mbpp_tests[tid]

        draft_text, ar_tokens, emitted_probs, latency = generate_ar_draft(
            ar_model, tok, prompt, max_new_tokens=160, device=device
        )
        clean_code = clean_functional_completion(prompt, draft_text)

        # Validate exact assembled program
        prog = prompt + ("\n" if not prompt.endswith("\n") else "") + clean_code
        is_ast_valid, ast_error = check_ast_validity(prog)
        assert isinstance(is_ast_valid, bool), "check_ast_validity must return a boolean"

        outcome, details = execute_code_sandboxed(prog, test_harness, timeout_seconds=3.0)
        passed = (outcome == ExecutionResult.PASSED)

        eligible_indices, _ = get_eligible_executable_tokens(tok, draft_text, prompt)

        ar_results.append({
            "task_id": tid,
            "prompt": prompt,
            "raw_draft": draft_text,
            "ar_tokens": ar_tokens,
            "emitted_probs": emitted_probs,
            "clean_code": clean_code,
            "assembled_program": prog,
            "ast_valid": is_ast_valid,
            "ast_error": ast_error,
            "passed": passed,
            "outcome": outcome.name,
            "details": details,
            "latency_seconds": latency,
            "eligible_positions": eligible_indices,
        })
        status_str = "PASS" if passed else f"FAIL ({outcome.name})"
        print(f"[{idx:02d}/30] {tid}: AR {status_str} in {latency:.2f}s (AST: {is_ast_valid}, Eligible: {len(eligible_indices)})")

    ar_pass_count = sum(1 for r in ar_results if r["passed"])
    ar_ast_count = sum(1 for r in ar_results if r["ast_valid"])
    print(f"\nAR Baseline Pass@1: {ar_pass_count}/30 ({ar_pass_count/30*100:.1f}%), AST Validity: {ar_ast_count}/30 ({ar_ast_count/30*100:.1f}%)")

    (out_dir / "ar_eval_results.json").write_text(json.dumps(ar_results, indent=2))
    print(f"Saved AR per-task records to {out_dir / 'ar_eval_results.json'}")

    del ar_model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()

    # ------------------------------------------------------------
    # Phase 2: Native MDLM Baseline
    # ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("2. RUNNING NATIVE MDLM BASELINE (Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1)")
    print("=" * 60)

    diff_model = load_qwen25_coder_model(diff_weights, device=device)
    mdlm_results = []

    for idx, case in enumerate(dev_cases, 1):
        tid = case["task_id"]
        prompt = case["prompt"]
        test_harness = mbpp_tests[tid]

        raw_mdlm, latency = generate_native_mdlm(
            diff_model, tok, prompt, mask_id=mask_id, pad_id=pad_id,
            max_new_tokens=128, block_size=64, steps=128, device=device
        )
        clean_code = clean_functional_completion(prompt, raw_mdlm)

        prog = prompt + ("\n" if not prompt.endswith("\n") else "") + clean_code
        is_ast_valid, ast_error = check_ast_validity(prog)
        assert isinstance(is_ast_valid, bool), "check_ast_validity must return a boolean"

        outcome, details = execute_code_sandboxed(prog, test_harness, timeout_seconds=3.0)
        passed = (outcome == ExecutionResult.PASSED)

        mdlm_results.append({
            "task_id": tid,
            "prompt": prompt,
            "raw_completion": raw_mdlm,
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
        print(f"[{idx:02d}/30] {tid}: MDLM {status_str} in {latency:.2f}s (AST: {is_ast_valid})")

    mdlm_pass_count = sum(1 for r in mdlm_results if r["passed"])
    mdlm_ast_count = sum(1 for r in mdlm_results if r["ast_valid"])
    print(f"\nMDLM Baseline Pass@1: {mdlm_pass_count}/30 ({mdlm_pass_count/30*100:.1f}%), AST Validity: {mdlm_ast_count}/30 ({mdlm_ast_count/30*100:.1f}%)")

    (out_dir / "mdlm_eval_results.json").write_text(json.dumps(mdlm_results, indent=2))
    print(f"Saved MDLM per-task records to {out_dir / 'mdlm_eval_results.json'}")

    # ------------------------------------------------------------
    # Phase 3: COROSred Routed Repair (Rank 1 and Budget k=3)
    # ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("3. RUNNING COROSred ROUTED REPAIR (Rank 1 & Budget k=3)")
    print("=" * 60)

    policies = [
        ("Rank 1 (Lowest p_AR)", 1, "corosred_rank1_results.json"),
        ("Budget k=3 (Lowest p_AR)", 3, "corosred_budget3_results.json"),
    ]

    policy_summaries = {}

    for pol_name, budget_k, file_name in policies:
        print(f"\n--- Running COROSred Policy: {pol_name} ---")
        coros_records = []
        fixes = []
        regressions = []
        accepted_edits_total = 0
        evaluated_positions_total = 0

        for idx, ar_r in enumerate(ar_results, 1):
            tid = ar_r["task_id"]
            prompt = ar_r["prompt"]
            test_harness = mbpp_tests[tid]
            ar_passed = ar_r["passed"]
            ar_tokens = list(ar_r["ar_tokens"])
            emitted_probs = ar_r["emitted_probs"]
            eligible = ar_r["eligible_positions"]

            prompt_ids = tok.encode(prompt, add_special_tokens=False).ids

            if not eligible:
                coros_records.append({
                    "task_id": tid,
                    "prompt": prompt,
                    "ar_passed": ar_passed,
                    "selected_positions": [],
                    "decisions": [],
                    "accepted_edits_count": 0,
                    "assembled_program": ar_r["assembled_program"],
                    "ast_valid": ar_r["ast_valid"],
                    "ast_error": ar_r["ast_error"],
                    "outcome": ar_r["outcome"],
                    "passed": ar_passed,
                    "status": "UNCHANGED_PASS" if ar_passed else "UNCHANGED_FAIL",
                    "latency_seconds": ar_r["latency_seconds"],
                })
                continue

            # Router: sort eligible tokens by lowest emitted p_AR
            sorted_by_prob = sorted(eligible, key=lambda p: emitted_probs[p])
            selected_positions = sorted_by_prob[:budget_k]

            # Infill masked positions
            repaired_seq = list(ar_tokens)
            for p in selected_positions:
                repaired_seq[p] = mask_id

            full_seq = prompt_ids + repaired_seq
            t_full = torch.tensor([full_seq], dtype=torch.long, device=device)

            t0_repair = time.monotonic()
            with torch.no_grad():
                diff_logits, _ = diff_model(t_full, is_causal=False)
            repair_latency = time.monotonic() - t0_repair

            decisions = []
            accepted_edits_task = []
            final_repaired_tokens = list(ar_tokens)

            for p in selected_positions:
                evaluated_positions_total += 1
                pos_idx = len(prompt_ids) + p
                l = diff_logits[0, pos_idx, :].float()
                p_dist = F.softmax(l, dim=-1)
                pred_id = int(l.argmax().item())
                p_mdlm = float(p_dist[pred_id].item())
                pred_str = tok.decode([pred_id])

                orig_id = ar_tokens[p]
                orig_str = tok.decode([orig_id])
                p_ar = emitted_probs[p]

                # Confidence Gate: p_MDLM >= 0.50 AND p_MDLM > p_AR AND pred_id != orig_id
                is_noop = (pred_id == orig_id)
                meets_floor = (p_mdlm >= 0.50)
                beats_ar = (p_mdlm > p_ar)
                accepted = (not is_noop) and meets_floor and beats_ar

                if is_noop:
                    reason = "natural_noop"
                elif not meets_floor:
                    reason = "below_floor_0.50"
                elif not beats_ar:
                    reason = "below_ar_prob"
                else:
                    reason = "accepted"

                decisions.append({
                    "pos": p,
                    "orig_token_id": orig_id,
                    "orig_token_str": orig_str,
                    "p_ar": p_ar,
                    "pred_token_id": pred_id,
                    "pred_token_str": pred_str,
                    "p_mdlm": p_mdlm,
                    "accepted": accepted,
                    "rejection_reason": reason,
                })

                if accepted:
                    accepted_edits_total += 1
                    accepted_edits_task.append(p)
                    final_repaired_tokens[p] = pred_id

            if accepted_edits_task:
                repaired_raw = tok.decode(final_repaired_tokens)
                repaired_clean = clean_functional_completion(prompt, repaired_raw)
                prog = prompt + ("\n" if not prompt.endswith("\n") else "") + repaired_clean

                is_ast_valid, ast_error = check_ast_validity(prog)
                outcome, details = execute_code_sandboxed(prog, test_harness, timeout_seconds=3.0)
                repaired_passed = (outcome == ExecutionResult.PASSED)

                if not ar_passed and repaired_passed:
                    fixes.append(tid)
                    status_str = "FIX (FAIL -> PASS)"
                elif ar_passed and not repaired_passed:
                    regressions.append(tid)
                    status_str = "REGRESSION (PASS -> FAIL)"
                else:
                    status_str = "MUTATED"
                print(f"  [{idx:02d}/30] {tid}: >>> {status_str} with {len(accepted_edits_task)} edit(s) <<<")
            else:
                prog = ar_r["assembled_program"]
                is_ast_valid = ar_r["ast_valid"]
                ast_error = ar_r["ast_error"]
                outcome_name = ar_r["outcome"]
                details = ar_r["details"]
                repaired_passed = ar_passed
                status_str = "UNCHANGED_PASS" if ar_passed else "UNCHANGED_FAIL"

            coros_records.append({
                "task_id": tid,
                "prompt": prompt,
                "ar_passed": ar_passed,
                "selected_positions": selected_positions,
                "decisions": decisions,
                "accepted_edits_count": len(accepted_edits_task),
                "assembled_program": prog,
                "ast_valid": is_ast_valid,
                "ast_error": ast_error,
                "outcome": outcome.name if accepted_edits_task else ar_r["outcome"],
                "passed": repaired_passed,
                "status": status_str,
                "latency_seconds": ar_r["latency_seconds"] + repair_latency,
            })

        coros_pass_count = sum(1 for r in coros_records if r["passed"])
        coros_ast_count = sum(1 for r in coros_records if r["ast_valid"])
        print(f"\n{pol_name} Summary:")
        print(f"  Pass@1:         {coros_pass_count}/30 ({coros_pass_count/30*100:.1f}%)")
        print(f"  AST Validity:   {coros_ast_count}/30 ({coros_ast_count/30*100:.1f}%)")
        print(f"  Fixes:          {len(fixes)} {fixes}")
        print(f"  Regressions:    {len(regressions)} {regressions}")
        print(f"  Accepted Edits: {accepted_edits_total} / {evaluated_positions_total}")

        (out_dir / file_name).write_text(json.dumps(coros_records, indent=2))
        print(f"Saved {pol_name} records to {out_dir / file_name}")

        policy_summaries[pol_name] = {
            "pass_at_1_pct": round(coros_pass_count / 30 * 100, 1),
            "pass_at_1_count": coros_pass_count,
            "ast_validity_pct": round(coros_ast_count / 30 * 100, 1),
            "fixes": fixes,
            "regressions": regressions,
            "accepted_edits": accepted_edits_total,
            "evaluated_positions": evaluated_positions_total,
        }

    # Consolidated Master Summary
    master_summary = {
        "model_family": "Qwen2.5-Coder-0.5B-Instruct",
        "ar_checkpoint": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        "mdlm_checkpoint": "dllm-hub/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1",
        "mask_token_id": mask_id,
        "benchmark": "MBPP (30 Tasks)",
        "timestamp": int(time.time()),
        "ar_baseline": {
            "pass_at_1_pct": round(ar_pass_count / 30 * 100, 1),
            "pass_at_1_count": ar_pass_count,
            "ast_validity_pct": round(ar_ast_count / 30 * 100, 1),
            "mean_latency_seconds": round(sum(r["latency_seconds"] for r in ar_results) / 30, 2),
        },
        "mdlm_baseline": {
            "pass_at_1_pct": round(mdlm_pass_count / 30 * 100, 1),
            "pass_at_1_count": mdlm_pass_count,
            "ast_validity_pct": round(mdlm_ast_count / 30 * 100, 1),
            "mean_latency_seconds": round(sum(r["latency_seconds"] for r in mdlm_results) / 30, 2),
        },
        "corosred_policies": policy_summaries,
    }
    (out_dir / "summary.json").write_text(json.dumps(master_summary, indent=2))
    print(f"\nSaved consolidated summary to {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
