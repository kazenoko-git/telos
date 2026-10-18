#!/usr/bin/env python3
"""Evaluate Qwen3-0.6B AR, MDLM, and COROSred via the Télos evaluation harness.

Evaluates actual generation across identical tasks, visible prompts, and execution tests:
1. Native Qwen3-0.6B AR (greedy causal generation)
2. Native Qwen3-0.6B MDLM (block-diffusion generation)
3. COROSred two-model system (AR draft -> uncertainty routing -> MDLM bidirectional repair)

Outputs:
- Initial AR drafts and final completions for every task
- Execution results (Pass@1, AST validity, outcome categories)
- End-to-end latency breakdowns
- Routing frequency and MDLM forward-pass counts
- Drafts broken vs drafts fixed analysis
"""
from __future__ import annotations

import argparse
import copy
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

from telos.eval.adapters.base import BaseModelAdapter
from telos.eval.runner import evaluate_functional
from telos.eval.executor import ExecutionResult

from scripts.qwen3_06b_model import (
    Qwen3_06B_Config,
    Qwen3_06B_Model,
    load_qwen3_06b_model,
    AR_WEIGHTS,
    DIFF_WEIGHTS,
)

TOKENIZER_FILE = (
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/tokenizer.json"
)

EOS_TOKENS = {151645, 151643}  # <|im_end|>, <|endoftext|>
MASK_TOKEN_ID = 151669  # <|mask|>
DEFAULT_STOP_STRINGS = ["\nclass ", "\ndef ", "\nif __name__"]


class Qwen3_AR_Adapter(BaseModelAdapter):
    """Adapter for native Qwen3-0.6B autoregressive generation with KV caching."""

    def __init__(self, model: Qwen3_06B_Model, tok: Tokenizer, device: str = "mps"):
        super().__init__(model_name="Qwen3-0.6B-AR", backend_name="pytorch_ar")
        self.model = model
        self.tok = tok
        self.device = device
        self.generation_records: List[Dict[str, Any]] = []

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        t0 = time.monotonic()
        stop_strings = list(stop) if stop else list(DEFAULT_STOP_STRINGS)
        max_tokens = min(max_new_tokens, 128)

        input_ids = self.tok.encode(prompt, add_special_tokens=False).ids
        prompt_len = len(input_ids)
        generated_ids: List[int] = []

        # Initial prompt evaluation with KV cache
        t_in = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        logits, past_kvs = self.model(t_in, is_causal=True, use_cache=True)
        pred_token = int(logits[0, -1, :].argmax().item())

        for _ in range(max_tokens):
            if pred_token in EOS_TOKENS:
                break

            generated_ids.append(pred_token)

            # Check stop strings on decoded text
            if stop_strings:
                text_so_far = self.tok.decode(generated_ids)
                if any(s in text_so_far for s in stop_strings):
                    break

            # Fast step with single token and KV cache
            t_curr = torch.tensor([[pred_token]], dtype=torch.long, device=self.device)
            logits, past_kvs = self.model(
                t_curr,
                is_causal=True,
                past_key_values=past_kvs,
                start_pos=prompt_len + len(generated_ids) - 1,
                use_cache=True,
            )
            pred_token = int(logits[0, -1, :].argmax().item())

        elapsed = time.monotonic() - t0
        completion = self.tok.decode(generated_ids)
        self.generation_records.append({
            "prompt": prompt,
            "completion": completion,
            "tokens_generated": len(generated_ids),
            "latency": elapsed,
        })
        return completion


class Qwen3_MDLM_Adapter(BaseModelAdapter):
    """Adapter for native dLLM Qwen3-0.6B masked diffusion generation."""

    def __init__(
        self,
        model: Qwen3_06B_Model,
        tok: Tokenizer,
        device: str = "mps",
        block_size: int = 32,
        steps: int = 16,
    ):
        super().__init__(model_name="Qwen3-0.6B-MDLM", backend_name="pytorch_mdlm")
        self.model = model
        self.tok = tok
        self.device = device
        self.block_size = block_size
        self.steps = steps
        self.generation_records: List[Dict[str, Any]] = []

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        t0 = time.monotonic()
        stop_strings = list(stop) if stop else list(DEFAULT_STOP_STRINGS)
        max_tokens = min(max_new_tokens, 128)

        prompt_ids = self.tok.encode(prompt, add_special_tokens=False).ids
        prompt_len = len(prompt_ids)
        pad_id = 151643  # <|endoftext|>

        total_len = prompt_len + max_tokens
        x = torch.full((1, total_len), pad_id, dtype=torch.long, device=self.device)
        x[0, :prompt_len] = torch.tensor(prompt_ids, dtype=torch.long, device=self.device)
        x[0, prompt_len:total_len] = MASK_TOKEN_ID

        num_blocks = max(1, max_tokens // self.block_size)
        steps_per_block = max(1, self.steps // num_blocks)
        positions = torch.arange(total_len, device=self.device)

        # Denoise block by block with low-confidence remasking
        for b in range(num_blocks):
            b_start = prompt_len + b * self.block_size
            b_end = b_start + self.block_size

            for s in range(steps_per_block):
                block_mask = (positions >= b_start) & (positions < b_end) & (x[0] == MASK_TOKEN_ID)
                num_masked = int(block_mask.sum().item())
                if num_masked == 0:
                    break

                logits = self.model(x, is_causal=False)[0]
                probs = F.softmax(logits.float(), dim=-1)
                x0 = torch.argmax(probs, dim=-1)
                x0_p = torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)

                # Transfer top-k confident predictions
                k_transfer = max(1, num_masked // (steps_per_block - s))
                confidence = torch.where(block_mask, x0_p, torch.tensor(-float("inf"), device=self.device))
                _, topk_idx = torch.topk(confidence, k=min(k_transfer, num_masked))

                x[0, topk_idx] = x0[topk_idx]

        new_ids = x[0, prompt_len:].tolist()
        # Truncate at EOS or stop string
        for i, tid in enumerate(new_ids):
            if tid in EOS_TOKENS:
                new_ids = new_ids[:i]
                break

        elapsed = time.monotonic() - t0
        completion = self.tok.decode(new_ids)
        self.generation_records.append({
            "prompt": prompt,
            "completion": completion,
            "tokens_generated": len(new_ids),
            "latency": elapsed,
        })
        return completion


class Qwen3_COROSred_Adapter(BaseModelAdapter):
    """Adapter for two-model confidence-routed selective re-diffusion (COROSred)."""

    def __init__(
        self,
        ar_model: Qwen3_06B_Model,
        diff_model: Qwen3_06B_Model,
        tok: Tokenizer,
        margin_threshold: float = 0.5,
        entropy_threshold: float = 1.0,
        device: str = "mps",
    ):
        super().__init__(model_name="Qwen3-0.6B-COROSred", backend_name="corosred_two_model")
        self.ar_model = ar_model
        self.diff_model = diff_model
        self.tok = tok
        self.margin_threshold = margin_threshold
        self.entropy_threshold = entropy_threshold
        self.device = device
        self.generation_records: List[Dict[str, Any]] = []

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        t0 = time.monotonic()
        stop_strings = list(stop) if stop else list(DEFAULT_STOP_STRINGS)
        max_tokens = min(max_new_tokens, 128)

        input_ids = self.tok.encode(prompt, add_special_tokens=False).ids
        prompt_len = len(input_ids)
        ar_tokens: List[int] = []
        token_confidences: List[Dict[str, float]] = []

        # Step 1: Generate AR draft with KV cache and uncertainty tracking
        t_ar_start = time.monotonic()
        t_in = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        logits, past_kvs = self.ar_model(t_in, is_causal=True, use_cache=True)
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
            token_confidences.append({
                "margin": p1 - p2,
                "entropy": entropy,
            })

            if stop_strings:
                text_so_far = self.tok.decode(ar_tokens)
                if any(s in text_so_far for s in stop_strings):
                    break

            t_curr = torch.tensor([[pred_token]], dtype=torch.long, device=self.device)
            logits, past_kvs = self.ar_model(
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

        ar_latency = time.monotonic() - t_ar_start
        ar_draft_text = self.tok.decode(ar_tokens)

        # Step 2: Identify uncertain spans without reference answers
        uncertain_positions = [
            i for i, conf in enumerate(token_confidences)
            if conf["margin"] < self.margin_threshold or conf["entropy"] > self.entropy_threshold
        ]

        # Step 3: Conditional routing to MDLM bidirectional repair
        if not uncertain_positions:
            # Routing did not trigger: accept AR draft directly
            total_latency = time.monotonic() - t0
            self.generation_records.append({
                "prompt": prompt,
                "ar_draft": ar_draft_text,
                "completion": ar_draft_text,
                "routed": False,
                "uncertain_tokens": 0,
                "mdlm_forward_passes": 0,
                "ar_latency": ar_latency,
                "repair_latency": 0.0,
                "total_latency": total_latency,
            })
            return ar_draft_text

        # Routing triggered: mask uncertain tokens in generated surrounding context
        t_repair_start = time.monotonic()
        repaired_tokens = list(ar_tokens)
        for idx in uncertain_positions:
            repaired_tokens[idx] = MASK_TOKEN_ID

        full_seq = input_ids + repaired_tokens
        t_full = torch.tensor([full_seq], dtype=torch.long, device=self.device)

        # Single MDLM forward pass under bidirectional attention
        diff_logits = self.diff_model(t_full, is_causal=False)[0]

        for idx in uncertain_positions:
            pos = prompt_len + idx
            pred_token = int(diff_logits[pos].argmax().item())
            repaired_tokens[idx] = pred_token

        repair_latency = time.monotonic() - t_repair_start
        final_text = self.tok.decode(repaired_tokens)
        total_latency = time.monotonic() - t0

        self.generation_records.append({
            "prompt": prompt,
            "ar_draft": ar_draft_text,
            "completion": final_text,
            "routed": True,
            "uncertain_tokens": len(uncertain_positions),
            "mdlm_forward_passes": 1,
            "ar_latency": ar_latency,
            "repair_latency": repair_latency,
            "total_latency": total_latency,
        })
        return final_text


def run_benchmark(
    suite_name: str,
    max_tasks: Optional[int],
    output_dir: Path,
    device: str = "mps",
) -> Dict[str, Any]:
    """Execute end-to-end evaluation across AR, MDLM, and COROSred."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(str(TOKENIZER_FILE))

    print("=" * 80)
    print(f"EVALUATING SYSTEMS ON SUITE: {suite_name.upper()} (Tasks: {max_tasks or 'ALL'})")
    print(f"Device: {device} | Pinned Qwen3-0.6B Models")
    print("=" * 80)

    # 1. Evaluate Native AR Baseline
    print("\n>>> System 1/3: Native Qwen3-0.6B AR Baseline <<<")
    ar_model = load_qwen3_06b_model(AR_WEIGHTS, device=device)
    ar_adapter = Qwen3_AR_Adapter(ar_model, tok, device=device)
    t0_ar = time.monotonic()
    ar_report = evaluate_functional(
        model=ar_adapter,
        tokenizer=None,
        backend="pytorch_ar",
        suite=suite_name,
        max_tasks=max_tasks,
        timeout_seconds=3.0,
        max_new_tokens=128,
    )
    ar_total_time = time.monotonic() - t0_ar
    print(f"AR Complete in {ar_total_time:.1f}s | Pass@1: {ar_report.get('pass_at_1_pct', 0.0):.1f}% | AST: {ar_report.get('ast_validity_pct', 0.0):.1f}%")

    # 2. Evaluate Native MDLM Baseline
    print("\n>>> System 2/3: Native Qwen3-0.6B MDLM Baseline <<<")
    diff_model = load_qwen3_06b_model(DIFF_WEIGHTS, device=device)
    mdlm_adapter = Qwen3_MDLM_Adapter(diff_model, tok, device=device, block_size=32, steps=32)
    t0_mdlm = time.monotonic()
    mdlm_report = evaluate_functional(
        model=mdlm_adapter,
        tokenizer=None,
        backend="pytorch_mdlm",
        suite=suite_name,
        max_tasks=max_tasks,
        timeout_seconds=3.0,
        max_new_tokens=128,
    )
    mdlm_total_time = time.monotonic() - t0_mdlm
    print(f"MDLM Complete in {mdlm_total_time:.1f}s | Pass@1: {mdlm_report.get('pass_at_1_pct', 0.0):.1f}% | AST: {mdlm_report.get('ast_validity_pct', 0.0):.1f}%")

    # 3. Evaluate COROSred Two-Model System
    print("\n>>> System 3/3: Two-Model Confidence-Routed COROSred <<<")
    coros_adapter = Qwen3_COROSred_Adapter(
        ar_model=ar_model,
        diff_model=diff_model,
        tok=tok,
        margin_threshold=0.5,
        entropy_threshold=1.0,
        device=device,
    )
    t0_coros = time.monotonic()
    coros_report = evaluate_functional(
        model=coros_adapter,
        tokenizer=None,
        backend="corosred_two_model",
        suite=suite_name,
        max_tasks=max_tasks,
        timeout_seconds=3.0,
        max_new_tokens=128,
    )
    coros_total_time = time.monotonic() - t0_coros
    print(f"COROSred Complete in {coros_total_time:.1f}s | Pass@1: {coros_report.get('pass_at_1_pct', 0.0):.1f}% | AST: {coros_report.get('ast_validity_pct', 0.0):.1f}%")

    # Cross-analyze per-task outcomes: drafts broken vs drafts fixed
    ar_tasks = ar_report.get("tasks", [])
    coros_tasks = coros_report.get("tasks", [])
    total_tasks = len(ar_tasks)

    drafts_fixed = []
    drafts_broken = []
    routing_triggered_count = sum(1 for r in coros_adapter.generation_records if r["routed"])
    total_mdlm_passes = sum(r["mdlm_forward_passes"] for r in coros_adapter.generation_records)

    per_task_comparisons = []
    for idx, (ar_t, coros_t) in enumerate(zip(ar_tasks, coros_tasks)):
        tid = ar_t.get("id", idx)
        ar_passed = (ar_t["outcome"] == ExecutionResult.PASSED.value)
        coros_passed = (coros_t["outcome"] == ExecutionResult.PASSED.value)
        gen_meta = coros_adapter.generation_records[idx] if idx < len(coros_adapter.generation_records) else {}

        if not ar_passed and coros_passed:
            drafts_fixed.append(tid)
        elif ar_passed and not coros_passed:
            drafts_broken.append(tid)

        per_task_comparisons.append({
            "task_id": tid,
            "category": ar_t.get("category"),
            "ar_outcome": ar_t["outcome"],
            "ar_ast_valid": ar_t["ast_valid"],
            "coros_outcome": coros_t["outcome"],
            "coros_ast_valid": coros_t["ast_valid"],
            "routed": gen_meta.get("routed", False),
            "uncertain_tokens": gen_meta.get("uncertain_tokens", 0),
            "ar_draft": gen_meta.get("ar_draft", ""),
            "final_completion": gen_meta.get("completion", ""),
            "ar_latency": gen_meta.get("ar_latency", 0.0),
            "repair_latency": gen_meta.get("repair_latency", 0.0),
            "total_latency": gen_meta.get("total_latency", 0.0),
        })

    # Compile master comparative summary
    comparative_summary = {
        "suite": suite_name,
        "total_tasks": total_tasks,
        "timestamp": int(time.time()),
        "pinned_revisions": {
            "ar_model": "Qwen/Qwen3-0.6B @ c1899de289a04d12100db370d81485cdf75e47ca",
            "diff_model": "dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1 @ c8d24a3f4adaeef46881b450e1bf7d1005203bd7",
        },
        "decoding_parameters": {
            "temperature": 0.0,
            "max_new_tokens": 128,
            "mdlm_block_size": mdlm_adapter.block_size,
            "mdlm_steps": mdlm_adapter.steps,
            "corosred_margin_threshold": coros_adapter.margin_threshold,
            "corosred_entropy_threshold": coros_adapter.entropy_threshold,
        },
        "systems": {
            "native_ar": {
                "pass_at_1_pct": ar_report.get("pass_at_1_pct", 0.0),
                "ast_validity_pct": ar_report.get("ast_validity_pct", 0.0),
                "total_time_seconds": round(ar_total_time, 2),
                "mean_latency_per_task": round(ar_total_time / max(1, total_tasks), 3),
                "outcomes": ar_report.get("execution_outcomes", {}),
            },
            "native_mdlm": {
                "pass_at_1_pct": mdlm_report.get("pass_at_1_pct", 0.0),
                "ast_validity_pct": mdlm_report.get("ast_validity_pct", 0.0),
                "total_time_seconds": round(mdlm_total_time, 2),
                "mean_latency_per_task": round(mdlm_total_time / max(1, total_tasks), 3),
                "outcomes": mdlm_report.get("execution_outcomes", {}),
            },
            "corosred": {
                "pass_at_1_pct": coros_report.get("pass_at_1_pct", 0.0),
                "ast_validity_pct": coros_report.get("ast_validity_pct", 0.0),
                "total_time_seconds": round(coros_total_time, 2),
                "mean_latency_per_task": round(coros_total_time / max(1, total_tasks), 3),
                "outcomes": coros_report.get("execution_outcomes", {}),
                "routing_frequency": f"{routing_triggered_count}/{total_tasks} ({routing_triggered_count/max(1, total_tasks)*100:.1f}%)",
                "total_mdlm_forward_passes": total_mdlm_passes,
                "drafts_fixed_count": len(drafts_fixed),
                "drafts_fixed_task_ids": drafts_fixed,
                "drafts_broken_count": len(drafts_broken),
                "drafts_broken_task_ids": drafts_broken,
            },
        },
    }

    # Save artifact files
    (output_dir / "comparative_summary.json").write_text(json.dumps(comparative_summary, indent=2))
    (output_dir / "ar_eval_report.json").write_text(json.dumps(ar_report, indent=2))
    (output_dir / "mdlm_eval_report.json").write_text(json.dumps(mdlm_report, indent=2))
    (output_dir / "corosred_eval_report.json").write_text(json.dumps(coros_report, indent=2))
    (output_dir / "per_task_comparisons.json").write_text(json.dumps(per_task_comparisons, indent=2))

    print(f"\nSaved all evaluation artifacts to {output_dir}")
    print("\n=== COMPARATIVE SUMMARY ===")
    print(json.dumps(comparative_summary["systems"], indent=2))
    return comparative_summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen3-0.6B AR, MDLM, and COROSred.")
    parser.add_argument("--suite", type=str, default="humaneval", help="Benchmark suite name")
    parser.add_argument("--smoke-test", action="store_true", help="Run 5 tasks smoke test")
    parser.add_argument("--max-tasks", type=int, default=None, help="Maximum tasks to evaluate")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory path")
    args = parser.parse_args()

    max_tasks = 5 if args.smoke_test else args.max_tasks
    out_dir_name = f"corosred_{args.suite}_{'smoke5' if args.smoke_test else (args.max_tasks or 'full')}_2026-10-18"
    out_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "research" / out_dir_name

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    run_benchmark(suite_name=args.suite, max_tasks=max_tasks, output_dir=out_dir, device=device)


if __name__ == "__main__":
    main()
