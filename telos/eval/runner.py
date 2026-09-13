"""
High-End Evaluation Engine for Télos Models.

Supports:
1. Contextual Probes Suite (1,000 deterministic probes across 8 categories, infill/causal).
2. Functional Execution Benchmark (Pass@1 with sandboxed subprocess isolation and hard timeouts).
3. Syntactic AST Analysis (inline ast.parse validation and error categorization).
4. Anti-Cheat & Suffix-Copy Detection (multi-token span chunk masking K in {1, 2, 4, 8, 16}).
5. Dual-Track Benchmark Suites: Primary Private Unseen vs Public Standard.
6. Statistical Rigor: 95% Bootstrap Confidence Intervals and paired significance testing.

Compatible with both MLX (.safetensors) and PyTorch (.pt) checkpoints.
"""

import os
import sys
import json
import time
import math
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from .probes import load_contextual_probes, PROBE_SUITE_100
from .syntax import check_ast_validity, categorize_syntax_error, analyze_syntax_batch
from .executor import execute_code_sandboxed, ExecutionResult
from .anticheat import evaluate_span_infill_probe, summarize_anticheat_suite
from telos.data.tokenizer import load_tokenizer
from telos.models import TelosConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def bootstrap_confidence_interval(
    binary_outcomes: List[float],
    n_resamples: int = 1000,
    confidence_level: float = 0.95
) -> Tuple[float, float]:
    """
    Computes empirical bootstrap confidence interval for binary outcomes.
    Returns (ci_lower_pct, ci_upper_pct).
    """
    if not binary_outcomes:
        return 0.0, 0.0
    arr = np.array(binary_outcomes, dtype=np.float32)
    n = len(arr)
    if n <= 1:
        val = float(arr[0] * 100.0) if n == 1 else 0.0
        return val, val

    np.random.seed(42)
    resample_means = [
        float(np.mean(np.random.choice(arr, size=n, replace=True)))
        for _ in range(n_resamples)
    ]
    alpha = (1.0 - confidence_level) / 2.0
    lower = float(np.percentile(resample_means, alpha * 100.0)) * 100.0
    upper = float(np.percentile(resample_means, (1.0 - alpha) * 100.0)) * 100.0
    return round(lower, 2), round(upper, 2)


def load_model_from_checkpoint(checkpoint_path: str | Path, config: dict | None = None):
    """Loads an MLX or PyTorch model along with architecture config from a checkpoint path."""
    cp = Path(checkpoint_path)
    if cp.is_dir():
        mlx_file = cp / "model.safetensors"
        pt_file = cp / "checkpoint_final.pt"
        if not pt_file.exists():
            pt_files = sorted(cp.glob("checkpoint_step_*.pt"))
            pt_file = pt_files[-1] if pt_files else pt_file

        if mlx_file.exists():
            cp = mlx_file
        elif pt_file.exists():
            cp = pt_file
        else:
            raise FileNotFoundError(f"No checkpoint file found in directory {checkpoint_path}")

    cfg_file = cp.parent / "config.json"
    if not cfg_file.exists():
        cfg_file = cp.parent / "config.yaml"
    full_cfg = {}
    m_cfg = {}
    if cfg_file.exists():
        try:
            with open(cfg_file, "r") as f:
                full_cfg = json.load(f)
        except Exception:
            try:
                import yaml
                with open(cfg_file, "r") as f:
                    full_cfg = yaml.safe_load(f) or {}
            except Exception:
                full_cfg = {}
        m_cfg = full_cfg.get("model", full_cfg)
    elif config:
        full_cfg = config
        m_cfg = config.get("model", config)
    elif cp.suffix in [".pt", ".bin"]:
        try:
            import torch
            ckpt_data = torch.load(str(cp), map_location="cpu")
            if isinstance(ckpt_data, dict) and "config" in ckpt_data:
                full_cfg = ckpt_data.get("config", {})
                m_cfg = full_cfg.get("model", full_cfg)
        except Exception:
            pass

    vocab_size = m_cfg.get("vocab_size", 8192)
    d_model = m_cfg.get("d_model", 512)
    n_layers = m_cfg.get("n_layers", 12)
    n_heads = m_cfg.get("n_heads", 16)
    n_kv_heads = m_cfg.get("n_kv_heads", n_heads)

    paradigm = str(full_cfg.get("paradigm", "") or m_cfg.get("paradigm", "")).lower()
    if not paradigm:
        for p in ["corosred", "mdlm", "undlm", "ar"]:
            if p in str(cp).lower():
                paradigm = p
                break

    is_causal = bool(m_cfg.get("is_causal", paradigm == "ar"))
    use_reliability_head = bool(m_cfg.get("use_reliability_head", paradigm == "corosred"))

    if cp.suffix == ".safetensors":
        from safetensors import safe_open
        with safe_open(str(cp), framework="numpy") as f:
            keys = list(f.keys())
        is_pytorch_keys = any("tok_embeddings" in k or ".attn." in k or "final_norm" in k for k in keys)

        if is_pytorch_keys:
            import torch
            import torch.nn as nn
            from safetensors.torch import load_file
            sd = load_file(str(cp))
            sd = {k.removeprefix("module.").removeprefix("_orig_mod."): v for k, v in sd.items()}

            if not use_reliability_head and any("reliability_head" in k for k in sd.keys()):
                use_reliability_head = True

            tc = TelosConfig(
                vocab_size=vocab_size,
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                tied_embeddings=True,
                is_causal=is_causal,
                use_reliability_head=use_reliability_head,
            )
            from telos.models import TelosTransformer
            model = TelosTransformer(tc)
            if "reliability_head.weight" in sd and hasattr(model, "reliability_head") and isinstance(model.reliability_head, nn.Sequential):
                model.reliability_head = nn.Linear(tc.d_model, 1, bias=False)

            model.load_state_dict(sd, strict=True)
            model.eval()
            return model, "pytorch", vocab_size
        else:
            import mlx.core as mx
            from telos.models import MLXTelosTransformer
            model = MLXTelosTransformer(
                vocab_size=vocab_size,
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                tied_embeddings=True,
                is_causal=is_causal,
                use_reliability_head=use_reliability_head,
            )
            model.load_weights(str(cp))
            model.paradigm = paradigm
            return model, "mlx", vocab_size

    elif cp.suffix in [".pt", ".bin"]:
        import torch
        import torch.nn as nn
        state = torch.load(str(cp), map_location="cpu")
        sd = state.get("model_state_dict", state)
        sd = {k.removeprefix("module.").removeprefix("_orig_mod."): v for k, v in sd.items()}

        if not use_reliability_head and any("reliability_head" in k for k in sd.keys()):
            use_reliability_head = True

        tc = TelosConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            tied_embeddings=True,
            is_causal=is_causal,
            use_reliability_head=use_reliability_head,
        )
        from telos.models import TelosTransformer
        model = TelosTransformer(tc)
        if "reliability_head.weight" in sd and hasattr(model, "reliability_head") and isinstance(model.reliability_head, nn.Sequential):
            model.reliability_head = nn.Linear(tc.d_model, 1, bias=False)

        model.load_state_dict(sd, strict=True)
        model.eval()
        model.paradigm = paradigm
        return model, "pytorch", vocab_size

    else:
        raise ValueError(f"Unrecognized checkpoint format: {cp.name}")


def _evaluate_single_probe_type(
    model,
    tokenizer,
    backend: str,
    probes_list: List[Dict[str, Any]],
    probe_type: str = "infill",
    mask_token_id: int = 1
) -> Dict[str, Any]:
    """Executes a single pass (infill or causal) of the contextual probes."""
    results = []
    category_stats: Dict[str, Dict[str, Any]] = {}

    for probe in probes_list:
        cat = probe.get("category", "General")
        prompt = probe["prompt"]
        target_str = probe["target"]

        if cat not in category_stats:
            category_stats[cat] = {"count": 0, "top1": 0, "top5": 0, "ce": [], "rank": []}

        # Context-aware tokenization to handle ByteLevel BPE leading-space byte ('Ġ')
        p_ids = tokenizer.encode(prompt).ids
        target_bpe = probe.get("target_bpe", "")
        target_tok = None
        if target_bpe:
            target_tok = tokenizer.token_to_id(target_bpe)

        if target_tok is None:
            sep = " " if ("Ġ" in target_bpe and not prompt.endswith(" ")) else ""
            full_text = prompt + sep + target_str
            full_ids = tokenizer.encode(full_text).ids
            if len(full_ids) > len(p_ids):
                target_tok = full_ids[len(p_ids)]
            else:
                target_ids = tokenizer.encode(target_str).ids
                target_tok = target_ids[0] if target_ids else 0

        if probe_type == "infill":
            suffix = probe.get("suffix", "")
            s_ids = tokenizer.encode(suffix).ids if suffix else []
            input_ids = list(p_ids) + [mask_token_id] + s_ids
            eval_idx = len(p_ids)
            mask_override = False
        else:
            input_ids = list(p_ids)
            eval_idx = len(p_ids) - 1
            mask_override = True

        if backend == "mlx":
            import mlx.core as mx
            x = mx.array([input_ids], dtype=mx.int32)
            logits = model(x, mask_override=mask_override)
            logits_pos = np.array(logits[0, eval_idx].astype(mx.float32))
        else:
            import torch
            x = torch.tensor([input_ids], dtype=torch.long)
            with torch.no_grad():
                logits = model(x, mask_override=mask_override)
            logits_pos = logits[0, eval_idx].detach().cpu().numpy()

        if probe_type == "infill":
            # Prevent predicting the [MASK] token itself
            logits_pos[mask_token_id] = -1e9

        # Numerically stable softmax probabilities
        shifted = logits_pos - np.max(logits_pos)
        probs = np.exp(shifted) / np.sum(np.exp(shifted))

        target_prob = max(float(probs[target_tok]), 1e-12)
        target_ce = -math.log(target_prob)

        sorted_indices = np.argsort(logits_pos)[::-1]
        rank = int(np.where(sorted_indices == target_tok)[0][0]) + 1
        is_top1 = (rank == 1)
        is_top5 = (rank <= 5)

        category_stats[cat]["count"] += 1
        if is_top1:
            category_stats[cat]["top1"] += 1
        if is_top5:
            category_stats[cat]["top5"] += 1
        category_stats[cat]["ce"].append(target_ce)
        category_stats[cat]["rank"].append(rank)

        results.append({
            "category": cat,
            "prompt": prompt,
            "target": target_str,
            "rank": rank,
            "target_ce": target_ce,
            "top1": is_top1,
            "top5": is_top5,
        })

    total_count = len(results)
    overall_top1 = sum(r["top1"] for r in results) / max(total_count, 1) * 100.0
    overall_top5 = sum(r["top5"] for r in results) / max(total_count, 1) * 100.0
    overall_ce = float(np.mean([r["target_ce"] for r in results])) if results else 0.0
    overall_rank = float(np.mean([r["rank"] for r in results])) if results else 0.0

    ci_top1 = bootstrap_confidence_interval([1.0 if r["top1"] else 0.0 for r in results])

    cat_breakdown = {}
    for cat, s in category_stats.items():
        cnt = s["count"]
        top1_pct = (s["top1"] / cnt) * 100.0 if cnt else 0.0
        top5_pct = (s["top5"] / cnt) * 100.0 if cnt else 0.0
        avg_r = float(np.mean(s["rank"])) if cnt else 0.0
        avg_ce = float(np.mean(s["ce"])) if cnt else 0.0
        cat_breakdown[cat] = {
            "count": cnt,
            "top1_pct": round(top1_pct, 1),
            "top5_pct": round(top5_pct, 1),
            "avg_rank": round(avg_r, 1),
            "avg_ce": round(avg_ce, 2)
        }

    title = "BIDIRECTIONAL INFILLING PROBES" if probe_type == "infill" else "CAUSAL CONTINUATION PROBES"
    print("\n" + "=" * 80)
    print(f"  TÉLOS CONTEXTUAL PROBES REPORT: {title} ({total_count} PROBES)")
    print("=" * 80)
    print(f"  {'Category':<26} | {'Count':<5} | {'Top-1 (%)':<9} | {'Top-5 (%)':<9} | {'Avg Rank':<8} | {'Avg CE':<6}")
    print("-" * 80)

    for cat, s in cat_breakdown.items():
        print(f"  {cat:<26} | {s['count']:<5d} | {s['top1_pct']:>8.1f}% | {s['top5_pct']:>8.1f}% | {s['avg_rank']:>8.1f} | {s['avg_ce']:>6.2f}")

    print("-" * 80)
    print(f"  {'OVERALL SUMMARY':<26} | {total_count:<5d} | {overall_top1:>8.1f}% | {overall_top5:>8.1f}% | {overall_rank:>8.1f} | {overall_ce:>6.2f}")
    print(f"  95% Bootstrap CI for Top-1: [{ci_top1[0]}%, {ci_top1[1]}%]")
    print("=" * 80 + "\n")

    return {
        "probe_type": probe_type,
        "overall": {
            "top1_acc_pct": round(overall_top1, 2),
            "top1_95ci": ci_top1,
            "top5_acc_pct": round(overall_top5, 2),
            "mean_rank": round(overall_rank, 1),
            "mean_ce": round(overall_ce, 2),
            "total_probes": total_count,
        },
        "categories": cat_breakdown,
        "probes": results,
    }


def evaluate_probes(
    model,
    tokenizer,
    backend: str,
    paradigm: str = "corosred",
    num_probes: int = 100,
    mask_token_id: int = 1,
    probe_type: str = "both"
) -> Dict[str, Any]:
    """
    Runs contextual probes with paradigm differentiation:
    - Pure causal AR models are evaluated ONLY on causal next-token completion.
    - Bidirectional models (COROSred) are evaluated on both causal completion and masked infilling.
    """
    probes_list = load_contextual_probes(num_probes)
    report_payload = {}

    # 1. Causal evaluation: Evaluates next-token prediction at prefix boundary across all models
    if probe_type in ["causal", "both"]:
        causal_probes = [p for p in probes_list if p.get("mode", "both") in ["both", "causal"]]
        res_causal = _evaluate_single_probe_type(
            model=model,
            tokenizer=tokenizer,
            backend=backend,
            probes_list=causal_probes,
            probe_type="causal",
            mask_token_id=mask_token_id
        )
        report_payload["causal"] = res_causal

    # 2. Infill evaluation: Evaluates bidirectional [MASK] infilling
    # AR models cannot condition bidirectionally without infill conditioning; marked N/A.
    if probe_type in ["infill", "both"]:
        if str(paradigm).lower() == "ar":
            report_payload["infill"] = {
                "status": "not_applicable",
                "reason": "AR models are strictly causal-only and cannot condition on bidirectional suffixes",
                "overall": {
                    "top1_acc_pct": None,
                    "top5_acc_pct": None,
                    "mean_ce": None,
                    "mean_rank": None,
                    "total_probes": 0,
                },
                "categories": {}
            }
        else:
            res_infill = _evaluate_single_probe_type(
                model=model,
                tokenizer=tokenizer,
                backend=backend,
                probes_list=probes_list,
                probe_type="infill",
                mask_token_id=mask_token_id
            )
            report_payload["infill"] = res_infill

    return report_payload


def _generate_greedy_completion(
    model,
    tokenizer,
    backend: str,
    prompt: str,
    max_new_tokens: int = 128,
    stop_tokens: Optional[List[int]] = None
) -> str:
    """Generates code completion using deterministic greedy decoding (Temperature = 0.0)."""
    p_ids = tokenizer.encode(prompt).ids
    curr_ids = list(p_ids)
    stop_set = set(stop_tokens or [0, 3])  # EOS / PAD

    for _ in range(max_new_tokens):
        if backend == "mlx":
            import mlx.core as mx
            x = mx.array([curr_ids], dtype=mx.int32)
            logits = model(x)
            next_tok = int(np.argmax(np.array(logits[0, -1].astype(mx.float32))))
        else:
            import torch
            x = torch.tensor([curr_ids], dtype=torch.long)
            with torch.no_grad():
                logits = model(x)
            next_tok = int(torch.argmax(logits[0, -1]).item())

        if next_tok in stop_set:
            break
        curr_ids.append(next_tok)

    # Decode only the newly generated continuation tokens
    continuation_ids = curr_ids[len(p_ids):]
    return tokenizer.decode(continuation_ids)


def evaluate_functional(
    model,
    tokenizer,
    backend: str,
    suite: str = "private_unseen",
    max_tasks: Optional[int] = None,
    timeout_seconds: float = 3.0,
    max_new_tokens: int = 128
) -> Dict[str, Any]:
    """
    Executes functional unit testing benchmark (Pass@1 with sandboxed subprocesses).
    """
    bench_dir = PROJECT_ROOT / "evals" / "benchmarks"
    if suite == "public_standard":
        data_file = bench_dir / "public_standard_suite.json"
    else:
        data_file = bench_dir / "private_unseen_suite.json"

    if not data_file.exists():
        raise FileNotFoundError(f"Benchmark dataset not found at {data_file}. Run scripts/build_evaluation_benchmarks.py first.")

    with open(data_file, "r") as f:
        tasks = json.load(f)

    if max_tasks and len(tasks) > max_tasks:
        # Balanced sampling across categories
        cats = {}
        for t in tasks:
            cats.setdefault(t.get("category", "General"), []).append(t)
        sampled = []
        per_cat = max(1, max_tasks // len(cats))
        for cat, cat_tasks in cats.items():
            sampled.extend(cat_tasks[:per_cat])
        tasks = sampled[:max_tasks]

    print("\n" + "=" * 80)
    print(f"  TÉLOS FUNCTIONAL EXECUTION BENCHMARK: {suite.upper()} ({len(tasks)} TASKS)")
    print(f"  Subprocess Sandbox: spawn | Timeout: {timeout_seconds}s | Decoding: Greedy (T=0.0)")
    print("=" * 80)

    category_stats: Dict[str, Dict[str, Any]] = {}
    task_results = []
    outcome_counts = {k.value: 0 for k in ExecutionResult}
    ast_valid_count = 0

    for idx, task in enumerate(tasks, 1):
        cat = task.get("category", "General")
        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "passed": 0, "ast_valid": 0, "outcomes": {}}

        prompt = task["prompt"]
        test_harness = task.get("test_harness", "")

        # 1. Generate code completion with greedy decoding
        completion = _generate_greedy_completion(
            model=model,
            tokenizer=tokenizer,
            backend=backend,
            prompt=prompt,
            max_new_tokens=max_new_tokens
        )
        full_candidate_code = prompt + completion

        # 2. Check inline AST syntax validity
        is_ast_valid, ast_err = check_ast_validity(full_candidate_code)
        if is_ast_valid:
            ast_valid_count += 1
            category_stats[cat]["ast_valid"] += 1

        # 3. Execute in sandboxed child process
        outcome, details = execute_code_sandboxed(
            code=full_candidate_code,
            test_harness=test_harness,
            timeout_seconds=timeout_seconds
        )

        outcome_counts[outcome.value] += 1
        is_passed = (outcome == ExecutionResult.PASSED)
        category_stats[cat]["total"] += 1
        if is_passed:
            category_stats[cat]["passed"] += 1

        cat_outcomes = category_stats[cat]["outcomes"]
        cat_outcomes[outcome.value] = cat_outcomes.get(outcome.value, 0) + 1

        task_results.append({
            "id": task.get("id", idx),
            "category": cat,
            "outcome": outcome.value,
            "ast_valid": is_ast_valid,
            "ast_error": ast_err,
            "details": details,
        })

        if idx % 10 == 0 or idx == len(tasks):
            print(f"  [{idx}/{len(tasks)}] Completed... Current Pass@1: {round(outcome_counts[ExecutionResult.PASSED.value]/idx*100, 1)}%")

    total_tasks = len(tasks)
    passed_total = outcome_counts[ExecutionResult.PASSED.value]
    overall_pass1 = (passed_total / max(total_tasks, 1)) * 100.0
    overall_ast_valid = (ast_valid_count / max(total_tasks, 1)) * 100.0

    ci_pass1 = bootstrap_confidence_interval([1.0 if r["outcome"] == ExecutionResult.PASSED.value else 0.0 for r in task_results])

    print("\n" + "-" * 80)
    print(f"  {'Category':<28} | {'Total':<5} | {'Pass@1 (%)':<10} | {'AST Valid (%)':<13}")
    print("-" * 80)
    cat_summary = {}
    for cat, s in category_stats.items():
        cnt = s["total"]
        pass_pct = round((s["passed"] / cnt) * 100.0, 1) if cnt else 0.0
        ast_pct = round((s["ast_valid"] / cnt) * 100.0, 1) if cnt else 0.0
        cat_summary[cat] = {
            "count": cnt,
            "pass_at_1_pct": pass_pct,
            "ast_valid_pct": ast_pct,
            "outcomes": s["outcomes"],
        }
        print(f"  {cat:<28} | {cnt:<5d} | {pass_pct:>9.1f}% | {ast_pct:>12.1f}%")

    print("-" * 80)
    print(f"  OVERALL PASS@1:       {overall_pass1:.2f}% (95% CI: [{ci_pass1[0]}%, {ci_pass1[1]}%])")
    print(f"  OVERALL AST VALIDITY: {overall_ast_valid:.2f}%")
    print(f"  Execution Breakdown:  {outcome_counts}")
    print("=" * 80 + "\n")

    return {
        "suite": suite,
        "total_tasks": total_tasks,
        "pass_at_1_pct": round(overall_pass1, 2),
        "pass_at_1_95ci": ci_pass1,
        "ast_validity_pct": round(overall_ast_valid, 2),
        "execution_outcomes": outcome_counts,
        "category_breakdown": cat_summary,
        "tasks": task_results,
    }


def evaluate_anticheat(
    model,
    tokenizer,
    backend: str,
    num_probes: int = 100,
    span_lengths: Optional[List[int]] = None
) -> Dict[str, Any]:
    """Evaluates multi-token chunk masking and suffix-copying cheat rates."""
    probes = load_contextual_probes(num_probes)
    span_lengths = span_lengths or [1, 2, 4, 8]

    print("\n" + "=" * 80)
    print(f"  TÉLOS ANTI-CHEAT & SUFFIX-COPY BENCHMARK ({len(probes)} PROBES)")
    print(f"  Span Lengths Tested: K in {span_lengths}")
    print("=" * 80)

    all_probe_results = []
    for p in probes:
        prefix = p["prompt"]
        target = p.get("multi_token_target") or p["target"]
        suffix = p.get("suffix", "")
        res = evaluate_span_infill_probe(
            model=model,
            tokenizer=tokenizer,
            backend=backend,
            prefix_text=prefix,
            target_text=target,
            suffix_text=suffix,
            span_lengths=span_lengths
        )
        all_probe_results.append(res)

    summary = summarize_anticheat_suite(all_probe_results)

    print("\n  SPAN-CHUNK INFILLING DEGRADATION & COPY PROFILE:")
    print(f"  {'Span (K)':<10} | {'Exact Match (%)':<15} | {'Token Acc (%)':<15} | {'Suffix Copy (%)':<15} | {'Prefix Copy (%)':<15}")
    print("-" * 80)
    for span_k, data in summary.get("span_breakdown", {}).items():
        print(f"  {span_k:<10} | {data['exact_match_pct']:>14.1f}% | {data['token_accuracy_pct']:>14.1f}% | {data['suffix_copy_rate_pct']:>14.1f}% | {data['prefix_copy_rate_pct']:>14.1f}%")

    print("-" * 80)
    if summary.get("is_suspect_cheater"):
        print("  [!] WARNING: Model exhibits high boundary suffix-copying cheat characteristics!")
    else:
        print("  [✓] PASSED: Model infilling performance shows robust multi-token semantic reasoning.")
    print("=" * 80 + "\n")

    return summary


def evaluate_sample(model, tokenizer, backend: str, prompts: list[str] | None = None, steps: int = 32):
    """Generates qualitative code completions for sample evaluation prompts."""
    default_prompts = [
        "def fibonacci(n: int) -> int:\n    if n <= 1:\n        return",
        "def quicksort(arr: list[int]) -> list[int]:\n    if len(arr) <= 1:\n",
        "class Node:\n    def __init__(self, value):\n        self.value = value\n        self.",
    ]
    prompts = prompts or default_prompts

    print("\n" + "=" * 76)
    print("  TÉLOS QUALITATIVE CODE COMPLETION EVALUATION")
    print("=" * 76)

    for i, p in enumerate(prompts, 1):
        print(f"\n--- [Prompt {i}] ---\n{p}")
        completion = _generate_greedy_completion(model, tokenizer, backend, p, max_new_tokens=steps)
        print(f"--- [Completion] ---\n{p + completion}\n")
    print("=" * 76 + "\n")


def _evaluate_single(
    checkpoint: str | Path,
    mode: str = "probes",
    suite: str = "private_unseen",
    probe_type: str = "both",
    num_probes: int = 100,
    max_tasks: Optional[int] = None,
    timeout: float = 3.0,
    tokenizer_path: str | Path | None = None,
    output_path: Optional[str | Path] = None,
    **kwargs
) -> Dict[str, Any]:
    """Evaluates a single model checkpoint."""
    model, backend, vocab_size = load_model_from_checkpoint(checkpoint)
    if tokenizer_path is None:
        if vocab_size == 8192 and (PROJECT_ROOT / "configs" / "tokenizer_mac.json").exists():
            tokenizer_path = str(PROJECT_ROOT / "configs" / "tokenizer_mac.json")
        elif (PROJECT_ROOT / "configs" / "shared" / "tokenizer_0.json").exists():
            tokenizer_path = str(PROJECT_ROOT / "configs" / "shared" / "tokenizer_0.json")
    tok = load_tokenizer(str(tokenizer_path) if tokenizer_path else None)

    paradigm = getattr(model, "paradigm", "")
    if not paradigm:
        for p in ["corosred", "ar", "mdlm", "undlm"]:
            if p in str(checkpoint).lower():
                paradigm = p
                break
    paradigm = paradigm or "corosred"

    report: Dict[str, Any] = {
        "model_checkpoint": str(checkpoint),
        "backend": backend,
        "paradigm": paradigm,
        "vocab_size": vocab_size,
        "timestamp": int(time.time()),
        "mode": mode,
    }

    if mode == "probes":
        report["probes"] = evaluate_probes(
            model, tok, backend, paradigm=paradigm, num_probes=num_probes, probe_type=probe_type
        )
    elif mode == "functional":
        report["functional"] = evaluate_functional(
            model, tok, backend, suite=suite, max_tasks=max_tasks, timeout_seconds=timeout
        )
    elif mode == "anticheat":
        report["anticheat"] = evaluate_anticheat(model, tok, backend, num_probes=min(num_probes, 200))
    elif mode == "sample":
        evaluate_sample(model, tok, backend)
        report["sample"] = {"status": "completed"}
    elif mode == "full":
        report["probes"] = evaluate_probes(
            model, tok, backend, paradigm=paradigm, num_probes=num_probes, probe_type=probe_type
        )
        report["functional"] = evaluate_functional(
            model, tok, backend, suite=suite, max_tasks=max_tasks, timeout_seconds=timeout
        )
        report["anticheat"] = evaluate_anticheat(model, tok, backend, num_probes=min(num_probes, 200))
    else:
        raise ValueError(f"Unknown evaluation mode: {mode}. Choose 'probes', 'functional', 'anticheat', 'full', or 'sample'.")

    # Save single report if output_path is provided
    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w") as f:
            json.dump(report, f, indent=2)
        print(f"✓ Saved evaluation report to {out_file}\n")

    return report


def print_multimodel_scorecard(multi_reports: Dict[str, Any], mode: str):
    """Renders a comparative scorecard across multiple evaluated models."""
    print("\n" + "=" * 115)
    print(f"                       TÉLOS MULTI-MODEL EVALUATION SCORECARD ({mode.upper()})")
    print("=" * 115)

    if mode in ["probes", "full"]:
        header = (
            f"{'Model / Checkpoint':<38} | {'Backend':<7} | {'Infill Top1':<11} | {'Infill CE':<9} | "
            f"{'Causal Top1':<11} | {'Causal CE':<9} | {'Infill Rank':<11}"
        )
        print(header)
        print("-" * 115)
        for name, rep in multi_reports.items():
            b = rep.get("backend", "unknown")
            pr = rep.get("probes", {})
            inf = pr.get("infill", {}).get("overall", {})
            cau = pr.get("causal", {}).get("overall", {})
            
            if pr.get("infill", {}).get("status") == "not_applicable":
                inf_t1 = "N/A (AR)"
                inf_ce = "N/A"
                inf_rnk = "N/A"
            else:
                inf_t1 = f"{inf.get('top1_acc_pct', 0.0):.1f}%" if inf.get('top1_acc_pct') is not None else "N/A"
                inf_ce = f"{inf.get('mean_ce', 0.0):.2f}" if inf.get('mean_ce') is not None else "N/A"
                inf_rnk = f"{inf.get('mean_rank', 0.0):.1f}" if inf.get('mean_rank') is not None else "N/A"

            cau_t1 = f"{cau.get('top1_acc_pct', 0.0):.1f}%" if cau.get('top1_acc_pct') is not None else "N/A"
            cau_ce = f"{cau.get('mean_ce', 0.0):.2f}" if cau.get('mean_ce') is not None else "N/A"
            disp_name = name if len(name) <= 38 else "..." + name[-35:]
            print(f"{disp_name:<38} | {b:<7} | {inf_t1:<11} | {inf_ce:<9} | {cau_t1:<11} | {cau_ce:<9} | {inf_rnk:<11}")

        # Category breakdown for deterministic probes
        print("\n  CATEGORY BREAKDOWN (TOP-1 ACCURACY):")
        cat_header = f"{'Model / Checkpoint':<38} | {'Identifiers':<14} | {'Keywords':<14} | {'Imports & Calls':<17} | {'Suffix-Clued Infill'}"
        print(cat_header)
        print("-" * 115)
        for name, rep in multi_reports.items():
            pr = rep.get("probes", {})
            cau_cats = pr.get("causal", {}).get("categories", {})
            inf_cats = pr.get("infill", {}).get("categories", {})
            id_t1 = f"{cau_cats.get('Contextually Deterministic Identifiers', {}).get('top1_pct', 0.0):.1f}%"
            kw_t1 = f"{cau_cats.get('Syntactic Keywords', {}).get('top1_pct', 0.0):.1f}%"
            imp_t1 = f"{cau_cats.get('Idiomatic Imports & Calls', {}).get('top1_pct', 0.0):.1f}%"
            
            if pr.get("infill", {}).get("status") == "not_applicable":
                sc_t1 = "N/A (AR)"
            else:
                sc_t1 = f"{inf_cats.get('Suffix-Clued Bidirectional Infill', {}).get('top1_pct', 0.0):.1f}%"
                
            disp_name = name if len(name) <= 38 else "..." + name[-35:]
            print(f"{disp_name:<38} | {id_t1:<14} | {kw_t1:<14} | {imp_t1:<17} | {sc_t1}")

    if mode in ["anticheat", "full"]:
        print("\n" + "-" * 115)
        print("  ANTI-CHEAT SUFFIX-COPY RATE & SPAN DEGRADATION:")
        print(f"{'Model / Checkpoint':<38} | {'K=1 Suffix Copy':<16} | {'K=2 Suffix Copy':<16} | {'K=4 Suffix Copy':<16} | {'Cheat Detected'}")
        print("-" * 115)
        for name, rep in multi_reports.items():
            ac = rep.get("anticheat", {})
            spans = ac.get("span_breakdown", ac.get("span_results", {}))
            k1 = f"{spans.get('span_1', {}).get('suffix_copy_rate_pct', spans.get('1', {}).get('suffix_copy_rate', 0.0)):.1f}%"
            k2 = f"{spans.get('span_2', {}).get('suffix_copy_rate_pct', spans.get('2', {}).get('suffix_copy_rate', 0.0)):.1f}%"
            k4 = f"{spans.get('span_4', {}).get('suffix_copy_rate_pct', spans.get('4', {}).get('suffix_copy_rate', 0.0)):.1f}%"
            cheat = "YES (CHEAT)" if ac.get("is_suspect_cheater", ac.get("cheat_detected")) else "NO (ROBUST)"
            disp_name = name if len(name) <= 38 else "..." + name[-35:]
            print(f"{disp_name:<38} | {k1:<16} | {k2:<16} | {k4:<16} | {cheat}")

    if mode in ["functional", "full"]:
        print("\n" + "-" * 115)
        print("  FUNCTIONAL PASS@1 & SYNTAX VALIDITY:")
        print(f"{'Model / Checkpoint':<38} | {'Pass@1 (%)':<11} | {'AST Valid (%)':<13} | {'Syntax Errors':<13} | {'Assertion Fails'}")
        print("-" * 115)
        for name, rep in multi_reports.items():
            fn = rep.get("functional", {})
            p1 = f"{fn.get('pass_at_1_pct', 0.0):.1f}%"
            ast = f"{fn.get('ast_validity_pct', 0.0):.1f}%"
            outcomes = fn.get("execution_outcomes", {})
            syn_err = outcomes.get("SYNTAX_ERROR", 0)
            ast_fail = outcomes.get("FAILED_ASSERTION", 0)
            disp_name = name if len(name) <= 38 else "..." + name[-35:]
            print(f"{disp_name:<38} | {p1:<11} | {ast:<13} | {syn_err:<13} | {ast_fail}")

    print("=" * 115 + "\n")


def evaluate(
    checkpoint: str | Path | list[str | Path] | tuple[str | Path, ...],
    mode: str = "probes",
    suite: str = "private_unseen",
    probe_type: str = "both",
    num_probes: int = 100,
    max_tasks: Optional[int] = None,
    timeout: float = 3.0,
    tokenizer_path: str | Path | None = None,
    output_path: Optional[str | Path] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Master programmatic entrypoint for Télos evaluation engine.
    Supports evaluating a single checkpoint or multiple checkpoints side-by-side.
    """
    if isinstance(checkpoint, (list, tuple)):
        checkpoints = list(checkpoint)
    else:
        checkpoints = [checkpoint]

    if len(checkpoints) == 1:
        rep = _evaluate_single(
            checkpoint=checkpoints[0],
            mode=mode,
            suite=suite,
            probe_type=probe_type,
            num_probes=num_probes,
            max_tasks=max_tasks,
            timeout=timeout,
            tokenizer_path=tokenizer_path,
            output_path=output_path,
            **kwargs
        )
        if not output_path:
            out_file = PROJECT_ROOT / "logs" / f"eval_report_{mode}_{int(time.time())}.json"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, "w") as f:
                json.dump(rep, f, indent=2)
            print(f"✓ Saved evaluation report to {out_file}\n")
        return rep

    # Multi-model evaluation workflow
    print("\n" + "=" * 80)
    print(f"  TÉLOS MULTI-MODEL BENCHMARK: EVALUATING {len(checkpoints)} MODELS ({mode.upper()})")
    print("=" * 80)

    multi_reports: Dict[str, Any] = {}
    for idx, cp in enumerate(checkpoints, 1):
        cp_path = Path(cp)
        rel_parts = cp_path.parts
        if "corosred" in rel_parts:
            sub = cp_path.parent.name if cp_path.is_file() else cp_path.name
            name = f"corosred/{sub}"
        elif "ar" in rel_parts:
            sub = cp_path.parent.name if cp_path.is_file() else cp_path.name
            name = f"ar/{sub}" if sub != "ar" else "ar/15m"
        else:
            name = cp_path.parent.name if cp_path.is_file() else cp_path.name
        if name in multi_reports:
            name = f"{name}_{idx}"

        print(f"\n[{idx}/{len(checkpoints)}] Evaluating {name} ({cp})...")
        single_rep = _evaluate_single(
            checkpoint=cp,
            mode=mode,
            suite=suite,
            probe_type=probe_type,
            num_probes=num_probes,
            max_tasks=max_tasks,
            timeout=timeout,
            tokenizer_path=tokenizer_path,
            output_path=None,
            **kwargs
        )
        multi_reports[name] = single_rep

    # Print comparative scorecard
    print_multimodel_scorecard(multi_reports, mode=mode)

    # Save consolidated report
    out_file = Path(output_path) if output_path else (PROJECT_ROOT / "logs" / f"eval_report_multimodel_{mode}_{int(time.time())}.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(multi_reports, f, indent=2)
    print(f"✓ Saved consolidated multi-model evaluation report to {out_file}\n")

    return multi_reports


def main():
    parser = argparse.ArgumentParser(description="Télos Next-Generation Evaluation Suite")
    parser.add_argument(
        "--checkpoint", "--checkpoints",
        type=str,
        nargs="+",
        dest="checkpoints",
        required=True,
        help="One or more paths to checkpoint file(s) (.safetensors, .pt) or directory(ies)"
    )
    parser.add_argument("--mode", type=str, default="probes", choices=["probes", "functional", "anticheat", "full", "sample"], help="Evaluation mode")
    parser.add_argument("--suite", type=str, default="private_unseen", choices=["private_unseen", "public_standard"], help="Benchmark suite track")
    parser.add_argument("--probe-type", type=str, default="both", choices=["infill", "causal", "both"], help="Probe benchmark type")
    parser.add_argument("--num-probes", type=int, default=100, help="Number of contextual probes to evaluate")
    parser.add_argument("--max-tasks", type=int, default=None, help="Maximum number of functional execution tasks")
    parser.add_argument("--timeout", type=float, default=3.0, help="Sandbox subprocess timeout in seconds")
    parser.add_argument("--tokenizer", type=str, default=None, help="Path to BPE tokenizer JSON")
    parser.add_argument("--output", type=str, default=None, help="Path to save output JSON report")
    args = parser.parse_args()

    evaluate(
        checkpoint=args.checkpoints,
        mode=args.mode,
        suite=args.suite,
        probe_type=args.probe_type,
        num_probes=args.num_probes,
        max_tasks=args.max_tasks,
        timeout=args.timeout,
        tokenizer_path=args.tokenizer,
        output_path=args.output
    )


if __name__ == "__main__":
    main()

