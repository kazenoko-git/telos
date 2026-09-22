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
import re
import json
import time
import math
import ast
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from .probes import load_contextual_probes, PROBE_SUITE_100
from .syntax import check_ast_validity, categorize_syntax_error, analyze_syntax_batch
from .executor import execute_code_sandboxed, ExecutionResult
from .anticheat import evaluate_span_infill_probe, summarize_anticheat_suite
from .linguistic import evaluate_linguistic, load_english_probes
from .tooluse import evaluate_tooluse, load_tooluse_suite
from .stats import bootstrap_confidence_interval
from .metrics import analyze_task_completion, aggregate_extended_metrics
from .adapters import BaseModelAdapter, load_adapter
from .polyglot_eval import execute_polyglot_task, clean_polyglot_completion
from .reasoning_eval import evaluate_gsm8k_sample, evaluate_arc_sample, evaluate_competition_math_sample
from .cyber_eval import evaluate_cybersecurity_challenge
from telos.data.tokenizer import load_tokenizer
from telos.models import TelosConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
        with safe_open(str(cp), framework="pt") as f:
            keys = list(f.keys())
            layer_indices = [int(k.split(".")[1]) for k in keys if "layers." in k and k.split(".")[1].isdigit()]
            if layer_indices:
                n_layers = max(layer_indices) + 1
            if "emb.weight" in keys:
                emb_shape = f.get_slice("emb.weight").get_shape()
                vocab_size = emb_shape[0]
                d_model = emb_shape[1]
            if "layers.0.qkv_proj.weight" in keys:
                qkv_shape = f.get_slice("layers.0.qkv_proj.weight").get_shape()
                qkv_out = qkv_shape[0]
                # QKV projection = 3 * n_heads * head_dim (head_dim = 64)
                total_heads = qkv_out // 64
                n_heads = max(1, total_heads // 3)
                n_kv_heads = n_heads

        if not use_reliability_head and any("reliability_head" in k for k in keys):
            use_reliability_head = True

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

        layer_indices = [int(k.split(".")[1]) for k in sd.keys() if "layers." in k and k.split(".")[1].isdigit()]
        if layer_indices:
            n_layers = max(layer_indices) + 1

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


def clean_functional_completion(prompt: str, raw_completion: str) -> str:
    """
    Cleans model completion for functional code execution:
    1. Extracts code from markdown code fences (```python ... ```) if generated by chat models.
    2. Deduplicates function signature line if repeated by instruction models.
    3. Truncates at the start of any new top-level function or class definition.
    4. Trims trailing incomplete lines caused by max_token limits to maximize valid AST recovery.
    """
    comp = raw_completion.rstrip()

    # 1. Extract from markdown code fences if present
    if "```" in raw_completion:
        blocks = re.findall(r"```(?:python)?\s*(.*?)```", raw_completion, re.DOTALL)
        if blocks:
            candidate = blocks[0].strip()
            # If candidate is a standalone complete Python function definition
            try:
                tree = ast.parse(candidate)
                if any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in ast.walk(tree)):
                    prompt_imports = [
                        l for l in prompt.splitlines()
                        if l.strip().startswith("import ") or l.strip().startswith("from ")
                    ]
                    if prompt_imports and not any(candidate.startswith(imp.split()[0]) for imp in prompt_imports):
                        return "\n".join(prompt_imports) + "\n" + candidate
                    return candidate
            except SyntaxError:
                pass
            comp = candidate
        else:
            lines = raw_completion.splitlines()
            code_lines = [l for l in lines if not l.strip().startswith("```")]
            comp = "\n".join(code_lines).strip()

    # 2. Deduplicate signature if model repeated the last line of prompt
    prompt_lines = [l.strip() for l in prompt.splitlines() if l.strip()]
    if prompt_lines:
        last_prompt_line = prompt_lines[-1]
        comp_lines = comp.splitlines(keepends=True)
        if comp_lines and comp_lines[0].strip() == last_prompt_line:
            comp = "".join(comp_lines[1:])
        elif comp_lines and comp_lines[0].strip().startswith("def ") and last_prompt_line.startswith("def "):
            comp = "".join(comp_lines[1:])

    # 3. Truncate at next top-level statement
    stop_phrases = ["\ndef ", "\nclass ", "\nif __name__", "\nprint("]
    for sp in stop_phrases:
        if sp in comp:
            comp = comp[:comp.index(sp)]

    # 4. AST back-trimming with compile() validation and indentation recovery
    import textwrap
    for candidate_text in [comp, textwrap.indent(comp, "    ")]:
        lines = candidate_text.splitlines(keepends=True)
        while lines:
            test_code = prompt + ("\n" if not prompt.endswith("\n") else "") + "".join(lines)
            try:
                compile(test_code, "<string>", "exec")
                return "".join(lines)
            except SyntaxError:
                lines.pop()

    return comp


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

    # Sequence length guardrail for models with fixed max_seq_len (e.g. 512)
    max_seq_len = getattr(getattr(model, "config", None), "max_seq_len", 512)
    if len(p_ids) >= max_seq_len - 16:
        # Left-truncate prompt to preserve function signature & docstring tail
        p_ids = p_ids[-(max_seq_len - 16):]

    curr_ids = list(p_ids)
    stop_set = set(stop_tokens or [0, 3])  # EOS / PAD
    stop_words = ["\ndef ", "\nclass ", "\nif __name__"]

    # Bound max generation so sequence length never exceeds max_seq_len
    actual_max_new = min(max_new_tokens, max_seq_len - len(curr_ids))

    for _ in range(actual_max_new):
        if backend == "mlx":
            import mlx.core as mx
            x = mx.array([curr_ids], dtype=mx.int32)
            logits = model(x)
            next_tok = int(np.argmax(np.array(logits[0, -1].astype(mx.float32))))
        else:
            import torch
            device = next(model.parameters()).device if hasattr(model, "parameters") else torch.device("cpu")
            x = torch.tensor([curr_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                logits = model(x)
            next_tok = int(torch.argmax(logits[0, -1]).item())

        if next_tok in stop_set:
            break
        curr_ids.append(next_tok)

        # Early stopping on function boundary
        cur_text = tokenizer.decode(curr_ids[len(p_ids):])
        if any(sw in cur_text for sw in stop_words):
            break

    # Decode only the newly generated continuation tokens
    continuation_ids = curr_ids[len(p_ids):]
    return tokenizer.decode(continuation_ids)


def evaluate_functional(
    model,
    tokenizer,
    backend: str,
    suite: str = "private_unseen",
    prompt_mode: str = "base",
    max_tasks: Optional[int] = None,
    timeout_seconds: float = 3.0,
    max_new_tokens: int = 128
) -> Dict[str, Any]:
    """
    Executes functional unit testing benchmark (Pass@1 with sandboxed subprocesses).
    Supports Python, Polyglot (.NET, Java, JS, TS, Rust, React), Math (GSM8K, MATH),
    Science (ARC-Challenge), and Cybersecurity.
    """
    bench_dir = PROJECT_ROOT / "evals" / "benchmarks"
    suite_files = {
        "humaneval": bench_dir / "humaneval_suite.json",
        "public_standard": bench_dir / "humaneval_suite.json",  # Backward-compatible alias
        "mbpp": bench_dir / "mbpp_suite.json",
        "private_unseen": bench_dir / "private_unseen_suite.json",
        "private_unseen_base": bench_dir / "private_unseen_base.json",
        "private_unseen_hint": bench_dir / "private_unseen_hint.json",
        "humaneval_cs": bench_dir / "humaneval_csharp_suite.json",
        "humaneval_csharp": bench_dir / "humaneval_csharp_suite.json",
        "humaneval_java": bench_dir / "humaneval_java_suite.json",
        "humaneval_js": bench_dir / "humaneval_js_suite.json",
        "humaneval_javascript": bench_dir / "humaneval_js_suite.json",
        "humaneval_ts": bench_dir / "humaneval_ts_suite.json",
        "humaneval_typescript": bench_dir / "humaneval_ts_suite.json",
        "humaneval_rust": bench_dir / "humaneval_rust_suite.json",
        "react": bench_dir / "react_javascript_suite.json",
        "react_javascript": bench_dir / "react_javascript_suite.json",
        "gsm8k": bench_dir / "gsm8k_suite.json",
        "arc": bench_dir / "arc_challenge_suite.json",
        "arc_challenge": bench_dir / "arc_challenge_suite.json",
        "gpqa": bench_dir / "gpqa_diamond_suite.json",
        "gpqa_diamond": bench_dir / "gpqa_diamond_suite.json",
        "mmlu": bench_dir / "mmlu_science_suite.json",
        "mmlu_science": bench_dir / "mmlu_science_suite.json",
        "math": bench_dir / "competition_math_suite.json",
        "competition_math": bench_dir / "competition_math_suite.json",
        "cyber": bench_dir / "cybersecurity_suite.json",
        "cybersecurity": bench_dir / "cybersecurity_suite.json",
        "tooluse": bench_dir / "tooluse_suite.json",
    }
    data_file = suite_files.get(suite, bench_dir / f"{suite}_suite.json")

    if suite == "private_unseen_hint":
        prompt_mode = "hint"
    elif suite == "private_unseen_base":
        prompt_mode = "base"

    if not data_file.exists():
        raise FileNotFoundError(f"Benchmark dataset not found at {data_file}. Run scripts/build_evaluation_benchmarks.py first.")

    with open(data_file, "r") as f:
        tasks = json.load(f)

    if max_tasks and len(tasks) > max_tasks:
        # Balanced sampling across categories
        cats = {}
        for t in tasks:
            cats.setdefault(t.get("category", t.get("domain", "General")), []).append(t)
        sampled = []
        per_cat = max(1, max_tasks // max(1, len(cats)))
        for cat, cat_tasks in cats.items():
            sampled.extend(cat_tasks[:per_cat])
        tasks = sampled[:max_tasks]

    mode_label = f" ({prompt_mode.upper()} PROMPTS)" if "private_unseen" in suite else ""
    print("\n" + "=" * 80)
    print(f"  TÉLOS BENCHMARK EVALUATION: {suite.upper()}{mode_label} ({len(tasks)} TASKS)")
    print(f"  Subprocess Sandbox: spawn | Timeout: {timeout_seconds}s | Decoding: Greedy (T=0.0)")
    print("=" * 80)

    category_stats: Dict[str, Dict[str, Any]] = {}
    task_results = []
    outcome_counts = {k.value: 0 for k in ExecutionResult}
    ast_valid_count = 0

    # Track classification
    is_gsm8k = "gsm8k" in suite
    is_arc = any(x in suite for x in ["arc", "gpqa", "mmlu"])
    is_math = "math" in suite
    is_cyber = "cyber" in suite
    is_polyglot = any(x in suite for x in ["csharp", "cs", "java", "js", "javascript", "ts", "typescript", "rust", "react"])

    for idx, task in enumerate(tasks, 1):
        cat = task.get("category", task.get("subject", task.get("domain", "General")))
        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "passed": 0, "ast_valid": 0, "outcomes": {}}

        # Select prompt variant based on prompt_mode
        if prompt_mode == "hint" and task.get("prompt_hint"):
            prompt = task["prompt_hint"]
        elif prompt_mode == "base" and task.get("prompt_base"):
            prompt = task["prompt_base"]
        else:
            prompt = task["prompt"]

        test_harness = task.get("test_harness", "")

        # 1. Allocate token budget: adapter models require at least 2048 tokens
        if isinstance(model, BaseModelAdapter):
            token_budget = max(max_new_tokens, 2048)
        elif any(x in suite for x in ["gpqa", "competition_math", "cyber", "mmlu"]):
            token_budget = max(max_new_tokens, 1024)
        elif (is_arc or is_math or is_gsm8k):
            token_budget = max(max_new_tokens, 384)
        else:
            token_budget = max_new_tokens
        if isinstance(model, BaseModelAdapter):
            raw_completion = model.generate(
                prompt=prompt,
                max_new_tokens=token_budget,
                temperature=0.0,
                stop=task.get("stop_tokens")
            )
        else:
            raw_completion = _generate_greedy_completion(
                model=model,
                tokenizer=tokenizer,
                backend=backend,
                prompt=prompt,
                max_new_tokens=token_budget
            )

        # 2. Domain-Specific Execution & Evaluation
        if is_gsm8k:
            is_passed, details = evaluate_gsm8k_sample(raw_completion, task.get("target_answer", ""))
            outcome = ExecutionResult.PASSED if is_passed else ExecutionResult.FAILED_ASSERTION
            is_ast_valid = True
            ast_err = ""
            completion = raw_completion
            full_candidate_code = raw_completion

        elif is_arc:
            is_passed, details = evaluate_arc_sample(raw_completion, task.get("answer_key", ""))
            outcome = ExecutionResult.PASSED if is_passed else ExecutionResult.FAILED_ASSERTION
            is_ast_valid = True
            ast_err = ""
            completion = raw_completion
            full_candidate_code = raw_completion

        elif is_math:
            is_passed, details = evaluate_competition_math_sample(raw_completion, task.get("target_answer", ""))
            outcome = ExecutionResult.PASSED if is_passed else ExecutionResult.FAILED_ASSERTION
            is_ast_valid = True
            ast_err = ""
            completion = raw_completion
            full_candidate_code = raw_completion

        elif is_cyber:
            eval_meta = evaluate_cybersecurity_challenge(raw_completion, task)
            is_passed = eval_meta["overall_passed"]
            outcome = ExecutionResult.PASSED if is_passed else ExecutionResult.FAILED_ASSERTION
            details = f"CWE={eval_meta['cwe_detected']}, RemScore={eval_meta['remediation_score']:.2f}"
            is_ast_valid = True
            ast_err = ""
            completion = raw_completion
            full_candidate_code = raw_completion

        elif is_polyglot:
            lang = task.get("language", suite.split("_")[-1])
            outcome, details = execute_polyglot_task(
                language=lang,
                prompt=prompt,
                completion=raw_completion,
                test_harness=test_harness,
                timeout_seconds=timeout_seconds,
                required_keywords=task.get("required_constructs")
            )
            is_passed = (outcome == ExecutionResult.PASSED)
            is_ast_valid = is_passed
            ast_err = "" if is_passed else details
            completion = clean_polyglot_completion(lang, prompt, raw_completion)
            full_candidate_code = prompt + "\n" + completion

        else:
            # Standard Python code execution
            completion = clean_functional_completion(prompt, raw_completion)
            # Use standalone definition directly if candidate already contains the target function
            try:
                tree = ast.parse(completion)
                if any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in ast.walk(tree)):
                    full_candidate_code = completion
                else:
                    full_candidate_code = prompt + ("\n" if not prompt.endswith("\n") else "") + completion
            except SyntaxError:
                full_candidate_code = prompt + ("\n" if not prompt.endswith("\n") else "") + completion

            is_ast_valid, ast_err = check_ast_validity(full_candidate_code)
            outcome, details = execute_code_sandboxed(
                code=full_candidate_code,
                test_harness=test_harness,
                timeout_seconds=timeout_seconds
            )
            is_passed = (outcome == ExecutionResult.PASSED)

        if is_ast_valid:
            ast_valid_count += 1
            category_stats[cat]["ast_valid"] += 1

        # 3. Extended generation dynamics & numerical accuracy
        comp_metrics = analyze_task_completion(
            prompt=prompt,
            raw_completion=raw_completion,
            clean_completion=completion,
            tokenizer=tokenizer
        )

        outcome_counts[outcome.value] = outcome_counts.get(outcome.value, 0) + 1
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
            "extended_metrics": comp_metrics,
        })

        if idx % 10 == 0 or idx == len(tasks):
            print(f"  [{idx}/{len(tasks)}] Completed... Current Pass@1: {round(outcome_counts[ExecutionResult.PASSED.value]/idx*100, 1)}%")

    total_tasks = len(tasks)
    passed_total = outcome_counts[ExecutionResult.PASSED.value]
    overall_pass1 = (passed_total / max(total_tasks, 1)) * 100.0
    overall_ast_valid = (ast_valid_count / max(total_tasks, 1)) * 100.0

    ci_pass1 = bootstrap_confidence_interval([1.0 if r["outcome"] == ExecutionResult.PASSED.value else 0.0 for r in task_results])

    # Aggregate extended repetition and numerical metrics
    all_extended = [r["extended_metrics"] for r in task_results]
    overall_ext = aggregate_extended_metrics(all_extended)

    print("\n" + "-" * 105)
    print(f"  {'Category':<28} | {'Total':<5} | {'Pass@1 (%)':<10} | {'AST Valid (%)':<13} | {'Rep-3 (%)':<9} | {'Num Recall':<10} | {'Num Prec'}")
    print("-" * 105)
    cat_summary = {}
    for cat, s in category_stats.items():
        cnt = s["total"]
        pass_pct = round((s["passed"] / cnt) * 100.0, 1) if cnt else 0.0
        ast_pct = round((s["ast_valid"] / cnt) * 100.0, 1) if cnt else 0.0
        cat_ext = aggregate_extended_metrics([r["extended_metrics"] for r in task_results if r["category"] == cat])
        
        rep3_pct = cat_ext.get("rep_3gram_pct", 0.0)
        num_rec = cat_ext.get("numerical", {}).get("numeric_recall_pct")
        num_prc = cat_ext.get("numerical", {}).get("numeric_precision_pct")
        rec_str = f"{num_rec:.1f}%" if num_rec is not None else "N/A"
        prc_str = f"{num_prc:.1f}%" if num_prc is not None else "N/A"

        cat_summary[cat] = {
            "count": cnt,
            "pass_at_1_pct": pass_pct,
            "ast_valid_pct": ast_pct,
            "outcomes": s["outcomes"],
            "extended_metrics": cat_ext,
        }
        print(f"  {cat:<28} | {cnt:<5d} | {pass_pct:>9.1f}% | {ast_pct:>12.1f}% | {rep3_pct:>8.1f}% | {rec_str:>10} | {prc_str}")

    print("-" * 105)
    print(f"  OVERALL PASS@1:       {overall_pass1:.2f}% (95% CI: [{ci_pass1[0]}%, {ci_pass1[1]}%])")
    print(f"  OVERALL AST VALIDITY: {overall_ast_valid:.2f}%")
    print(f"  Execution Breakdown:  {outcome_counts}")

    print("\n  [GENERATION DYNAMICS & REPETITION METRICS]")
    print(f"  Avg Token Length:     {overall_ext.get('avg_token_length', 0.0)} tokens | Early Exits (<=4 tok): {overall_ext.get('early_exit_pct', 0.0)}%")
    print(f"  N-Gram Repetition:    2-gram: {overall_ext.get('rep_2gram_pct', 0.0)}% | 3-gram: {overall_ext.get('rep_3gram_pct', 0.0)}% | 4-gram: {overall_ext.get('rep_4gram_pct', 0.0)}%")
    print(f"  Redundancy & Loops:   Line Repetition: {overall_ext.get('line_repetition_pct', 0.0)}% | Consecutive Dups: {overall_ext.get('consecutive_dup_line_pct', 0.0)}% | Degenerate Loops: {overall_ext.get('degenerate_loop_pct', 0.0)}%")

    num_stats = overall_ext.get("numerical", {})
    if num_stats.get("tasks_with_prompt_numbers", 0) > 0:
        print("\n  [NUMERICAL & CONSTANT RETENTION (TASKS WITH PROMPT/DOCSTRING NUMBERS)]")
        print(f"  Evaluated Tasks:      {num_stats.get('tasks_with_prompt_numbers')} tasks containing explicit numeric specifications")
        print(f"  Numeric Recall:       {num_stats.get('numeric_recall_pct', 0.0)}% of required numbers appeared in completions")
        print(f"  Numeric Precision:    {num_stats.get('numeric_precision_pct', 0.0)}% of generated numbers were ground-truth matches")
        print(f"  Exact Constant Match: {num_stats.get('exact_number_match_pct', 0.0)}% of tasks preserved ALL required numbers")
        print(f"  Zero Numbers Emitted: {num_stats.get('zero_numbers_emitted_pct', 0.0)}% of tasks omitted numbers entirely (e.g. 'return val')")
        print(f"  Hallucinated Numbers: {num_stats.get('hallucinated_numbers_pct', 0.0)}% of tasks generated invented numbers")
    print("=" * 105 + "\n")

    return {
        "suite": suite,
        "total_tasks": total_tasks,
        "pass_at_1_pct": round(overall_pass1, 2),
        "pass_at_1_95ci": ci_pass1,
        "ast_validity_pct": round(overall_ast_valid, 2),
        "execution_outcomes": outcome_counts,
        "repetition": {
            "avg_token_length": overall_ext.get("avg_token_length"),
            "early_exit_pct": overall_ext.get("early_exit_pct"),
            "rep_2gram_pct": overall_ext.get("rep_2gram_pct"),
            "rep_3gram_pct": overall_ext.get("rep_3gram_pct"),
            "rep_4gram_pct": overall_ext.get("rep_4gram_pct"),
            "line_repetition_pct": overall_ext.get("line_repetition_pct"),
            "consecutive_dup_line_pct": overall_ext.get("consecutive_dup_line_pct"),
            "degenerate_loop_pct": overall_ext.get("degenerate_loop_pct"),
        },
        "numerical_accuracy": num_stats,
        "extended_metrics": overall_ext,
        "category_breakdown": cat_summary,
        "tasks": task_results,
    }


def evaluate_anticheat(
    model,
    tokenizer,
    backend: str,
    paradigm: str = "corosred",
    num_probes: int = 100,
    span_lengths: Optional[List[int]] = None
) -> Dict[str, Any]:
    """Evaluates multi-token chunk masking and suffix-copying cheat rates."""
    if str(paradigm).lower() == "ar":
        print("\n" + "=" * 80)
        print("  TÉLOS ANTI-CHEAT & SUFFIX-COPY BENCHMARK")
        print("  [Notice] Model is AR (causal-only). Bidirectional infill and suffix copy N/A.")
        print("=" * 80 + "\n")
        return {
            "status": "not_applicable",
            "reason": "AR models are causal-only and cannot condition on bidirectional suffixes",
            "is_suspect_cheater": False,
            "span_breakdown": {
                "span_1": {"suffix_copy_rate_pct": None, "exact_match_pct": None, "token_accuracy_pct": None},
                "span_2": {"suffix_copy_rate_pct": None, "exact_match_pct": None, "token_accuracy_pct": None},
                "span_4": {"suffix_copy_rate_pct": None, "exact_match_pct": None, "token_accuracy_pct": None},
                "span_8": {"suffix_copy_rate_pct": None, "exact_match_pct": None, "token_accuracy_pct": None},
            }
        }

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
        cheat_mode = summary.get("cheat_mode")
        if cheat_mode == "degenerate_copy":
            print("  [!] FAILED (DEGENERATE): Model exhibits degenerate suffix-copying with near-zero accuracy.")
        else:
            print("  [!] WARNING: Model exhibits high boundary suffix-copying cheat characteristics!")
    else:
        s1_acc = summary.get("span_breakdown", {}).get("span_1", {}).get("exact_match_pct", 0.0)
        if s1_acc < 5.0:
            print("  [?] UNCONVERGED: Model infilling accuracy is near-zero; representations not yet formed.")
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
    benchmark_type: str = "all",
    language: str = "auto",
    **kwargs
) -> Dict[str, Any]:
    """Evaluates a single model checkpoint across requested benchmark types."""
    b_type = str(kwargs.get("type", benchmark_type)).lower()
    lang = str(language).lower()

    # Determine which benchmark tracks to run
    if b_type == "all":
        types_to_run = ["code", "math", "science", "cyber", "tooluse", "linguistic"]
    elif b_type in ["code", "math", "science", "cyber", "tooluse", "linguistic"]:
        types_to_run = [b_type]
    else:
        types_to_run = [b_type]

    # Universal Model Loading: Route to Télos Native, OpenAI API, Gemini API, or Hugging Face
    adapter = None
    if isinstance(checkpoint, BaseModelAdapter):
        adapter = checkpoint
        model = adapter
        backend = adapter.backend_name
        vocab_size = getattr(adapter, "vocab_size", 32000)
    elif (
        kwargs.get("api_base")
        or kwargs.get("api_key")
        or kwargs.get("backend") in ["openai_api", "gemini_api", "huggingface", "mlx_lm", "swift", "swift_afm"]
        or any(x in str(checkpoint).lower() for x in ["afm-3", "afm", "ternary-bonsai", "gemma", "gemini", "http://", "https://"])
    ):
        adapter = load_adapter(
            model_identifier=str(checkpoint),
            backend=kwargs.get("backend"),
            api_base=kwargs.get("api_base"),
            api_key=kwargs.get("api_key"),
            concurrency=kwargs.get("concurrency", 8),
            device=kwargs.get("device", "auto"),
            torch_dtype=kwargs.get("torch_dtype", "bfloat16"),
            quantization=kwargs.get("quantization"),
        )
        model = adapter
        backend = adapter.backend_name
        vocab_size = getattr(adapter, "vocab_size", 32000)
    else:
        # Local Télos Checkpoint
        model, backend, vocab_size = load_model_from_checkpoint(checkpoint)

    if tokenizer_path is None:
        if vocab_size == 8192 and (PROJECT_ROOT / "configs" / "tokenizer_mac.json").exists():
            tokenizer_path = str(PROJECT_ROOT / "configs" / "tokenizer_mac.json")
        elif (PROJECT_ROOT / "configs" / "shared" / "tokenizer_0.json").exists():
            tokenizer_path = str(PROJECT_ROOT / "configs" / "shared" / "tokenizer_0.json")
    try:
        tok = load_tokenizer(str(tokenizer_path) if tokenizer_path else None)
    except Exception:
        tok = None

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
        "benchmark_type": b_type,
        "language": lang,
        "suites_evaluated": types_to_run,
    }

    # 1. Code Benchmarks (Python & Polyglot)
    if "code" in types_to_run or suite in [
        "humaneval", "mbpp", "private_unseen", "private_unseen_base", "private_unseen_hint",
        "humaneval_cs", "humaneval_java", "humaneval_js", "humaneval_ts", "humaneval_rust",
        "react", "react_javascript"
    ]:
        code_report = {}
        prompt_mode = str(kwargs.get("prompt_mode", "base")).lower()
        if mode == "probes":
            if (hasattr(model, "parameters") or backend in ["mlx", "pytorch"]) and tok is not None:
                code_report["probes"] = evaluate_probes(
                    model, tok, backend, paradigm=paradigm, num_probes=num_probes, probe_type=probe_type
                )
            else:
                print("Notice: Probes require local logits/logprobs. Skipping probes for black-box API.")
        elif mode == "functional":
            code_report["functional"] = evaluate_functional(
                model, tok, backend, suite=suite, prompt_mode=prompt_mode, max_tasks=max_tasks, timeout_seconds=timeout
            )
        elif mode == "anticheat":
            if tok is not None:
                code_report["anticheat"] = evaluate_anticheat(model, tok, backend, paradigm=paradigm, num_probes=min(num_probes, 200))
        elif mode == "sample":
            if tok is not None:
                evaluate_sample(model, tok, backend)
                code_report["sample"] = {"status": "completed"}
        elif mode == "full":
            if (hasattr(model, "parameters") or backend in ["mlx", "pytorch"]) and tok is not None:
                code_report["probes"] = evaluate_probes(
                    model, tok, backend, paradigm=paradigm, num_probes=num_probes, probe_type=probe_type
                )
            code_report["functional"] = evaluate_functional(
                model, tok, backend, suite=suite, prompt_mode=prompt_mode, max_tasks=max_tasks, timeout_seconds=timeout
            )
        report["code"] = code_report
        for k, v in code_report.items():
            report[k] = v

    # 2. Math Benchmarks (GSM8K, Competition MATH)
    if ("math" in types_to_run or suite in ["gsm8k", "math", "competition_math"]) and mode in ["functional", "full"]:
        m_suite = suite if suite in ["gsm8k", "competition_math", "math"] else "gsm8k"
        report["math"] = evaluate_functional(
            model, tok, backend, suite=m_suite, max_tasks=max_tasks, timeout_seconds=timeout
        )

    # 3. Science Benchmarks (ARC-Challenge, GPQA Diamond, MMLU Science)
    if ("science" in types_to_run or any(x in suite for x in ["arc", "arc_challenge", "gpqa", "gpqa_diamond", "mmlu", "mmlu_science"])) and mode in ["functional", "full"]:
        s_suite = suite if suite in ["arc", "arc_challenge", "gpqa", "gpqa_diamond", "mmlu", "mmlu_science"] else "arc"
        report["science"] = evaluate_functional(
            model, tok, backend, suite=s_suite, max_tasks=max_tasks, timeout_seconds=timeout
        )

    # 4. Cybersecurity Benchmarks
    if ("cyber" in types_to_run or suite in ["cyber", "cybersecurity"]) and mode in ["functional", "full"]:
        report["cyber"] = evaluate_functional(
            model, tok, backend, suite="cyber", max_tasks=max_tasks, timeout_seconds=timeout
        )

    clean_kwargs = {
        k: v for k, v in kwargs.items()
        if k not in ["backend", "tokenizer", "model", "max_tasks", "paradigm", "language", "mode", "num_probes", "probe_type"]
    }

    # 5. Tool-Use Benchmarks
    if ("tooluse" in types_to_run or suite == "tooluse") and mode in ["functional", "full"]:
        report["tooluse"] = evaluate_tooluse(
            model=model,
            tokenizer=tok,
            backend=backend,
            max_tasks=max_tasks,
            **clean_kwargs
        )

    # 6. Linguistic Benchmarks
    if "linguistic" in types_to_run and tok is not None and (hasattr(model, "parameters") or backend in ["mlx", "pytorch"]):
        target_lang = "english" if lang in ["auto", "english"] else lang
        ling_mode = mode if mode in ["probes", "perplexity", "sample", "full"] else "probes"
        report["linguistic"] = evaluate_linguistic(
            model=model,
            tokenizer=tok,
            backend=backend,
            paradigm=paradigm,
            language=target_lang,
            mode=ling_mode,
            num_probes=num_probes,
            probe_type=probe_type,
            **clean_kwargs
        )

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

    has_code = any("code" in rep or "probes" in rep for rep in multi_reports.values())
    has_linguistic = any("linguistic" in rep for rep in multi_reports.values())
    has_tooluse = any("tooluse" in rep for rep in multi_reports.values())

    if has_code and mode in ["functional", "full"]:
        header = (
            f"{'Model / Checkpoint':<38} | {'Backend':<7} | {'Suite':<18} | {'Pass@1 (%)':<11} | "
            f"{'AST Valid (%)':<14} | {'Rep-3 (%)':<10} | {'Passed/Total'}"
        )
        print("\n  [CODE BENCHMARK: FUNCTIONAL EXECUTION & PASS@1]")
        print(header)
        print("-" * 115)
        for name, rep in multi_reports.items():
            b = rep.get("backend", "unknown")
            fn = rep.get("code", {}).get("functional") or rep.get("functional", {})
            suite_name = fn.get("suite", "unknown")
            p1 = f"{fn.get('pass_at_1_pct', 0.0):.1f}%" if fn.get('pass_at_1_pct') is not None else "N/A"
            ast_val = f"{fn.get('ast_validity_pct', 0.0):.1f}%" if fn.get('ast_validity_pct') is not None else "N/A"
            rep3 = f"{fn.get('extended_metrics', {}).get('rep_3gram_pct', 0.0):.1f}%" if fn.get('extended_metrics') else "N/A"
            passed = fn.get("execution_outcomes", {}).get("PASSED", 0)
            total = fn.get("total_tasks", 0)
            p_str = f"{passed}/{total}"
            disp_name = name if len(name) <= 38 else "..." + name[-35:]
            print(f"{disp_name:<38} | {b:<7} | {suite_name:<18} | {p1:<11} | {ast_val:<14} | {rep3:<10} | {p_str}")

    if has_code and mode in ["probes", "full"]:
        header = (
            f"{'Model / Checkpoint':<38} | {'Backend':<7} | {'Infill Top1':<11} | {'Infill CE':<9} | "
            f"{'Causal Top1':<11} | {'Causal CE':<9} | {'Infill Rank':<11}"
        )
        print("\n  [CODE BENCHMARK: CONTEXTUAL PROBES]")
        print(header)
        print("-" * 115)
        for name, rep in multi_reports.items():
            b = rep.get("backend", "unknown")
            pr = rep.get("code", {}).get("probes") or rep.get("probes", {})
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

    if has_linguistic:
        print("\n  [ENGLISH LINGUISTIC BENCHMARK: SYNTAX & COMMON SENSE]")
        ling_header = (
            f"{'Model / Checkpoint':<38} | {'Backend':<7} | {'Causal Top1':<11} | {'Infill Top1':<11} | "
            f"{'Causal CE':<10} | {'Perplexity':<10}"
        )
        print(ling_header)
        print("-" * 115)
        for name, rep in multi_reports.items():
            b = rep.get("backend", "unknown")
            lr = rep.get("linguistic", {})
            pr = lr.get("probes", {})
            cau = pr.get("causal", {}).get("overall", {})
            inf = pr.get("infill", {}).get("overall", {})
            c_t1 = f"{cau.get('top1_acc_pct', 0.0):.1f}%" if cau.get('top1_acc_pct') is not None else "N/A"
            c_ce = f"{cau.get('mean_ce', 0.0):.2f}" if cau.get('mean_ce') is not None else "N/A"
            if pr.get("infill", {}).get("status") == "not_applicable":
                i_t1 = "N/A (AR)"
            else:
                i_t1 = f"{inf.get('top1_acc_pct', 0.0):.1f}%" if inf.get('top1_acc_pct') is not None else "N/A"
            ppl_val = lr.get("perplexity", {}).get("overall_ppl")
            ppl_str = f"{ppl_val:.2f}" if ppl_val is not None else "N/A"
            disp_name = name if len(name) <= 38 else "..." + name[-35:]
            print(f"{disp_name:<38} | {b:<7} | {c_t1:<11} | {i_t1:<11} | {c_ce:<10} | {ppl_str:<10}")

    if has_tooluse:
        print("\n  [TOOL-USE & FUNCTION CALLING BENCHMARK]")
        tool_header = (
            f"{'Model / Checkpoint':<38} | {'Backend':<7} | {'Tool Acc (%)':<13} | {'Syntax (%)':<11} | {'Pass Rate (%)'}"
        )
        print(tool_header)
        print("-" * 115)
        for name, rep in multi_reports.items():
            b = rep.get("backend", "unknown")
            tr = rep.get("tooluse", {})
            t_acc = f"{tr.get('tool_accuracy_pct', 0.0):.1f}%"
            s_acc = f"{tr.get('syntax_validity_pct', 0.0):.1f}%"
            p_acc = f"{tr.get('pass_rate_pct', 0.0):.1f}%"
            disp_name = name if len(name) <= 38 else "..." + name[-35:]
            print(f"{disp_name:<38} | {b:<7} | {t_acc:<13} | {s_acc:<11} | {p_acc}")

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
    benchmark_type: str = "all",
    language: str = "auto",
    prompt_mode: str = "base",
    **kwargs
) -> Dict[str, Any]:
    """
    Master programmatic entrypoint for Télos evaluation engine.
    Supports evaluating a single checkpoint or multiple checkpoints side-by-side across
    benchmark types ('all', 'code', 'linguistic', 'tooluse') and languages ('auto', 'english', 'python').
    """
    b_type = str(kwargs.get("type", benchmark_type)).lower()

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
            benchmark_type=b_type,
            language=language,
            prompt_mode=prompt_mode,
            **kwargs
        )
        if not output_path:
            out_file = PROJECT_ROOT / "logs" / f"eval_report_{b_type}_{mode}_{int(time.time())}.json"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, "w") as f:
                json.dump(rep, f, indent=2)
            print(f"✓ Saved evaluation report to {out_file}\n")
        return rep

    # Multi-model evaluation workflow
    print("\n" + "=" * 80)
    print(f"  TÉLOS MULTI-MODEL BENCHMARK: EVALUATING {len(checkpoints)} MODELS (TYPE: {b_type.upper()}, MODE: {mode.upper()})")
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
            benchmark_type=b_type,
            language=language,
            prompt_mode=prompt_mode,
            **kwargs
        )
        multi_reports[name] = single_rep

    # Print comparative scorecard
    print_multimodel_scorecard(multi_reports, mode=mode)

    # Save consolidated report
    out_file = Path(output_path) if output_path else (PROJECT_ROOT / "logs" / f"eval_report_multimodel_{b_type}_{mode}_{int(time.time())}.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(multi_reports, f, indent=2)
    print(f"✓ Saved consolidated multi-model evaluation report to {out_file}\n")

    return multi_reports


def main():
    parser = argparse.ArgumentParser(description="Télos Next-Generation Universal Evaluation Suite")
    parser.add_argument(
        "--checkpoint", "--checkpoints", "--model", "--models",
        type=str,
        nargs="+",
        dest="checkpoints",
        required=True,
        help="One or more paths to checkpoint file(s) (.safetensors, .pt), HF models, or API model names (e.g. 'afm-3-core', 'ternary-bonsai-27b', 'gemma-4-12b', 'gemini-4-26b-a4b')"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "telos_native", "openai_api", "gemini_api", "huggingface", "mlx_lm", "swift", "swift_afm"],
        help="Model runtime backend. 'auto' selects based on model string."
    )
    parser.add_argument("--api-base", type=str, default=None, help="Base URL for OpenAI-compatible REST server (e.g. http://localhost:8000/v1)")
    parser.add_argument("--api-key", type=str, default=None, help="API key for OpenAI or Gemini")
    parser.add_argument("--concurrency", type=int, default=8, help="Number of concurrent worker threads for API models")
    parser.add_argument("--device", type=str, default="auto", help="Torch device ('auto', 'cpu', 'cuda', 'mps')")
    parser.add_argument("--torch-dtype", type=str, default="bfloat16", help="Precision for HuggingFace models ('bfloat16', 'float16', 'float32')")
    parser.add_argument("--quantization", type=str, default=None, choices=["4bit", "8bit", None], help="Quantization for HF models")
    parser.add_argument(
        "--type", "--benchmark-type",
        type=str,
        default="all",
        choices=["all", "code", "math", "science", "cyber", "tooluse", "linguistic"],
        dest="benchmark_type",
        help="Type of benchmark to execute ('all', 'code', 'math', 'science', 'cyber', 'tooluse', 'linguistic'). Default is 'all'."
    )
    parser.add_argument(
        "--language", "--lang",
        type=str,
        default="auto",
        choices=["auto", "english", "python"],
        dest="language",
        help="Target benchmark language ('auto', 'english', 'python'). Default is 'auto'."
    )
    parser.add_argument("--mode", type=str, default="probes", choices=["probes", "functional", "anticheat", "perplexity", "full", "sample"], help="Evaluation mode")
    parser.add_argument(
        "--suite",
        type=str,
        default="private_unseen",
        choices=[
            "private_unseen", "private_unseen_base", "private_unseen_hint",
            "humaneval", "mbpp", "public_standard",
            "humaneval_cs", "humaneval_java", "humaneval_js", "humaneval_ts", "humaneval_rust",
            "react", "gsm8k", "arc", "gpqa", "gpqa_diamond", "mmlu", "mmlu_science",
            "competition_math", "cyber", "tooluse"
        ],
        help="Benchmark suite track. Default is 'private_unseen'."
    )
    parser.add_argument(
        "--prompt-mode",
        type=str,
        default="base",
        choices=["base", "hint"],
        dest="prompt_mode",
        help="Prompt variant to use ('base' for standard problem spec, 'hint' for algorithmic guidance). Default is 'base'."
    )
    parser.add_argument(
        "--hint",
        action="store_const",
        dest="prompt_mode",
        const="hint",
        help="Use hint prompt mode (algorithmic guidance in docstrings)."
    )
    parser.add_argument(
        "--nohint",
        action="store_const",
        dest="prompt_mode",
        const="base",
        help="Use base prompt mode (standard problem specification without hints)."
    )
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
        output_path=args.output,
        benchmark_type=args.benchmark_type,
        language=args.language,
        prompt_mode=args.prompt_mode,
        backend=args.backend,
        api_base=args.api_base,
        api_key=args.api_key,
        concurrency=args.concurrency,
        device=args.device,
        torch_dtype=args.torch_dtype,
        quantization=args.quantization,
    )


if __name__ == "__main__":
    main()



