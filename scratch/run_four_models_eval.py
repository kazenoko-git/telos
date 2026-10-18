"""
Comprehensive Evaluation Suite for the 4 Newly Downloaded Models:
1. 100M COROSred TPU Run 2 (5B tokens)
2. 100M COROSred Schedstep 1 (5B tokens)
3. 100M AR TPU (5B tokens)
4. 50M AR TPU (2.5B tokens)

Evaluates:
- 1,000 Contextual Probes (Infill & Causal Top-1/Top-5/CE across 8 categories)
- Anticheat Suite (Span-infill memorization vs generalization across spans 1, 2, 4, 8)
- Tool-Use (30 tasks)
- HumanEval Python (164 tasks)
- Private Unseen Python (512 tasks)

Saves results incrementally to logs/eval_report_four_downloaded_models.json
"""

import os
import sys
import json
import time
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from telos.eval.runner import (
    load_model_from_checkpoint,
    _evaluate_single_probe_type,
    evaluate_anticheat,
    evaluate_functional,
    evaluate_tooluse,
    load_contextual_probes
)
from telos.data.tokenizer import load_tokenizer

MODELS = [
    ("100M COROSred TPU 2", "checkpoints/corosred/unified/100m_tpu_2/checkpoint_final.pt", "corosred"),
    ("100M COROSred Schedstep 1", "checkpoints/corosred/unified/100m_schedstep1/checkpoint_final.pt", "corosred"),
    ("100M AR TPU (5B)", "checkpoints/ar/100m_5b_tpu/checkpoint_final.pt", "ar"),
    ("50M AR TPU (2.5B)", "checkpoints/ar/50m_2.5b_tpu/checkpoint_final.pt", "ar"),
]

OUTPUT_PATH = PROJECT_ROOT / "logs" / "eval_report_four_downloaded_models.json"

def save_report(report_data):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"\n[REPORT SAVED] Updated {OUTPUT_PATH}", flush=True)

def main():
    print("=" * 90, flush=True)
    print("  TÉLOS BENCHMARK EVALUATION — 4 NEWLY DOWNLOADED MODELS", flush=True)
    print("=" * 90, flush=True)

    gpu_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"GPU device: {gpu_device} | CPU for probe forward passes", flush=True)
    tokenizer = load_tokenizer()

    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, "r") as f:
            all_reports = json.load(f)
    else:
        all_reports = {}

    # Load 1,000 contextual probes dataset
    probes_1000_file = PROJECT_ROOT / "evals" / "benchmarks" / "contextual_probes_1000.json"
    with open(probes_1000_file, "r") as f:
        probes_1000 = json.load(f)

    for name, ckpt_path, paradigm in MODELS:
        print("\n" + "#" * 90, flush=True)
        print(f"  MODEL: {name.upper()}", flush=True)
        print(f"  Checkpoint: {ckpt_path} | Paradigm: {paradigm}", flush=True)
        print("#" * 90, flush=True)

        if name not in all_reports:
            all_reports[name] = {"checkpoint": ckpt_path, "paradigm": paradigm}

        model, backend, _ = load_model_from_checkpoint(ckpt_path)

        # -------------------------------------------------------------
        # 1. 1,000 CONTEXTUAL PROBES (CPU for stable tensor handling)
        # -------------------------------------------------------------
        if "probes_1000" not in all_reports[name]:
            print("\n>>> [1/5] Evaluating 1,000 Contextual Probes...", flush=True)
            model = model.to("cpu").eval()
            t0 = time.time()
            causal_res = _evaluate_single_probe_type(
                model=model, tokenizer=tokenizer, backend=backend,
                probes_list=probes_1000, probe_type="causal"
            )
            c_top1 = causal_res["overall"]["top1_acc_pct"]
            c_top5 = causal_res["overall"]["top5_acc_pct"]
            c_ce = causal_res["overall"]["mean_ce"]
            print(f"  Causal: Top-1={c_top1:.1f}% | Top-5={c_top5:.1f}% | Mean CE={c_ce:.2f}", flush=True)

            if paradigm == "ar":
                infill_res = {
                    "status": "not_applicable",
                    "overall": {"top1_acc_pct": None, "top5_acc_pct": None, "mean_ce": None}
                }
                print("  Infill: N/A (AR pure causal)", flush=True)
            else:
                infill_res = _evaluate_single_probe_type(
                    model=model, tokenizer=tokenizer, backend=backend,
                    probes_list=probes_1000, probe_type="infill"
                )
                i_top1 = infill_res["overall"]["top1_acc_pct"]
                i_top5 = infill_res["overall"]["top5_acc_pct"]
                i_ce = infill_res["overall"]["mean_ce"]
                print(f"  Infill: Top-1={i_top1:.1f}% | Top-5={i_top5:.1f}% | Mean CE={i_ce:.2f}", flush=True)

            all_reports[name]["probes_1000"] = {
                "causal": causal_res["overall"],
                "infill": infill_res["overall"],
                "elapsed_sec": round(time.time() - t0, 2)
            }
            save_report(all_reports)
        else:
            print(">>> [1/5] Contextual Probes already evaluated, skipping.", flush=True)

        # -------------------------------------------------------------
        # 2. ANTICHEAT SUITE (CPU for span infill)
        # -------------------------------------------------------------
        if "anticheat" not in all_reports[name]:
            print("\n>>> [2/5] Evaluating Anticheat Suite...", flush=True)
            model = model.to("cpu").eval()
            t0 = time.time()
            ac_res = evaluate_anticheat(
                model=model, tokenizer=tokenizer, backend=backend,
                paradigm=paradigm, num_probes=100, span_lengths=[1, 2, 4, 8]
            )
            all_reports[name]["anticheat"] = {
                "summary": ac_res.get("span_breakdown", {}),
                "is_suspect_cheater": ac_res.get("is_suspect_cheater", False),
                "elapsed_sec": round(time.time() - t0, 2)
            }
            save_report(all_reports)
        else:
            print(">>> [2/5] Anticheat already evaluated, skipping.", flush=True)

        # Move model to GPU/MPS for autoregressive functional generation
        model = model.to(gpu_device).eval()

        # -------------------------------------------------------------
        # 3. TOOL-USE BENCHMARK (30 TASKS)
        # -------------------------------------------------------------
        if "tooluse" not in all_reports[name]:
            print("\n>>> [3/5] Evaluating Tool-Use Suite (30 Tasks)...", flush=True)
            t0 = time.time()
            tu_res = evaluate_tooluse(
                model=model, tokenizer=tokenizer, backend=backend,
                max_tasks=30, max_new_tokens=64
            )
            all_reports[name]["tooluse"] = {
                "pass_at_1_pct": tu_res.get("pass_at_1_pct"),
                "ast_validity_pct": tu_res.get("ast_validity_pct"),
                "total_tasks": tu_res.get("total_tasks"),
                "elapsed_sec": round(time.time() - t0, 2)
            }
            save_report(all_reports)
        else:
            print(">>> [3/5] Tool-Use already evaluated, skipping.", flush=True)

        # -------------------------------------------------------------
        # 4. HUMANEVAL PYTHON (164 TASKS)
        # -------------------------------------------------------------
        if "humaneval" not in all_reports[name]:
            print("\n>>> [4/5] Evaluating HumanEval Python (164 Tasks)...", flush=True)
            t0 = time.time()
            he_res = evaluate_functional(
                model=model, tokenizer=tokenizer, backend=backend,
                suite="humaneval", max_tasks=164, timeout_seconds=1.5
            )
            all_reports[name]["humaneval"] = {
                "pass_at_1_pct": he_res.get("pass_at_1_pct"),
                "pass_at_1_95ci": he_res.get("pass_at_1_95ci"),
                "ast_validity_pct": he_res.get("ast_validity_pct"),
                "syntax_error_rate_pct": he_res.get("syntax_error_rate_pct"),
                "execution_outcomes": he_res.get("execution_outcomes"),
                "total_tasks": he_res.get("total_tasks"),
                "elapsed_sec": round(time.time() - t0, 2)
            }
            save_report(all_reports)
        else:
            print(">>> [4/5] HumanEval already evaluated, skipping.", flush=True)

        # -------------------------------------------------------------
        # 5. PRIVATE UNSEEN PYTHON (512 TASKS)
        # -------------------------------------------------------------
        if "private_unseen" not in all_reports[name]:
            print("\n>>> [5/5] Evaluating Private Unseen Python (512 Tasks)...", flush=True)
            t0 = time.time()
            pu_res = evaluate_functional(
                model=model, tokenizer=tokenizer, backend=backend,
                suite="private_unseen", max_tasks=512, timeout_seconds=1.5
            )
            all_reports[name]["private_unseen"] = {
                "pass_at_1_pct": pu_res.get("pass_at_1_pct"),
                "pass_at_1_95ci": pu_res.get("pass_at_1_95ci"),
                "ast_validity_pct": pu_res.get("ast_validity_pct"),
                "syntax_error_rate_pct": pu_res.get("syntax_error_rate_pct"),
                "execution_outcomes": pu_res.get("execution_outcomes"),
                "category_breakdown": {
                    k: {"pass@1": v.get("pass@1"), "ast": v.get("ast_validity"), "total": v.get("total")}
                    for k, v in pu_res.get("category_breakdown", {}).items()
                },
                "total_tasks": pu_res.get("total_tasks"),
                "elapsed_sec": round(time.time() - t0, 2)
            }
            save_report(all_reports)
        else:
            print(">>> [5/5] Private Unseen already evaluated, skipping.", flush=True)

        # Clean up GPU memory
        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    print("\n" + "=" * 90, flush=True)
    print("  ALL EVALUATIONS COMPLETE FOR THE 4 DOWNLOADED MODELS!", flush=True)
    print(f"  Results saved to: {OUTPUT_PATH}", flush=True)
    print("=" * 90, flush=True)

if __name__ == "__main__":
    main()
