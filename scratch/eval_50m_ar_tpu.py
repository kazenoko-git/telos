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
    evaluate_functional,
)
from telos.data.tokenizer import load_tokenizer

def main():
    print("=" * 80)
    print("  EVALUATING DOWNLOADED 50M AR (TPU 2.5B) MODEL")
    print("=" * 80)

    device = torch.device("cpu")
    ckpt_path = "checkpoints/ar/50m_2.5b_tpu/checkpoint_final.pt"
    print(f"Loading checkpoint: {ckpt_path}")
    model, backend, _ = load_model_from_checkpoint(ckpt_path)
    model = model.to(device).eval()
    tokenizer = load_tokenizer()

    # 1. 1,000 Contextual Probes
    probes_file = PROJECT_ROOT / "evals" / "benchmarks" / "contextual_probes_1000.json"
    with open(probes_file, "r") as f:
        probes_1000 = json.load(f)

    print("\n>>> [1/2] Evaluating 1,000 Causal Contextual Probes on CPU...")
    t0 = time.time()
    causal_res = _evaluate_single_probe_type(
        model=model, tokenizer=tokenizer, backend=backend,
        probes_list=probes_1000, probe_type="causal"
    )
    t_probes = round(time.time() - t0, 2)
    c_top1 = causal_res["overall"]["top1_acc_pct"]
    c_top5 = causal_res["overall"]["top5_acc_pct"]
    c_ce = causal_res["overall"]["mean_ce"]
    print(f"  [1k Probes] Causal Top-1: {c_top1:.1f}% | Top-5: {c_top5:.1f}% | Mean CE: {c_ce:.2f} ({t_probes}s)")

    # 2. HumanEval (164 Tasks)
    print("\n>>> [2/2] Evaluating HumanEval Python (164 Tasks) on CPU...")
    t0 = time.time()
    he_res = evaluate_functional(
        model=model, tokenizer=tokenizer, backend=backend,
        suite="humaneval", max_tasks=164, timeout_seconds=1.5
    )
    t_he = round(time.time() - t0, 2)

    results = {
        "model": "50M AR TPU (2.5B)",
        "checkpoint": ckpt_path,
        "probes_1000": {
            "causal": causal_res["overall"],
            "elapsed_sec": t_probes
        },
        "humaneval": {
            "pass_at_1_pct": he_res.get("pass_at_1_pct"),
            "ast_validity_pct": he_res.get("ast_validity_pct"),
            "syntax_error_rate_pct": he_res.get("syntax_error_rate_pct"),
            "execution_outcomes": he_res.get("execution_outcomes"),
            "total_tasks": he_res.get("total_tasks"),
            "elapsed_sec": t_he
        }
    }

    out_file = PROJECT_ROOT / "logs" / "eval_50m_ar_tpu_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[DONE] Saved results to {out_file}")

if __name__ == "__main__":
    main()
