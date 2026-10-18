"""
Official Head-to-Head Evaluation of 50M Lightning vs New 50M Run (alpha0.6, causal0.7)
Using:
1. 1,000 Contextual Probes (evals/benchmarks/contextual_probes_1000.json)
2. 512 Functional Unit Test Tasks (evals/benchmarks/private_unseen_suite.json)
"""

import sys
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from telos.eval.runner import (
    load_model_from_checkpoint,
    _evaluate_single_probe_type,
    evaluate_functional,
    print_multimodel_scorecard
)
from telos.data.tokenizer import load_tokenizer

MODELS = [
    ("corosred/50m_lightning", "checkpoints/corosred/50m_lightning/checkpoint_final.pt"),
    ("corosred/50m_alpha06_causal07", "checkpoints/corosred/50m_alpha06_causal07/checkpoint_final.pt")
]

def run_evaluation():
    tok = load_tokenizer()
    
    # -------------------------------------------------------------
    # 1. 1,000 CONTEXTUAL PROBES
    # -------------------------------------------------------------
    probes_file = PROJECT_ROOT / "evals" / "benchmarks" / "contextual_probes_1000.json"
    with open(probes_file, "r") as f:
        probes_1000 = json.load(f)
        
    print("\n" + "=" * 90)
    print(f"  TÉLOS 1,000 CONTEXTUAL PROBES BENCHMARK ({len(probes_1000)} PROBES ACROSS 8 CATEGORIES)")
    print("=" * 90)
    
    probes_results = {}
    for name, cp in MODELS:
        print(f"\nEvaluating Probes for {name} ({cp})...")
        model, backend, _ = load_model_from_checkpoint(cp)
        
        # Infill evaluation
        inf = _evaluate_single_probe_type(model, tok, backend, probes_1000, probe_type="infill")
        # Causal evaluation
        cau = _evaluate_single_probe_type(model, tok, backend, probes_1000, probe_type="causal")
        
        probes_results[name] = {"infill": inf, "causal": cau}
        print(f"  Infill  Top-1: {inf['overall']['top1_acc_pct']:.1f}% | Top-5: {inf['overall']['top5_acc_pct']:.1f}% | Mean Rank: {inf['overall']['mean_rank']:.1f} | Mean CE: {inf['overall']['mean_ce']:.2f}")
        print(f"  Causal  Top-1: {cau['overall']['top1_acc_pct']:.1f}% | Top-5: {cau['overall']['top5_acc_pct']:.1f}% | Mean Rank: {cau['overall']['mean_rank']:.1f} | Mean CE: {cau['overall']['mean_ce']:.2f}")
        
    # Save probes results
    probes_out = PROJECT_ROOT / "logs" / f"eval_report_50m_head_to_head_1000_probes_{int(time.time())}.json"
    with open(probes_out, "w") as f:
        json.dump(probes_results, f, indent=2)
    print(f"\n✓ Saved 1,000 contextual probes report to {probes_out}")

    # -------------------------------------------------------------
    # 2. 512 FUNCTIONAL TASKS & PASS@1
    # -------------------------------------------------------------
    print("\n" + "=" * 90)
    print("  TÉLOS 512 FUNCTIONAL UNIT TEST BENCHMARK (evals/benchmarks/private_unseen_suite.json)")
    print("=" * 90)
    
    functional_results = {}
    for name, cp in MODELS:
        print(f"\nExecuting 512 Functional Sandbox Benchmark for {name}...")
        model, backend, _ = load_model_from_checkpoint(cp)
        func_rep = evaluate_functional(
            model=model,
            tokenizer=tok,
            backend=backend,
            suite="private_unseen",
            timeout_seconds=3.0
        )
        functional_results[name] = func_rep
        
    # Save functional results
    func_out = PROJECT_ROOT / "logs" / f"eval_report_50m_head_to_head_512_functional_{int(time.time())}.json"
    with open(func_out, "w") as f:
        json.dump(functional_results, f, indent=2)
    print(f"\n✓ Saved 512 functional tasks report to {func_out}")

if __name__ == "__main__":
    run_evaluation()
