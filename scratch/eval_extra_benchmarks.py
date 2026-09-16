"""
Runner for 1,000 contextual probes and Memorization vs Generalization perturbation controls across all 7 models.
"""

import sys
import json
import time
import math
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from telos.eval.runner import load_model_from_checkpoint, _evaluate_single_probe_type, bootstrap_confidence_interval
from telos.data.tokenizer import load_tokenizer

CHECKPOINTS = [
    ("corosred/100m_5b", "checkpoints/corosred/unified/100m_5b/checkpoint_final.pt"),
    ("ar/100m", "checkpoints/ar/100m/checkpoint_final.pt"),
    ("ar/100m_5b", "checkpoints/ar/100m_5b/checkpoint_final.pt"),
    ("corosred/50m_2.5b", "checkpoints/corosred/unified/50m_2.5b/checkpoint_final.pt"),
    ("ar/50m_2.5b", "checkpoints/ar/50m_2.5b/checkpoint_final.pt"),
    ("corosred/15m_retrained", "checkpoints/corosred/unified/15m_retrained/checkpoint_final.pt"),
    ("ar/15m", "checkpoints/ar/checkpoint_final.pt"),
]

# Perturbation control pairs: Canonical (ubiquitous unigram) vs Perturbed (novel identifier)
PERTURBATION_PROBES = [
    # Scope binding 1: simple variable reference
    {
        "type": "scope_reference",
        "canonical": {"prompt": "x = 10\nprint(", "target": "x", "target_bpe": "x"},
        "perturbed": {"prompt": "alpha_var = 10\nprint(", "target": "alpha_var", "target_bpe": "alpha"}
    },
    # Scope binding 2: loop variable reference
    {
        "type": "loop_reference",
        "canonical": {"prompt": "total = 0\nfor item in items:\n    total +=", "target": "item", "target_bpe": "Ġitem"},
        "perturbed": {"prompt": "total = 0\nfor element_val in items:\n    total +=", "target": "element_val", "target_bpe": "Ġelement"}
    },
    # Scope binding 3: attribute getter resolution
    {
        "type": "getter_resolution",
        "canonical": {"prompt": "def get_name(self):\n    return self._", "target": "name", "target_bpe": "name"},
        "perturbed": {"prompt": "def get_payload(self):\n    return self._", "target": "payload", "target_bpe": "payload"}
    },
    # Scope binding 4: constructor parameter assignment
    {
        "type": "constructor_param",
        "canonical": {"prompt": "def __init__(self, config):\n    self.config =", "target": "config", "target_bpe": "Ġconfig"},
        "perturbed": {"prompt": "def __init__(self, hyperparams):\n    self.hyperparams =", "target": "hyperparams", "target_bpe": "Ġhyper"}
    },
    # Scope binding 5: dict unpacked key
    {
        "type": "dict_key",
        "canonical": {"prompt": "key = 'user_id'\nval = cache[", "target": "key", "target_bpe": "key"},
        "perturbed": {"prompt": "lookup_tok = 'user_id'\nval = cache[", "target": "lookup_tok", "target_bpe": "lookup"}
    },
    # Scope binding 6: boundary span subtraction
    {
        "type": "boundary_span",
        "canonical": {"prompt": "start, end = range_vals\nlength = end -", "target": "start", "target_bpe": "Ġstart"},
        "perturbed": {"prompt": "lo_bound, hi_bound = range_vals\nlength = hi_bound -", "target": "lo_bound", "target_bpe": "Ġlo"}
    }
]


def run_1000_probes():
    tok = load_tokenizer("configs/tokenizer_mac.json")
    with open("evals/benchmarks/contextual_probes_1000.json") as f:
        probes_1000 = json.load(f)

    print("\n" + "=" * 115)
    print("           TÉLOS 1,000 CONTEXTUAL PROBES BENCHMARK SCORECARD")
    print("=" * 115)
    print(f"{'Model / Checkpoint':<38} | {'Infill Top1':<11} | {'Infill CE':<9} | {'Causal Top1':<11} | {'Causal CE':<9}")
    print("-" * 115)

    results_1000 = {}
    for name, cp in CHECKPOINTS:
        model, backend, _ = load_model_from_checkpoint(cp)
        p = getattr(model, "paradigm", "")
        if not p:
            p = "ar" if "ar" in name else "corosred"

        # Causal evaluation
        cau = _evaluate_single_probe_type(model, tok, backend, probes_1000, probe_type="causal")
        cau_t1 = f"{cau['overall']['top1_acc_pct']:.1f}%"
        cau_ce = f"{cau['overall']['mean_ce']:.2f}"

        # Infill evaluation
        if p == "ar":
            inf_t1 = "N/A (AR)"
            inf_ce = "N/A"
            inf = {"status": "not_applicable"}
        else:
            inf = _evaluate_single_probe_type(model, tok, backend, probes_1000, probe_type="infill")
            inf_t1 = f"{inf['overall']['top1_acc_pct']:.1f}%"
            inf_ce = f"{inf['overall']['mean_ce']:.2f}"

        print(f"{name:<38} | {inf_t1:<11} | {inf_ce:<9} | {cau_t1:<11} | {cau_ce:<9}")
        results_1000[name] = {"causal": cau, "infill": inf}

    out_file = PROJECT_ROOT / "logs" / f"eval_report_multimodel_probes_1000_{int(time.time())}.json"
    with open(out_file, "w") as f:
        json.dump(results_1000, f, indent=2)
    print(f"\n✓ Saved 1000 probes report to {out_file}\n")


def run_perturbation_controls():
    tok = load_tokenizer("configs/tokenizer_mac.json")

    print("\n" + "=" * 115)
    print("      MEMORIZATION VS GENERALIZATION CONTROLS: NOVEL IDENTIFIER PERTURBATION PROBES")
    print("=" * 115)
    print(f"{'Model / Checkpoint':<38} | {'Canonical Acc':<14} | {'Perturbed Acc':<14} | {'Gen Ratio (%)':<14} | {'Diagnosis'}")
    print("-" * 115)

    results_pert = {}
    for name, cp in CHECKPOINTS:
        model, backend, _ = load_model_from_checkpoint(cp)
        
        canon_correct = 0
        pert_correct = 0
        total = len(PERTURBATION_PROBES)

        probe_details = []
        for pair in PERTURBATION_PROBES:
            c_probe = pair["canonical"]
            p_probe = pair["perturbed"]

            # Evaluate Canonical
            c_p_ids = tok.encode(c_probe["prompt"]).ids
            c_t_id = tok.token_to_id(c_probe["target_bpe"]) or tok.encode(c_probe["target"]).ids[0]
            with torch.no_grad():
                c_out = model(torch.tensor([c_p_ids], dtype=torch.long), mask_override=True)
            c_pred_id = int(torch.argmax(c_out[0, -1]).item())
            c_hit = (c_pred_id == c_t_id)
            if c_hit:
                canon_correct += 1

            # Evaluate Perturbed
            p_p_ids = tok.encode(p_probe["prompt"]).ids
            p_t_id = tok.token_to_id(p_probe["target_bpe"]) or tok.encode(p_probe["target"]).ids[0]
            with torch.no_grad():
                p_out = model(torch.tensor([p_p_ids], dtype=torch.long), mask_override=True)
            p_pred_id = int(torch.argmax(p_out[0, -1]).item())
            p_hit = (p_pred_id == p_t_id)
            if p_hit:
                pert_correct += 1

            c_pred_str = tok.decode([c_pred_id])
            p_pred_str = tok.decode([p_pred_id])
            probe_details.append({
                "type": pair["type"],
                "canonical_prompt": c_probe["prompt"],
                "canonical_target": c_probe["target"],
                "canonical_pred": c_pred_str,
                "canonical_hit": c_hit,
                "perturbed_prompt": p_probe["prompt"],
                "perturbed_target": p_probe["target"],
                "perturbed_pred": p_pred_str,
                "perturbed_hit": p_hit,
            })

        c_acc = (canon_correct / total) * 100.0
        p_acc = (pert_correct / total) * 100.0
        gen_ratio = (p_acc / c_acc * 100.0) if c_acc > 0 else 0.0

        if gen_ratio >= 60.0:
            diag = "ROBUST GENERALIZATION"
        elif c_acc >= 40.0 and gen_ratio < 20.0:
            diag = "UNIGRAM / IDIOM MEMORIZATION"
        elif c_acc < 20.0:
            diag = "UNDERFITTED / UNCONVERGED"
        else:
            diag = "MODERATE MEMORIZATION"

        print(f"{name:<38} | {c_acc:>12.1f}% | {p_acc:>12.1f}% | {gen_ratio:>12.1f}% | {diag}")
        results_pert[name] = {
            "canonical_acc_pct": c_acc,
            "perturbed_acc_pct": p_acc,
            "generalization_ratio_pct": gen_ratio,
            "diagnosis": diag,
            "details": probe_details
        }

    out_file = PROJECT_ROOT / "logs" / f"eval_report_multimodel_perturbations_{int(time.time())}.json"
    with open(out_file, "w") as f:
        json.dump(results_pert, f, indent=2)
    print(f"\n✓ Saved perturbation controls report to {out_file}\n")


if __name__ == "__main__":
    print("Starting comprehensive benchmark run for 1,000 probes and perturbation controls...")
    run_1000_probes()
    run_perturbation_controls()
