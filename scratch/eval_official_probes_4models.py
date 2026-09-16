"""
Run the official evaluate_probes from telos.eval.runner on the exact 4 models.
"""

import json
from pathlib import Path
from telos.eval.runner import load_model_from_checkpoint, evaluate_probes
from telos.data.tokenizer import load_tokenizer

MODELS = {
    "15M COROSred (TPU)": "checkpoints/corosred/unified/15m/checkpoint_final.pt",
    "50M COROSred (CUDA, Lightning)": "checkpoints/corosred/50m_lightning/checkpoint_final.pt",
    "75M COROSred (TPU)": "checkpoints/corosred/unified/75m_python/checkpoint_final.pt",
    "100M COROSred (TPU, alpha=0.5)": "checkpoints/corosred/unified/100m_5b/checkpoint_final.pt",
}

tokenizer = load_tokenizer("configs/tokenizer_mac.json")

scorecard = {}

for name, ckpt_path in MODELS.items():
    print("=" * 80)
    print(f"RUNNING OFFICIAL PROBES: {name}")
    print("=" * 80)
    model, backend, vocab_size = load_model_from_checkpoint(ckpt_path)
    res = evaluate_probes(model, tokenizer, backend, paradigm="corosred", num_probes=100, probe_type="both")
    
    cau = res["causal"]["overall"]
    inf = res["infill"]["overall"]
    
    scorecard[name] = {
        "causal_top1": cau["top1_acc_pct"],
        "causal_ce": cau["mean_ce"],
        "causal_rank": cau["mean_rank"],
        "infill_top1": inf["top1_acc_pct"],
        "infill_ce": inf["mean_ce"],
        "infill_rank": inf["mean_rank"],
        "causal_categories": res["causal"]["categories"],
        "infill_categories": res["infill"]["categories"]
    }

print("\n" + "=" * 90)
print(f"{'Model Name':<32} | {'Causal Top1 (75)':<16} | {'Causal CE':<10} | {'Infill Top1 (100)':<17} | {'Infill CE'}")
print("-" * 90)
for name, s in scorecard.items():
    print(f"{name:<32} | {s['causal_top1']:>13.1f}% | {s['causal_ce']:>9.2f} | {s['infill_top1']:>15.1f}% | {s['infill_ce']:>9.2f}")
print("=" * 90)

with open("logs/official_probes_exact_4models.json", "w") as f:
    json.dump(scorecard, f, indent=2)
