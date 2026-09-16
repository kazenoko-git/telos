"""
Compute Validation Loss Curves across Checkpoints:
1. 100M AR (5B tokens)
2. 100M COROSred Unified (old alpha_min=0.50, 5B tokens)
3. 100M COROSred Unified (new alpha_min=0.75, 5B tokens)
4. 50M COROSred Unified (2.5B tokens, final)
"""

import os
import sys
import json
import time
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from telos.eval.runner import load_model_from_checkpoint

# 1. Load standardized validation slice (100 sequences x 512 tokens = 51.2k tokens)
file_path = PROJECT_ROOT / "data" / "python_corpus_5b.bin"
num_seqs = 100
seq_len = 512
offset = 500_000_000 * 2  # 500M tokens in, far from start
data = np.memmap(file_path, dtype=np.uint16, mode="r", offset=offset, shape=(num_seqs, seq_len))
val_tokens = torch.tensor(data.astype(np.int64))

# Prepare fixed 20% mask for infilling evaluation (reproducible seed)
torch.manual_seed(42)
mask_prob = 0.20
mask_rand = torch.rand(num_seqs, seq_len)
# Don't mask position 0 (BOS)
mask_rand[:, 0] = 1.0
infill_mask = mask_rand < mask_prob
masked_tokens = val_tokens.clone()
masked_tokens[infill_mask] = 1  # 1 = [MASK] token id

steps = [3500, 7000, 10500, 14000, 17500, 21000, 24500, "final"]

model_configs = {
    "100M_AR": {
        "dir": PROJECT_ROOT / "checkpoints/ar/100m_5b",
        "steps": steps,
        "is_causal": True
    },
    "100M_COROSred_alpha050": {
        "dir": PROJECT_ROOT / "checkpoints/corosred/unified/100m_5b",
        "steps": steps,
        "is_causal": True
    },
    "100M_COROSred_alpha075": {
        "dir": PROJECT_ROOT / "checkpoints/corosred/unified/100m_5b_alpha075",
        "steps": steps,
        "is_causal": True
    }
}

device = torch.device("cpu")  # CPU is rock-solid and fast for 100 sequences
val_tokens = val_tokens.to(device)
masked_tokens = masked_tokens.to(device)
infill_mask = infill_mask.to(device)

results = {}

for model_name, info in model_configs.items():
    results[model_name] = []
    print(f"\nEvaluating Loss Curve for: {model_name}")
    print("-" * 70)
    print(f"{'Step':<10} | {'Causal CE Loss':<16} | {'Causal PPL':<12} | {'Infill Loss':<14} | {'Elapsed':<8}")
    print("-" * 70)

    for step in info["steps"]:
        fname = f"checkpoint_step_{step}.pt" if step != "final" else "checkpoint_final.pt"
        ckpt_path = info["dir"] / fname
        if not ckpt_path.exists():
            print(f"Skipping {fname}, not found.")
            continue

        t0 = time.time()
        model, backend, _ = load_model_from_checkpoint(ckpt_path)
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            # 1. Causal Cross-Entropy Loss
            logits = model(val_tokens)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = val_tokens[:, 1:].contiguous()
            causal_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            ).item()
            causal_ppl = float(np.exp(min(causal_loss, 20.0)))

            # 2. Infill Loss (loss on the masked tokens)
            infill_logits = model(masked_tokens)
            target_labels = val_tokens[infill_mask]
            pred_logits = infill_logits[infill_mask]
            infill_loss = F.cross_entropy(pred_logits, target_labels).item()

        elapsed = time.time() - t0
        step_label = str(step) if step != "final" else "25432 (final)"
        print(f"{step_label:<10} | {causal_loss:<16.4f} | {causal_ppl:<12.2f} | {infill_loss:<14.4f} | {elapsed:<6.1f}s")

        results[model_name].append({
            "step": step_label,
            "step_num": step if step != "final" else 25432,
            "causal_loss": round(causal_loss, 4),
            "causal_ppl": round(causal_ppl, 2),
            "infill_loss": round(infill_loss, 4),
        })

# Also evaluate 50M COROSred final for reference
ckpt_50m = PROJECT_ROOT / "checkpoints/corosred/unified/50m_2.5b/checkpoint_final.pt"
if ckpt_50m.exists():
    print(f"\nEvaluating Reference: 50M COROSred (2.5B final)")
    model_50m, _, _ = load_model_from_checkpoint(ckpt_50m)
    model_50m = model_50m.to(device)
    model_50m.eval()
    with torch.no_grad():
        logits_50m = model_50m(val_tokens)
        shift_logits = logits_50m[:, :-1, :].contiguous()
        shift_labels = val_tokens[:, 1:].contiguous()
        causal_loss_50m = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).item()
        infill_logits = model_50m(masked_tokens)
        infill_loss_50m = F.cross_entropy(infill_logits[infill_mask], val_tokens[infill_mask]).item()
    results["50M_COROSred_ref"] = {
        "step": "12716 (final)",
        "causal_loss": round(causal_loss_50m, 4),
        "causal_ppl": round(float(np.exp(causal_loss_50m)), 2),
        "infill_loss": round(infill_loss_50m, 4),
    }
    print(f"50M Final   | Causal CE: {causal_loss_50m:.4f} (PPL: {np.exp(causal_loss_50m):.2f}) | Infill Loss: {infill_loss_50m:.4f}")

out_path = PROJECT_ROOT / "logs" / "loss_curves_comparison.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Saved loss curves to {out_path}")
