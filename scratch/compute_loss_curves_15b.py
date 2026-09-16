"""
Validation Loss Curves on Truly Held-Out 15B Python Slice:
Evaluates:
  - 100M AR (5B tokens)
  - 100M COROSred Unified (alpha_min=0.50, 5B tokens)
  - 100M COROSred Unified (alpha_min=0.75, 5B tokens)
  - 50M COROSred Unified (2.5B tokens, final reference)
Dataset: data/python_corpus_15b.bin (Offset: 6,000,000,000 tokens — completely held out)
Metrics: Causal Cross-Entropy Loss (PPL) & 20% Masked Infill Loss
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

# Load 32 sequences x 512 tokens = 16,384 tokens from python_corpus_15b.bin at offset 6B tokens
file_path = PROJECT_ROOT / "data" / "python_corpus_15b.bin"
offset_tokens = 6_000_000_000
offset_bytes = offset_tokens * 2
num_seqs = 32
seq_len = 512

print(f"Reading {num_seqs} held-out sequences ({num_seqs * seq_len} tokens) from {file_path.name} at token offset {offset_tokens:,}...")
data = np.memmap(file_path, dtype=np.uint16, mode="r", offset=offset_bytes, shape=(num_seqs, seq_len))
val_tokens = torch.tensor(data.astype(np.int64))

# Fixed 20% random mask
torch.manual_seed(42)
mask_prob = 0.20
mask_rand = torch.rand(num_seqs, seq_len)
mask_rand[:, 0] = 1.0  # Keep BOS
infill_mask = mask_rand < mask_prob
masked_tokens = val_tokens.clone()
masked_tokens[infill_mask] = 1  # 1 = [MASK] token id

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Evaluation Device: {device}")

val_tokens = val_tokens.to(device)
masked_tokens = masked_tokens.to(device)
infill_mask = infill_mask.to(device)

steps = [3500, 7000, 10500, 14000, 17500, 21000, 24500, "final"]

model_configs = {
    "100M_AR": {
        "dir": PROJECT_ROOT / "checkpoints/ar/100m_5b",
        "label": "100M AR (5B)"
    },
    "100M_COROSred_alpha050": {
        "dir": PROJECT_ROOT / "checkpoints/corosred/unified/100m_5b",
        "label": "100M COROSred (alpha=0.50)"
    },
    "100M_COROSred_alpha075": {
        "dir": PROJECT_ROOT / "checkpoints/corosred/unified/100m_5b_alpha075",
        "label": "100M COROSred (alpha=0.75)"
    }
}

all_results = {}

for key, cfg in model_configs.items():
    all_results[key] = []
    print(f"\n=======================================================")
    print(f"  Evaluating: {cfg['label']}")
    print(f"=======================================================")
    print(f"{'Step':<10} | {'Causal CE':<12} | {'Causal PPL':<12} | {'Infill Loss':<12} | {'Time':<8}")
    print("-" * 65)

    for step in steps:
        fname = f"checkpoint_step_{step}.pt" if step != "final" else "checkpoint_final.pt"
        ckpt_path = cfg["dir"] / fname
        if not ckpt_path.exists():
            continue

        t0 = time.time()
        model, backend, _ = load_model_from_checkpoint(ckpt_path)
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            # 1. Causal Next-Token CE
            logits = model(val_tokens)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = val_tokens[:, 1:].contiguous()
            causal_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            ).item()
            causal_ppl = float(np.exp(min(causal_loss, 20.0)))

            # 2. Infill Loss on 20% masked spans
            infill_logits = model(masked_tokens)
            infill_loss = F.cross_entropy(
                infill_logits[infill_mask],
                val_tokens[infill_mask]
            ).item()

            if device.type == "mps":
                torch.mps.synchronize()

        elapsed = time.time() - t0
        step_label = str(step) if step != "final" else "25432"
        print(f"{step_label:<10} | {causal_loss:<12.4f} | {causal_ppl:<12.2f} | {infill_loss:<12.4f} | {elapsed:<6.2f}s")

        all_results[key].append({
            "step": int(step_label),
            "causal_loss": round(causal_loss, 4),
            "causal_ppl": round(causal_ppl, 2),
            "infill_loss": round(infill_loss, 4)
        })

# Reference: 50M COROSred
ref_path = PROJECT_ROOT / "checkpoints/corosred/unified/50m_2.5b/checkpoint_final.pt"
if ref_path.exists():
    print("\nEvaluating Reference: 50M COROSred (2.5B final)...")
    m50, _, _ = load_model_from_checkpoint(ref_path)
    m50 = m50.to(device)
    m50.eval()
    with torch.no_grad():
        logits = m50(val_tokens)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = val_tokens[:, 1:].contiguous()
        causal_loss_50m = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).item()
        infill_logits = m50(masked_tokens)
        infill_loss_50m = F.cross_entropy(infill_logits[infill_mask], val_tokens[infill_mask]).item()
        if device.type == "mps":
            torch.mps.synchronize()
    all_results["50M_COROSred_ref"] = {
        "step": 12716,
        "causal_loss": round(causal_loss_50m, 4),
        "causal_ppl": round(float(np.exp(causal_loss_50m)), 2),
        "infill_loss": round(infill_loss_50m, 4)
    }
    print(f"50M Final  | Causal CE: {causal_loss_50m:.4f} | Causal PPL: {np.exp(causal_loss_50m):.2f} | Infill Loss: {infill_loss_50m:.4f}")

out_json = PROJECT_ROOT / "logs" / "loss_curves_heldout_15b.json"
with open(out_json, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\n✓ Saved results to {out_json}")
