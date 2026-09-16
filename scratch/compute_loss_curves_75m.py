"""
Compute Detailed Validation Loss Trajectory & Diagnostic Probe Trajectory for 75M COROSred:
Steps: [4000, 8000, 12000, 16000, 19074 (final)]
Evaluates on:
1. Held-Out 15B Python validation slice (causal CE, causal PPL, 20% masked infill loss)
2. 100 Official Contextual Probes (Causal CE, Causal Top-1, Infill CE, Infill Top-1)
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

from telos.eval.runner import load_model_from_checkpoint, evaluate_probes
from telos.eval.probes import load_contextual_probes
from telos.data.tokenizer import load_tokenizer

# 1. Held-Out Validation Slice
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

# Use CPU for deterministic probe evaluation across devices
device = torch.device("cpu")
print(f"Evaluation Device: {device}")

val_tokens = val_tokens.to(device)
masked_tokens = masked_tokens.to(device)
infill_mask = infill_mask.to(device)

tokenizer = load_tokenizer()
probes = load_contextual_probes()

ckpt_dir = PROJECT_ROOT / "checkpoints/corosred/unified/75m_python"
step_list = [
    (4000, ckpt_dir / "checkpoint_step_4000.pt"),
    (8000, ckpt_dir / "checkpoint_step_8000.pt"),
    (12000, ckpt_dir / "checkpoint_step_12000.pt"),
    (16000, ckpt_dir / "checkpoint_step_16000.pt"),
    (19074, ckpt_dir / "checkpoint_final.pt"),
]

trajectory = []

print("\n" + "=" * 95)
print(f"  75M COROSred TRAINING TRAJECTORY & LOSS CURVE AUDIT")
print("=" * 95)
print(f"{'Step':<8} | {'Causal CE':<10} | {'Causal PPL':<10} | {'Infill Loss':<11} | {'Probe Causal':<16} | {'Probe Infill':<16} | {'Time'}")
print("-" * 95)

for step_num, ckpt_path in step_list:
    if not ckpt_path.exists():
        print(f"Skipping step {step_num} (file {ckpt_path.name} not found)")
        continue

    t0 = time.time()
    model, backend, _ = load_model_from_checkpoint(ckpt_path)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        # 1. Causal Next-Token CE on 15B held-out validation slice
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
        infill_ppl = float(np.exp(min(infill_loss, 20.0)))

    # 3. Contextual Probes Evaluation (100 probes: Causal Top-1 & Infill Top-1)
    probe_results = evaluate_probes(
        model=model,
        tokenizer=tokenizer,
        backend=backend,
        paradigm="corosred",
        num_probes=100,
        probe_type="both"
    )

    causal_p1 = probe_results.get("causal", {}).get("top1_pct", 0.0)
    causal_p_ce = probe_results.get("causal", {}).get("overall_ce", 0.0)
    infill_p1 = probe_results.get("infill", {}).get("top1_pct", 0.0)
    infill_p_ce = probe_results.get("infill", {}).get("overall_ce", 0.0)

    elapsed = time.time() - t0

    causal_str = f"{causal_p1:>5.1f}% (CE {causal_p_ce:.2f})"
    infill_str = f"{infill_p1:>5.1f}% (CE {infill_p_ce:.2f})"

    print(f"{step_num:<8d} | {causal_loss:<10.4f} | {causal_ppl:<10.2f} | {infill_loss:<11.4f} | {causal_str:<16} | {infill_str:<16} | {elapsed:<5.1f}s")

    trajectory.append({
        "step": step_num,
        "causal_loss": round(causal_loss, 4),
        "causal_ppl": round(causal_ppl, 2),
        "infill_loss": round(infill_loss, 4),
        "infill_ppl": round(infill_ppl, 2),
        "probe_causal_top1_pct": causal_p1,
        "probe_causal_ce": round(causal_p_ce, 3),
        "probe_infill_top1_pct": infill_p1,
        "probe_infill_ce": round(infill_p_ce, 3),
    })

out_json = PROJECT_ROOT / "logs" / "loss_curve_75m_trajectory.json"
with open(out_json, "w") as f:
    json.dump(trajectory, f, indent=2)

print("\n" + "=" * 95)
print(f"✓ Saved 75M loss curve trajectory to {out_json}")
print("=" * 95)
