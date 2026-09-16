"""
Fast CPU calculation of:
1. alpha * Delta L_causal and beta * Delta L_infill
2. Gradient norms:
   G_c = ||nabla_theta L_c||_2 (L2 norm of causal loss gradient)
   G_i = ||nabla_theta L_i||_2 (L2 norm of infill loss gradient)
   R = (alpha * G_c) / (beta * G_i) (Effective gradient force ratio)
Across 50M, 75M, 100M COROSred and pure AR models.
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from telos.eval.runner import load_model_from_checkpoint

# Load held-out validation batch from python_corpus_15b.bin
file_path = PROJECT_ROOT / "data" / "python_corpus_15b.bin"
offset_tokens = 6_000_000_000
data = np.memmap(file_path, dtype=np.uint16, mode="r", offset=offset_tokens * 2, shape=(16, 512))
val_tokens = torch.tensor(data.astype(np.int64))

torch.manual_seed(42)
infill_mask = (torch.rand(16, 512) < 0.20)
infill_mask[:, 0] = False
masked_tokens = val_tokens.clone()
masked_tokens[infill_mask] = 1

device = torch.device("cpu")
print(f"Evaluation Device: {device}", flush=True)

models = [
    {
        "name": "50M COROSred (CUDA)",
        "path": "checkpoints/corosred/50m_lightning/checkpoint_final.pt",
        "alpha_final": 0.50, "beta_final": 0.35,
        "alpha_avg": 0.627, "beta_avg": 0.278
    },
    {
        "name": "75M COROSred (TPU)",
        "path": "checkpoints/corosred/unified/75m_python/checkpoint_final.pt",
        "alpha_final": 0.50, "beta_final": 0.35,
        "alpha_avg": 0.627, "beta_avg": 0.278
    },
    {
        "name": "100M COROSred (alpha=0.50)",
        "path": "checkpoints/corosred/unified/100m_5b/checkpoint_final.pt",
        "alpha_final": 0.50, "beta_final": 0.35,
        "alpha_avg": 0.627, "beta_avg": 0.278
    },
    {
        "name": "100M COROSred (alpha=0.75)",
        "path": "checkpoints/corosred/unified/100m_5b_alpha075/checkpoint_final.pt",
        "alpha_final": 0.75, "beta_final": 0.20,
        "alpha_avg": 0.784, "beta_avg": 0.165
    },
    {
        "name": "100M Pure AR (5.0B)",
        "path": "checkpoints/ar/100m_5b/checkpoint_final.pt",
        "alpha_final": 1.00, "beta_final": 0.00,
        "alpha_avg": 1.00, "beta_avg": 0.00
    }
]

# Baseline 100M AR reference losses
AR_CAUSAL_REF = 1.7842
AR_INFILL_REF = 6.9852

computed_stats = []

for m_info in models:
    p = PROJECT_ROOT / m_info["path"]
    if not p.exists():
        continue

    print(f"Evaluating {m_info['name']}...", flush=True)
    model, _, _ = load_model_from_checkpoint(p)
    model = model.to(device)
    model.train()  # Enable gradient tracking

    # 1. Causal Loss & Gradient Norm G_c
    model.zero_grad()
    logits_c = model(val_tokens)
    loss_c = F.cross_entropy(
        logits_c[:, :-1, :].reshape(-1, logits_c.size(-1)),
        val_tokens[:, 1:].reshape(-1)
    )
    loss_c.backward()

    # Calculate L2 norm of gradients
    grad_norms_c = [param.grad.data.norm(2).item() ** 2 for param in model.parameters() if param.grad is not None]
    G_c = float(np.sqrt(sum(grad_norms_c)))
    l_c_val = loss_c.item()

    # 2. Infill Loss & Gradient Norm G_i
    model.zero_grad()
    logits_i = model(masked_tokens)
    
    # Clean vectorized masked cross-entropy without boolean indexing in autograd graph
    loss_flat = F.cross_entropy(logits_i.view(-1, logits_i.size(-1)), val_tokens.view(-1), reduction="none")
    mask_flat = infill_mask.view(-1).float()
    loss_i = (loss_flat * mask_flat).sum() / mask_flat.sum()
    loss_i.backward()

    grad_norms_i = [param.grad.data.norm(2).item() ** 2 for param in model.parameters() if param.grad is not None]
    G_i = float(np.sqrt(sum(grad_norms_i)))
    l_i_val = loss_i.item()

    alpha = m_info["alpha_final"]
    beta = m_info["beta_final"]

    delta_lc = l_c_val - AR_CAUSAL_REF
    delta_li = l_i_val - AR_INFILL_REF

    weighted_dlc = alpha * delta_lc
    weighted_dli = beta * delta_li
    net_balance = weighted_dlc + weighted_dli

    # Gradient force ratio R = (alpha * G_c) / (beta * G_i)
    if beta > 0 and G_i > 0:
        R_final = (alpha * G_c) / (beta * G_i)
        R_avg = (m_info["alpha_avg"] * G_c) / (m_info["beta_avg"] * G_i)
    else:
        R_final = float("inf")
        R_avg = float("inf")

    raw_grad_ratio = G_c / G_i if G_i > 0 else float("inf")

    computed_stats.append({
        "name": m_info["name"],
        "loss_c": l_c_val,
        "loss_i": l_i_val,
        "delta_lc": delta_lc,
        "delta_li": delta_li,
        "alpha": alpha,
        "beta": beta,
        "weighted_dlc": weighted_dlc,
        "weighted_dli": weighted_dli,
        "net_balance": net_balance,
        "G_c": G_c,
        "G_i": G_i,
        "raw_grad_ratio": raw_grad_ratio,
        "R_final": R_final,
        "R_avg": R_avg
    })

print("\n" + "=" * 110, flush=True)
print(f"  PART 1: WEIGHTED DELTA LOSS AUDIT (alpha * Delta L_c and beta * Delta L_i)", flush=True)
print("=" * 110, flush=True)
print(f"{'Model Name':<28} | {'Delta L_c':<10} | {'alpha*Delta L_c':<16} | {'Delta L_i':<10} | {'beta*Delta L_i':<16} | {'Net Balance'}", flush=True)
print("-" * 110, flush=True)
for s in computed_stats:
    print(f"{s['name']:<28} | {s['delta_lc']:>+9.4f}  | {s['weighted_dlc']:>+15.4f}  | {s['delta_li']:>+9.4f}  | {s['weighted_dli']:>+15.4f}  | {s['net_balance']:>+10.4f} nats", flush=True)
print("=" * 110, flush=True)

print("\n" + "=" * 110, flush=True)
print(f"  PART 2: GRADIENT NORM AUDIT: G_c = ||grad L_c||, G_i = ||grad L_i||, R = (alpha*G_c) / (beta*G_i)", flush=True)
print("=" * 110, flush=True)
print(f"{'Model Name':<28} | {'G_c (Causal)':<12} | {'G_i (Infill)':<12} | {'Raw G_c/G_i':<12} | {'R (Final a/b)':<14} | {'R (Time-Avg a/b)'}", flush=True)
print("-" * 110, flush=True)
for s in computed_stats:
    r_fin_str = f"{s['R_final']:.3f}" if s['R_final'] != float('inf') else "inf (Pure AR)"
    r_avg_str = f"{s['R_avg']:.3f}" if s['R_avg'] != float('inf') else "inf (Pure AR)"
    print(f"{s['name']:<28} | {s['G_c']:<12.4f} | {s['G_i']:<12.4f} | {s['raw_grad_ratio']:<12.3f} | {r_fin_str:<14} | {r_avg_str}", flush=True)
print("=" * 110, flush=True)
