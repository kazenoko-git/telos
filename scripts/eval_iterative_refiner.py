"""Empirical Benchmark & Evaluation for COROSred Multi-Stage Iterative Refiner.

Compares:
1. Baseline Causal AR Draft
2. Old Tokenwise Binary Re-diffusion (uncalibrated / single token)
3. Region-Dilated Re-diffusion (morphological span routing)
4. Expected-Gain Routed Re-diffusion (v_i = E[ΔL] compute allocation)
5. Self-Conditioned Multi-Stage Refiner (trained on draft-context distribution)
"""

import os
import sys
import math
import time
import json
import numpy as np
import torch
import torch.nn.functional as F

from telos.eval.runner import load_model_from_checkpoint
from telos.diffusion.corosred import (
    crsr_phase_a_loss_fn_pytorch,
    crsr_expected_gain_loss_fn_pytorch,
    crsr_phase_b_self_conditioned_loss_fn_pytorch
)
from telos.diffusion.region import form_refinement_regions
from telos.diffusion.refiner import COROSredRefiner


def run_benchmark(
    ckpt_path: str = "checkpoints/corosred/15m/phase_b",
    dataset_path: str = "data/python_corpus_2.5b.bin",
    num_eval_seqs: int = 100,
    device: str = "mps" if torch.backends.mps.is_available() else "cpu",
    train_steps: int = 150
):
    print("=" * 80)
    print("TELOS COROSRED ITERATIVE REFINEMENT BENCHMARK")
    print(f"Checkpoint: {ckpt_path} | Device: {device} | Eval Seqs: {num_eval_seqs}")
    print("=" * 80)

    # 1. Load Evaluation Corpus
    seq_len = 512
    raw_memmap = np.memmap(dataset_path, dtype=np.uint16, mode="r")
    num_samples = len(raw_memmap) // seq_len
    np.random.seed(42)
    val_indices = np.random.randint(num_samples - 5000, num_samples, size=num_eval_seqs)
    val_matrix = np.array([raw_memmap[idx*seq_len : (idx+1)*seq_len] for idx in val_indices], dtype=np.int64)
    eval_tensor = torch.from_numpy(val_matrix).to(device)

    # 2. Load Base Model
    model, _, vocab_size = load_model_from_checkpoint(ckpt_path)
    model.to(device)
    model.eval()

    # -------------------------------------------------------------------------
    # BASELINE 1: Causal AR Draft Performance
    # -------------------------------------------------------------------------
    print("\n[1/5] Evaluating Baseline Causal AR Draft...")
    with torch.no_grad():
        causal_logits, _ = model(eval_tensor, return_reliability=True, mask_override=True)
        shift_causal = causal_logits[:, :-1, :]
        shift_targets = eval_tensor[:, 1:]
        draft_preds = torch.argmax(shift_causal, dim=-1)

        total_tokens = shift_targets.numel()
        base_correct = (draft_preds == shift_targets).sum().item()
        base_acc = (base_correct / total_tokens) * 100.0

    print(f"✓ Baseline Causal AR Accuracy: {base_acc:.2f}%")

    # -------------------------------------------------------------------------
    # BASELINE 2: Old Single-Token Re-Diffusion (Top 20% Flagged)
    # -------------------------------------------------------------------------
    print("\n[2/5] Evaluating Old Single-Token Re-Diffusion (Naive Tokenwise Surgery)...")
    with torch.no_grad():
        _, raw_r = model(eval_tensor, return_reliability=True, mask_override=True)
        shift_r = raw_r[:, :-1]
        
        # Flag worst 20% tokens with radius=0, max_gap=0 (isolated positions)
        mask_old = form_refinement_regions(shift_r, mode="reliability", top_k_ratio=0.20, radius=0, max_gap=0)
        corrupted_old = torch.cat([eval_tensor[:, :1], torch.where(mask_old, torch.ones_like(draft_preds), draft_preds)], dim=1)
        
        denoise_old = model(corrupted_old, mask_override=False)
        preds_old = torch.argmax(denoise_old[:, 1:], dim=-1)
        repaired_old = torch.where(mask_old, preds_old, draft_preds)
        
        acc_old = ((repaired_old == shift_targets).sum().item() / total_tokens) * 100.0
        mask_pct_old = mask_old.float().mean().item() * 100.0

    print(f"✓ Old Single-Token Accuracy: {acc_old:.2f}% (Mask Rate: {mask_pct_old:.1f}%)")

    # -------------------------------------------------------------------------
    # METHOD 3: Region Routing with Morphological Dilation (Radius=1, Gap=1)
    # -------------------------------------------------------------------------
    print("\n[3/5] Evaluating Region Routing (Span Dilation & Gap Bridging)...")
    with torch.no_grad():
        # Seed 7% flags, dilating into contiguous spans (~15-18% effective mask rate)
        mask_region = form_refinement_regions(shift_r, mode="reliability", top_k_ratio=0.07, radius=1, max_gap=1)
        corrupted_region = torch.cat([eval_tensor[:, :1], torch.where(mask_region, torch.ones_like(draft_preds), draft_preds)], dim=1)
        
        denoise_region = model(corrupted_region, mask_override=False)
        preds_region = torch.argmax(denoise_region[:, 1:], dim=-1)
        repaired_region = torch.where(mask_region, preds_region, draft_preds)
        
        acc_region = ((repaired_region == shift_targets).sum().item() / total_tokens) * 100.0
        mask_pct_region = mask_region.float().mean().item() * 100.0

    print(f"✓ Region Routing Accuracy: {acc_region:.2f}% (Effective Mask Rate: {mask_pct_region:.1f}%)")

    # -------------------------------------------------------------------------
    # METHOD 4 & 5: Train Expected-Gain Router & Self-Conditioned Denoiser
    # -------------------------------------------------------------------------
    print(f"\n[4/5] Training Expected-Gain Router & Self-Conditioned Denoiser ({train_steps} steps)...")
    model_sc, _, _ = load_model_from_checkpoint(ckpt_path)
    model_sc.to(device)
    model_sc.train()

    # Step 4a: Fast recalibration of Expected-Gain Router Head (freeze backbone)
    for p in model_sc.parameters():
        p.requires_grad = False
    for p in model_sc.reliability_head.parameters():
        p.requires_grad = True

    opt_router = torch.optim.AdamW(model_sc.reliability_head.parameters(), lr=1e-3, weight_decay=0.01)
    t0 = time.time()
    for step in range(train_steps):
        idx_b = np.random.randint(0, num_samples - 5000, size=16)
        b_tr = torch.from_numpy(np.array([raw_memmap[i*seq_len : (i+1)*seq_len] for i in idx_b], dtype=np.int64)).to(device)
        
        opt_router.zero_grad()
        loss_gain, _ = crsr_expected_gain_loss_fn_pytorch(model_sc, b_tr, vocab_size, mask_token_id=1, mask_prob=0.15)
        loss_gain.backward()
        opt_router.step()

    # Step 4b: Fast self-conditioned adaptation of the Denoiser backbone (small lr fine-tune)
    for p in model_sc.parameters():
        p.requires_grad = True

    opt_all = torch.optim.AdamW(model_sc.parameters(), lr=3e-5, weight_decay=0.01)
    for step in range(train_steps):
        idx_b = np.random.randint(0, num_samples - 5000, size=8)
        b_tr = torch.from_numpy(np.array([raw_memmap[i*seq_len : (i+1)*seq_len] for i in idx_b], dtype=np.int64)).to(device)
        
        opt_all.zero_grad()
        loss_sc, _ = crsr_phase_b_self_conditioned_loss_fn_pytorch(
            model_sc, b_tr, vocab_size, mask_token_id=1, mask_prob=0.15, self_cond_prob=0.6
        )
        loss_sc.backward()
        opt_all.step()

    model_sc.eval()
    print(f"✓ Training complete in {time.time() - t0:.1f}s")

    # -------------------------------------------------------------------------
    # METHOD 5: Multi-Stage Iterative Refiner Evaluation (1-Pass and 2-Pass)
    # -------------------------------------------------------------------------
    print("\n[5/5] Evaluating Multi-Stage Expected-Gain Refiner...")
    refiner_1pass = COROSredRefiner(
        model_sc,
        mask_token_id=1,
        mode="expected_gain",
        threshold=0.20,
        top_k_ratio=0.08,
        radius=1,
        max_gap=1,
        max_passes=1
    )

    refiner_2pass = COROSredRefiner(
        model_sc,
        mask_token_id=1,
        mode="expected_gain",
        threshold=0.20,
        top_k_ratio=0.08,
        radius=1,
        max_gap=1,
        max_passes=2
    )

    initial_draft_full = torch.cat([eval_tensor[:, :1], draft_preds], dim=1)

    with torch.no_grad():
        refined_1pass, m_1p = refiner_1pass.refine(initial_draft_full, prompt_len=1, return_metrics=True)
        acc_1pass = ((refined_1pass[:, 1:] == shift_targets).sum().item() / total_tokens) * 100.0

        refined_2pass, m_2p = refiner_2pass.refine(initial_draft_full, prompt_len=1, return_metrics=True)
        acc_2pass = ((refined_2pass[:, 1:] == shift_targets).sum().item() / total_tokens) * 100.0

    print(f"✓ 1-Pass Expected-Gain Refiner: {acc_1pass:.2f}% (Refined: {m_1p['mask_pct_per_pass']})")
    print(f"✓ 2-Pass Expected-Gain Refiner: {acc_2pass:.2f}% (Refined: {m_2p['mask_pct_per_pass']})")

    # -------------------------------------------------------------------------
    # SUMMARY TABLE
    # -------------------------------------------------------------------------
    summary = {
        "baseline_ar_draft_acc": round(base_acc, 2),
        "old_single_token_red_acc": round(acc_old, 2),
        "region_routing_acc": round(acc_region, 2),
        "expected_gain_1pass_acc": round(acc_1pass, 2),
        "expected_gain_2pass_acc": round(acc_2pass, 2),
        "delta_over_baseline": round(acc_2pass - base_acc, 2),
        "delta_over_old_red": round(acc_2pass - acc_old, 2),
    }

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY & EMPIRICAL GAIN COMPARISON")
    print("=" * 80)
    print(f"Baseline Causal Draft (AR):       {summary['baseline_ar_draft_acc']}%")
    print(f"Old Single-Token Re-Diffusion:    {summary['old_single_token_red_acc']}%")
    print(f"Region Routing (Morphology):      {summary['region_routing_acc']}%")
    print(f"1-Pass Expected-Gain Refiner:     {summary['expected_gain_1pass_acc']}%")
    print(f"2-Pass Expected-Gain Refiner:     {summary['expected_gain_2pass_acc']}%")
    print("-" * 80)
    print(f"Net Gain Over AR Baseline:        +{summary['delta_over_baseline']}%")
    print(f"Net Gain Over Old Re-Diffusion:   +{summary['delta_over_old_red']}%")
    print("=" * 80)

    # Save results
    os.makedirs("logs", exist_ok=True)
    with open("logs/eval_iterative_refiner_benchmark.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved benchmark report to logs/eval_iterative_refiner_benchmark.json")

    return summary


if __name__ == "__main__":
    run_benchmark()
