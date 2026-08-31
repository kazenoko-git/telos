"""
Comprehensive Evaluation Suite for 100M COROSred (Phase A & AR Capabilities).

Benchmarks:
1. Language Model Validation Loss & Perplexity (Causal Next-Token Prediction)
2. Stratified Reliability ROC-AUC:
   - High-Entropy Stratum (Learned Error Calibration vs Softmax Entropy: Delta-AUC)
   - Low-Entropy Stratum (Confidently Wrong Recall: Detection of Subtle Hallucinations)
3. Synthetic Out-of-Distribution (OOD) Corruption Localization ROC-AUC
4. Python Code Syntax & Multi-Token Probe Completion
"""

import os
import sys
import math
import json
import yaml
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mdiff.model.transformer import TelosTransformer, TelosConfig
from mdiff.data.tokenizer import load_tokenizer


def compute_roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Computes Receiver Operating Characteristic Area Under Curve (ROC-AUC)."""
    order = np.argsort(-scores)
    sorted_labels = labels[order]

    n_pos = np.sum(sorted_labels == 1)
    n_neg = np.sum(sorted_labels == 0)

    if n_pos == 0 or n_neg == 0:
        return 0.5

    tpr = np.cumsum(sorted_labels == 1) / n_pos
    fpr = np.cumsum(sorted_labels == 0) / n_neg

    tpr = np.concatenate(([0.0], tpr))
    fpr = np.concatenate(([0.0], fpr))

    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(tpr, fpr))
    elif hasattr(np, "trapz"):
        return float(np.trapz(tpr, fpr))
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1])) / 2.0)


def evaluate_corosred_100m(
    ckpt_path: str,
    config_path: str = "configs/unified/100m/telos_100m_r1.yaml",
    dataset_path: str = "data/python_corpus_2.5b.bin",
    device: str = "mps" if torch.backends.mps.is_available() else "cpu",
    entropy_threshold: float = 1.5,
    k_amb: int = 5,
    num_eval_seqs: int = 500
):
    print("=" * 80)
    print(f"EVALUATING 100M COROSRED MODEL: {ckpt_path}")
    print(f"Device: {device} | Eval Sequences: {num_eval_seqs}")
    print("=" * 80)

    # 1. Load Architecture Config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    m_cfg = cfg["model"]

    telos_cfg = TelosConfig(
        vocab_size=m_cfg["vocab_size"],
        d_model=m_cfg["d_model"],
        n_layers=m_cfg["n_layers"],
        n_heads=m_cfg["n_heads"],
        n_kv_heads=m_cfg["n_kv_heads"],
        max_seq_len=m_cfg["seq_len"],
        is_causal=True
    )

    # 2. Instantiate and Load Weights
    model = TelosTransformer(telos_cfg)
    state_dict = load_file(ckpt_path)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"✓ Model loaded successfully ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")

    # 3. Load Validation Dataset
    if not os.path.exists(dataset_path):
        print(f"Dataset {dataset_path} not found. Searching alternatives...")
        for alt in ["data/python_corpus_1.7b.bin", "data/python_corpus.bin"]:
            if os.path.exists(alt):
                dataset_path = alt
                break

    seq_len = m_cfg["seq_len"]
    itemsize = 2 if "2.5b" in dataset_path else 4
    dtype = np.uint16 if itemsize == 2 else np.uint32
    raw_memmap = np.memmap(dataset_path, dtype=dtype, mode="r")
    num_samples = len(raw_memmap) // seq_len
    print(f"✓ Loaded evaluation corpus: {num_samples:,} total sequences")

    # Sample evaluation batch from the tail (validation split)
    np.random.seed(42)
    val_indices = np.random.randint(max(0, num_samples - 50000), num_samples, size=num_eval_seqs)
    eval_matrix = np.array([raw_memmap[idx*seq_len : (idx+1)*seq_len] for idx in val_indices], dtype=np.int64)

    # 4. Evaluate Teacher-Forced Causal AR & Reliability Head
    print("\n[1/3] Running Teacher-Forced AR & Stratified Reliability Evaluation...")
    total_loss = 0.0
    all_entropy = []
    all_r_scores = []
    all_labels = []
    all_valid_masks = []

    batch_size = 16
    num_batches = math.ceil(num_eval_seqs / batch_size)

    with torch.no_grad():
        for b in range(num_batches):
            b_seqs = torch.from_numpy(eval_matrix[b*batch_size : (b+1)*batch_size]).to(device)
            logits, r_scores = model(b_seqs, return_reliability=True)

            shift_logits = logits[:, :-1, :].contiguous()
            shift_r_scores = r_scores[:, :-1].contiguous()
            shift_targets = b_seqs[:, 1:].contiguous()

            # Loss computation
            loss = nn.functional.cross_entropy(shift_logits.view(-1, m_cfg["vocab_size"]), shift_targets.view(-1))
            total_loss += loss.item() * len(b_seqs)

            # Softmax Entropy: H(p) = -sum(p * log(p))
            probs = torch.softmax(shift_logits, dim=-1)
            log_probs = torch.log(torch.clamp(probs, min=1e-10, max=1.0))
            entropy = -torch.sum(probs * log_probs, dim=-1)

            # Correctness labels: 1 if top-1 matches target, else 0
            argmax_preds = torch.argmax(shift_logits, dim=-1)
            labels = (argmax_preds == shift_targets)

            # Ambiguity exclusion (top-k)
            _, top_k_indices = torch.topk(shift_logits, k_amb, dim=-1)
            expanded_targets = shift_targets.unsqueeze(-1)
            is_target_in_top_k = (top_k_indices == expanded_targets).any(dim=-1)
            is_ambiguous = is_target_in_top_k & ~labels
            valid_mask = ~is_ambiguous

            all_entropy.append(entropy.cpu().numpy().flatten())
            all_r_scores.append(shift_r_scores.cpu().numpy().flatten())
            all_labels.append(labels.cpu().numpy().flatten())
            all_valid_masks.append(valid_mask.cpu().numpy().flatten())

    val_loss = total_loss / num_eval_seqs
    val_ppl = math.exp(min(val_loss, 20.0))

    np_entropy = np.concatenate(all_entropy)
    np_r_scores = np.concatenate(all_r_scores)
    np_labels = np.concatenate(all_labels).astype(np.int32)
    np_valid = np.concatenate(all_valid_masks).astype(bool)

    # Filter by valid (non-ambiguous) mask
    valid_entropy = np_entropy[np_valid]
    valid_r_scores = np_r_scores[np_valid]
    valid_labels = np_labels[np_valid]

    # Stratum 1: High Entropy (Learned Calibration vs Softmax Entropy)
    high_ent_mask = valid_entropy >= entropy_threshold
    if np.sum(high_ent_mask) > 10:
        high_entropy_ent_auc = compute_roc_auc(-valid_entropy[high_ent_mask], valid_labels[high_ent_mask])
        high_entropy_head_auc = compute_roc_auc(valid_r_scores[high_ent_mask], valid_labels[high_ent_mask])
        delta_auc = high_entropy_head_auc - high_entropy_ent_auc
    else:
        high_entropy_ent_auc, high_entropy_head_auc, delta_auc = 0.5, 0.5, 0.0

    # Stratum 2: Low Entropy (Confidently Wrong Recall)
    low_ent_mask = valid_entropy < entropy_threshold
    confidently_wrong = (valid_labels[low_ent_mask] == 0)
    total_conf_wrong = np.sum(confidently_wrong)
    if total_conf_wrong > 0:
        detected_by_head = np.sum((valid_r_scores[low_ent_mask] < 0.0) & confidently_wrong)
        low_entropy_recall = float(detected_by_head / total_conf_wrong)
    else:
        low_entropy_recall = 0.0

    # Overall Head AUC
    overall_head_auc = compute_roc_auc(valid_r_scores, valid_labels)
    overall_ent_auc = compute_roc_auc(-valid_entropy, valid_labels)

    # 5. Evaluate Synthetic Prose Corruption Localization (OOD Test)
    print("\n[2/3] Running Synthetic Out-of-Distribution Corruption Localization...")
    B_ood, T_ood = eval_matrix.shape[0], eval_matrix.shape[1]
    rand_mask = (np.random.uniform(0, 1, (B_ood, T_ood)) < 0.05)
    rand_mask[:, 0] = False  # Keep prompt token clean

    corrupted_matrix = eval_matrix.copy()
    corrupted_matrix[rand_mask] = np.random.randint(10, m_cfg["vocab_size"], size=np.sum(rand_mask))

    all_ood_scores = []
    with torch.no_grad():
        for b in range(num_batches):
            b_seqs = torch.from_numpy(corrupted_matrix[b*batch_size : (b+1)*batch_size]).to(device)
            _, r_scores = model(b_seqs, return_reliability=True)
            all_ood_scores.append(r_scores.cpu().numpy().flatten())

    ood_scores = np.concatenate(all_ood_scores)
    clean_ground_truth = (~rand_mask).flatten().astype(np.int32)
    ood_localization_auc = compute_roc_auc(ood_scores, clean_ground_truth)

    # 6. Summary Report
    results = {
        "model": Path(ckpt_path).name,
        "parameters": f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M",
        "val_loss": round(val_loss, 4),
        "val_perplexity": round(val_ppl, 2),
        "overall_head_auc": round(overall_head_auc, 4),
        "overall_entropy_auc": round(overall_ent_auc, 4),
        "high_entropy_head_auc": round(high_entropy_head_auc, 4),
        "high_entropy_entropy_auc": round(high_entropy_ent_auc, 4),
        "delta_auc": round(delta_auc, 4),
        "phase_a_gate_passed": bool(delta_auc >= 0.05),
        "low_entropy_error_recall": round(low_entropy_recall, 4),
        "ood_corruption_localization_auc": round(ood_localization_auc, 4),
    }

    print("\n" + "=" * 80)
    print("COROSRED 100M EVALUATION RESULTS")
    print("=" * 80)
    print(f"Validation Loss:         {results['val_loss']} (Perplexity: {results['val_perplexity']})")
    print(f"Overall Head ROC-AUC:    {results['overall_head_auc']} vs Entropy: {results['overall_entropy_auc']}")
    print(f"High-Entropy Head AUC:   {results['high_entropy_head_auc']} vs Entropy: {results['high_entropy_entropy_auc']}")
    print(f"Delta-AUC (Gain):        +{results['delta_auc']} {'[GATE PASSED ✓]' if results['phase_a_gate_passed'] else '[GATE FAILED]'}")
    print(f"Confidently-Wrong Recall: {results['low_entropy_error_recall']*100:.1f}%")
    print(f"OOD Corruption Loc AUC:  {results['ood_corruption_localization_auc']}")
    print("=" * 80)

    # Save results to evals directory
    out_dir = PROJECT_ROOT / "evals" / "probe_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "corosred_100m_r1_eval.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved detailed evaluation metrics to {out_file}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate 100M COROSred Model")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint safetensors file")
    parser.add_argument("--dataset", type=str, default="data/python_corpus_2.5b.bin", help="Path to evaluation dataset")
    args = parser.parse_args()

    ckpt = args.ckpt
    if ckpt is None:
        ckpt_dir = Path("checkpoints/corosred/100m/telos_100m_r1")
        if (ckpt_dir / "model.safetensors").exists():
            ckpt = str(ckpt_dir / "model.safetensors")
        else:
            safetensors_files = sorted(list(ckpt_dir.glob("*.safetensors")), key=lambda p: p.stat().st_mtime)
            if safetensors_files:
                ckpt = str(safetensors_files[-1])
            else:
                raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")

    evaluate_corosred_100m(ckpt_path=ckpt, dataset_path=args.dataset)
