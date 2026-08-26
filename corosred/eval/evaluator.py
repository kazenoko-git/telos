"""
COROSred Experiment 0 Evaluator.

Benchmarks:
1. Per-Stratum Head AUC vs. Raw Softmax Entropy AUC (High-Entropy vs. Low-Entropy).
2. Router Exposure Bias (Teacher-Forced token reliability vs. Free-Run draft reliability).
3. Synthetic Prose Corruption Localization (OOD Test).
"""

import math
import numpy as np
import mlx.core as mx


def compute_roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Computes Receiver Operating Characteristic Area Under Curve (ROC-AUC)."""
    # Sort examples by descending order of predicted score
    order = np.argsort(-scores)
    sorted_labels = labels[order]

    n_pos = np.sum(sorted_labels == 1)
    n_neg = np.sum(sorted_labels == 0)

    if n_pos == 0 or n_neg == 0:
        return 0.5  # Degenerate case when all labels are identical (kinda pointless)

    # Calculate cumulative true positive and false positive rates
    tpr = np.cumsum(sorted_labels == 1) / n_pos
    fpr = np.cumsum(sorted_labels == 0) / n_neg

    # Numerical integration using trapezoidal rule
    return float(np.trapz(tpr, fpr))


class COROSredExperiment0Evaluator:
    """Evaluates Phase A Reliability Head against raw softmax entropy baselines."""

    def __init__(self, model, entropy_threshold: float = 1.5, k_amb: int = 5):
        self.model = model
        self.entropy_threshold = entropy_threshold
        self.k_amb = k_amb

    def evaluate_teacher_forced(self, sequences: mx.array) -> dict:
        """
        Test A: Teacher-Forced Token Reliability.
        Compares Head AUC vs Raw Entropy AUC separated into High-Entropy and Low-Entropy subsets.
        """
        # Forward pass under causal mask
        logits, raw_r_scores = self.model(sequences, is_causal=True, return_reliability=True)

        shift_logits = logits[:, :-1, :].astype(mx.float32)
        shift_r_scores = raw_r_scores[:, :-1].astype(mx.float32)
        shift_targets = sequences[:, 1:]

        # Compute softmax distribution and entropy: H(p) = -sum(p * log(p))
        probs = mx.softmax(shift_logits, axis=-1)
        log_probs = mx.log(mx.clip(probs, 1e-10, 1.0))
        entropy = -mx.sum(probs * log_probs, axis=-1)

        # Ground truth correctness labels: 1 if model top pick matches GT, else 0
        argmax_preds = mx.argmax(shift_logits, axis=-1)
        labels = (argmax_preds == shift_targets)

        # Ambiguity exclusion: drop positions where target is in top-K but not top-1
        top_k_indices = mx.argpartition(shift_logits, -self.k_amb, axis=-1)[..., -self.k_amb:]
        expanded_targets = mx.expand_dims(shift_targets, -1)
        is_target_in_top_k = mx.any(top_k_indices == expanded_targets, axis=-1)
        is_ambiguous = mx.logical_and(is_target_in_top_k, mx.logical_not(labels))
        valid_eval_mask = mx.logical_not(is_ambiguous)

        # Convert tensors to numpy for stratified metrics computation
        np_entropy = np.array(entropy)[np.array(valid_eval_mask)]
        np_r_scores = np.array(shift_r_scores)[np.array(valid_eval_mask)]
        np_labels = np.array(labels)[np.array(valid_eval_mask)].astype(np.int32)

        # Stratum 1: High Entropy Subset
        high_ent_mask = np_entropy >= self.entropy_threshold
        if np.sum(high_ent_mask) > 10:
            # Negative entropy as error predictor (lower entropy -> higher reliability)
            high_entropy_ent_auc = compute_roc_auc(-np_entropy[high_ent_mask], np_labels[high_ent_mask])
            high_entropy_head_auc = compute_roc_auc(np_r_scores[high_ent_mask], np_labels[high_ent_mask])
            delta_auc = high_entropy_head_auc - high_entropy_ent_auc
        else:
            high_entropy_ent_auc, high_entropy_head_auc, delta_auc = 0.5, 0.5, 0.0

        # Stratum 2: Low Entropy Subset (confidently-wrong recall test)
        low_ent_mask = np_entropy < self.entropy_threshold
        if np.sum(low_ent_mask) > 10:
            confidently_wrong_errors = np_labels[low_ent_mask] == 0
            # Check how many confidently-wrong errors receive negative reliability logits
            detected_by_head = np.sum((np_r_scores[low_ent_mask] < 0.0) & confidently_wrong_errors)
            total_wrong = max(np.sum(confidently_wrong_errors), 1)
            low_entropy_recall = float(detected_by_head / total_wrong)
        else:
            low_entropy_recall = 0.0

        return {
            "high_entropy_head_auc": high_entropy_head_auc,
            "high_entropy_entropy_auc": high_entropy_ent_auc,
            "delta_auc": delta_auc,
            "phase_a_gate_passed": bool(delta_auc >= 0.05),
            "low_entropy_error_recall": low_entropy_recall,
        }

    def evaluate_prose_corruption(self, clean_sequences: mx.array, corruption_prob: float = 0.05) -> dict:
        """
        Test B: Synthetic Prose Corruption Localization (OOD Generalization Test).
        """
        B, T = clean_sequences.shape
        # Inject random synthetic corruption tokens
        rand_mask = np.random.uniform(0, 1, (B, T)) < corruption_prob
        # Exclude position 0 from corruption to maintain valid prompt prefix
        rand_mask[:, 0] = False

        corrupted = np.array(clean_sequences)
        corrupted[rand_mask] = np.random.randint(10, 1000, size=np.sum(rand_mask))

        # Forward pass on corrupted sequences
        _, raw_r_scores = self.model(mx.array(corrupted), is_causal=True, return_reliability=True)
        scores = np.array(raw_r_scores)

        # Ground truth label: 0 if corrupted (unreliable), 1 if clean (reliable)
        corruption_labels = (~rand_mask).astype(np.int32)
        localization_auc = compute_roc_auc(scores.flatten(), corruption_labels.flatten())

        return {
            "prose_localization_auc": localization_auc,
            "total_corruptions_evaluated": int(np.sum(rand_mask)),
        }
