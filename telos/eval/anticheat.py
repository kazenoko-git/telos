"""
Anti-Cheating & Suffix-Copy Detection Engine for Télos Bidirectional & Infill Models.

Directly diagnoses boundary cheating shortcuts:
- Single-token mask: prefix + [MASK] + suffix
  Models often learn a degenerate shortcut: copying the literal first token of the suffix (suffix[0]),
  which frequently coincides with punctuation (':', ')', newline) creating an illusion of accuracy.
- Multi-token span chunk masking: prefix + [MASK]*K + suffix (K in {1, 2, 4, 8, 16})
  When K > 1, the adjacent token is another [MASK] and copying suffix[0] to the first position
  fails, exposing whether the model learned true semantic infilling or boundary copying.
"""

import math
from typing import Dict, Any, List, Optional
import numpy as np


def compute_boundary_copy_metrics(
    predicted_tokens: List[int],
    prefix_tokens: List[int],
    suffix_tokens: List[int],
    target_tokens: List[int]
) -> Dict[str, Any]:
    """
    Computes whether the model copied literal boundary tokens rather than predicting target.

    Args:
        predicted_tokens: Sequence of predicted token IDs for the masked span.
        prefix_tokens: Sequence of prompt token IDs preceding the mask.
        suffix_tokens: Sequence of context token IDs following the mask.
        target_tokens: True ground-truth token IDs that were masked.

    Returns:
        Dict with boolean flags for boundary copying and token match accuracy.
    """
    if not predicted_tokens:
        return {
            "copied_suffix_first": False,
            "copied_prefix_last": False,
            "is_exact_match": False,
            "token_accuracy": 0.0,
        }

    first_pred = predicted_tokens[0]

    # Check if the first predicted token matches the literal first token of the suffix
    has_suffix = len(suffix_tokens) > 0
    copied_suffix_first = bool(has_suffix and (first_pred == suffix_tokens[0]))

    # Check if the first predicted token matches the literal last token of the prefix
    has_prefix = len(prefix_tokens) > 0
    copied_prefix_last = bool(has_prefix and (first_pred == prefix_tokens[-1]))

    # Target comparison
    span_len = len(target_tokens)
    is_exact_match = (predicted_tokens[:span_len] == target_tokens[:span_len])

    matched_tokens = sum(
        1 for p, t in zip(predicted_tokens[:span_len], target_tokens[:span_len]) if p == t
    )
    token_acc = matched_tokens / max(span_len, 1)

    return {
        "copied_suffix_first": copied_suffix_first,
        "copied_prefix_last": copied_prefix_last,
        "is_exact_match": is_exact_match,
        "token_accuracy": float(token_acc),
    }


def evaluate_span_infill_probe(
    model,
    tokenizer,
    backend: str,
    prefix_text: str,
    target_text: str,
    suffix_text: str,
    mask_token_id: int = 1,
    span_lengths: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Evaluates infilling performance across varying span lengths K in {1, 2, 4, 8, 16}.
    For each span length K, tests if the model predicts the ground truth or cheats by
    copying adjacent boundary tokens.

    Returns:
        Dict detailing results per span length.
    """
    if span_lengths is None:
        span_lengths = [1, 2, 4, 8]

    # Context-aware tokenization to handle ByteLevel BPE leading-space byte ('Ġ')
    prefix_ids = tokenizer.encode(prefix_text).ids
    full_target_ids = tokenizer.encode(prefix_text + target_text).ids[len(prefix_ids):]
    if not full_target_ids:
        full_target_ids = tokenizer.encode(target_text).ids

    suffix_ids = tokenizer.encode(suffix_text).ids if suffix_text else []

    span_results = {}

    for k in span_lengths:
        # Determine target segment of length k
        curr_target = full_target_ids[:k]
        actual_k = len(curr_target)
        if actual_k == 0:
            continue

        # Construct masked sequence: prefix + [MASK]*actual_k + suffix
        input_ids = list(prefix_ids) + [mask_token_id] * actual_k + list(suffix_ids)
        mask_start_idx = len(prefix_ids)

        predicted_tokens = []

        # Model forward inference
        if backend == "mlx":
            import mlx.core as mx
            x = mx.array([input_ids], dtype=mx.int32)
            logits = model(x, mask_override=False)
            # Iterate over the masked positions
            for pos in range(mask_start_idx, mask_start_idx + actual_k):
                pos_logits = np.array(logits[0, pos].astype(mx.float32))
                # Explicitly prevent predicting the [MASK] token itself
                pos_logits[mask_token_id] = -1e9
                pred_tok = int(np.argmax(pos_logits))
                predicted_tokens.append(pred_tok)
        else:
            import torch
            x = torch.tensor([input_ids], dtype=torch.long)
            with torch.no_grad():
                logits = model(x, mask_override=False)
            for pos in range(mask_start_idx, mask_start_idx + actual_k):
                pos_logits = logits[0, pos].detach().cpu().numpy()
                pos_logits[mask_token_id] = -1e9
                pred_tok = int(np.argmax(pos_logits))
                predicted_tokens.append(pred_tok)

        # Compute cheat metrics
        metrics = compute_boundary_copy_metrics(
            predicted_tokens=predicted_tokens,
            prefix_tokens=prefix_ids,
            suffix_tokens=suffix_ids,
            target_tokens=curr_target
        )

        span_results[f"span_{k}"] = {
            "requested_k": k,
            "actual_k": actual_k,
            "exact_match": metrics["is_exact_match"],
            "token_accuracy": metrics["token_accuracy"],
            "copied_suffix_first": metrics["copied_suffix_first"],
            "copied_prefix_last": metrics["copied_prefix_last"],
        }

    return span_results


def summarize_anticheat_suite(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregates anti-cheat span evaluations into a global summary report.
    Computes suffix_copy_rate, prefix_copy_rate, and exact match across span lengths.
    """
    total_samples = len(results)
    if total_samples == 0:
        return {}

    summary = {}
    all_spans = set()
    for r in results:
        all_spans.update(r.keys())

    for span_key in sorted(all_spans):
        span_items = [r[span_key] for r in results if span_key in r]
        count = len(span_items)
        if count == 0:
            continue

        exact_matches = sum(1 for it in span_items if it["exact_match"])
        suffix_copies = sum(1 for it in span_items if it["copied_suffix_first"])
        prefix_copies = sum(1 for it in span_items if it["copied_prefix_last"])
        mean_tok_acc = float(np.mean([it["token_accuracy"] for it in span_items]))

        summary[span_key] = {
            "count": count,
            "exact_match_pct": round((exact_matches / count) * 100.0, 2),
            "token_accuracy_pct": round(mean_tok_acc * 100.0, 2),
            "suffix_copy_rate_pct": round((suffix_copies / count) * 100.0, 2),
            "prefix_copy_rate_pct": round((prefix_copies / count) * 100.0, 2),
        }

    # Flag potential cheating: high accuracy on span 1 with high suffix copy, dropping on span 4,
    # OR degenerate copying where suffix copy is high (>15%) while semantic accuracy is near-zero.
    span_1 = summary.get("span_1", {})
    span_4 = summary.get("span_4", {})
    is_suspect_cheater = False
    cheat_mode = None
    if span_1 and span_4:
        s1_copy = span_1.get("suffix_copy_rate_pct", 0)
        s1_acc = span_1.get("exact_match_pct", 0)
        s4_acc = span_4.get("exact_match_pct", 0)
        
        if s1_copy > 15.0:
            # Mode A: Classical boundary cheat (high span 1 match that plummets on span 4)
            if s1_acc > (s4_acc * 3.0) and s1_acc > 5.0:
                is_suspect_cheater = True
                cheat_mode = "boundary_cheat"
            # Mode B: Degenerate copying (high copy rate with near-zero semantic reasoning)
            elif s1_acc < 10.0:
                is_suspect_cheater = True
                cheat_mode = "degenerate_copy"

    return {
        "total_probes_evaluated": total_samples,
        "span_breakdown": summary,
        "is_suspect_cheater": is_suspect_cheater,
        "cheat_mode": cheat_mode,
    }
