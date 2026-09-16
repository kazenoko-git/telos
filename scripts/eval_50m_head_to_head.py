"""
Comprehensive Head-to-Head Evaluation Suite for 50M Models:
1. 50M AR Baseline (300M tokens)
2. 50M CoroSRED Phase A (1.0B tokens, Causal + Reliability Head)
3. 50M CoroSRED Phase B (2.0B tokens total: 1.0B Phase A + 1.0B Self-Conditioned Denoiser)
4. 50M AR Baseline (2.0B tokens, Compute-Matched Pure Autoregressive)

Evaluates each model under both:
- Pure Causal Next-Token Prediction (mask_override=True)
- Bidirectional Masked Denoising (mask_override=False, 15% random mask, fixed seed)
- 100 Contextual Probes (both Causal prefix and Bidirectional [MASK] infilling)
- Reliability Head Calibration (for Phase A & Phase B)
"""

import os
import math
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from telos.eval.runner import load_model_from_checkpoint
from telos.eval.probes import PROBE_SUITE_100
from telos.data.tokenizer import load_tokenizer


def evaluate_probes_mode(model, tok, mode: str = "causal", mask_token_id: int = 1, device: str = "cpu") -> dict:
    """
    Evaluates the 100 contextual probes in either:
    - 'causal': prefix only, next-token prediction at last token position (mask_override=True)
    - 'bidirectional': prefix + [MASK], denoise at mask position (mask_override=False)
    """
    category_stats = {}
    all_ranks = []
    all_ces = []
    top1_count = 0
    top5_count = 0

    for probe in PROBE_SUITE_100:
        cat = probe["category"]
        prompt = probe["prompt"]
        target_str = probe["target"]

        if cat not in category_stats:
            category_stats[cat] = {"count": 0, "top1": 0, "top5": 0, "ce": [], "rank": []}

        p_ids = tok.encode(prompt).ids
        target_ids = tok.encode(target_str).ids
        target_tok = target_ids[0] if target_ids else 0

        if mode == "causal":
            # Autoregressive mode: predict next token at last prefix position
            input_ids = list(p_ids)
            eval_idx = len(input_ids) - 1
            x = torch.tensor([input_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                logits = model(x, mask_override=True)
        else:
            # Bidirectional mode: denoise [MASK] token at target position
            input_ids = list(p_ids) + [mask_token_id]
            eval_idx = len(input_ids) - 1
            x = torch.tensor([input_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                logits = model(x, mask_override=False)

        logits_pos = logits[0, eval_idx].detach().cpu().numpy()

        # Numerically stable softmax
        shifted = logits_pos - np.max(logits_pos)
        probs = np.exp(shifted) / np.sum(np.exp(shifted))

        target_prob = max(float(probs[target_tok]), 1e-12)
        target_ce = -math.log(target_prob)

        sorted_indices = np.argsort(logits_pos)[::-1]
        rank = int(np.where(sorted_indices == target_tok)[0][0]) + 1
        is_top1 = (rank == 1)
        is_top5 = (rank <= 5)

        category_stats[cat]["count"] += 1
        if is_top1:
            category_stats[cat]["top1"] += 1
            top1_count += 1
        if is_top5:
            category_stats[cat]["top5"] += 1
            top5_count += 1
        category_stats[cat]["ce"].append(target_ce)
        category_stats[cat]["rank"].append(rank)
        all_ranks.append(rank)
        all_ces.append(target_ce)

    summary_by_cat = {}
    for cat, s in category_stats.items():
        cnt = s["count"]
        summary_by_cat[cat] = {
            "count": cnt,
            "top1_pct": round((s["top1"] / cnt) * 100.0, 1),
            "top5_pct": round((s["top5"] / cnt) * 100.0, 1),
            "avg_rank": round(float(np.mean(s["rank"])), 1),
            "avg_ce": round(float(np.mean(s["ce"])), 2),
        }

    total_probes = len(PROBE_SUITE_100)
    return {
        "top1_pct": round((top1_count / total_probes) * 100.0, 2),
        "top5_pct": round((top5_count / total_probes) * 100.0, 2),
        "avg_rank": round(float(np.mean(all_ranks)), 1),
        "avg_ce": round(float(np.mean(all_ces)), 2),
        "categories": summary_by_cat
    }


def evaluate_dataset_validation(
    model,
    dataset_matrix: np.ndarray,
    batch_size: int = 16,
    num_batches: int = 10,
    is_causal: bool = True,
    device: str = "cpu",
    seed: int = 42
) -> dict:
    """
    Computes validation cross-entropy, perplexity, and top-1/top-5 accuracies on held-out dataset.
    If is_causal=True: next-token prediction across entire sequence.
    If is_causal=False: 15% random mask denoising (deterministic across runs with fixed seed).
    """
    seq_len = dataset_matrix.shape[1]
    total_tokens = 0
    total_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0

    # Reliability calibration metrics (for Phase A / Phase B)
    has_reliability = getattr(model.config, "use_reliability_head", False)
    all_rel_scores = []
    high_rel_correct = 0
    high_rel_total = 0
    low_rel_correct = 0
    low_rel_total = 0

    # Deterministic generator for mask corruption reproducibility
    rng = torch.Generator(device="cpu").manual_seed(seed)

    with torch.no_grad():
        for b in range(num_batches):
            batch_slice = dataset_matrix[b * batch_size : (b + 1) * batch_size]
            x = torch.from_numpy(batch_slice.astype(np.int64)).to(device)

            if is_causal:
                if has_reliability:
                    logits, r_scores = model(x, mask_override=True, return_reliability=True)
                    all_rel_scores.extend(r_scores.detach().cpu().numpy().flatten().tolist())
                else:
                    logits = model(x, mask_override=True)
                    r_scores = None

                shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
                shift_targets = x[:, 1:].contiguous().view(-1)
                
                loss = F.cross_entropy(shift_logits, shift_targets, reduction="sum")
                n_tok = shift_targets.numel()

                # Top-1 & Top-5 accuracy
                _, pred_top5 = shift_logits.topk(5, dim=-1)
                pred_top1 = pred_top5[:, 0]
                correct_1 = (pred_top1 == shift_targets)
                correct_top1 += correct_1.sum().item()
                correct_5 = (pred_top5 == shift_targets.unsqueeze(-1)).any(dim=-1)
                correct_top5 += correct_5.sum().item()

                if r_scores is not None:
                    shift_r = r_scores[:, :-1].contiguous().view(-1)
                    high_mask = (shift_r >= 0.5)
                    low_mask = ~high_mask
                    high_rel_total += high_mask.sum().item()
                    high_rel_correct += correct_1[high_mask].sum().item()
                    low_rel_total += low_mask.sum().item()
                    low_rel_correct += correct_1[low_mask].sum().item()

            else:
                # Denoising loss on 15% random mask generated deterministically on CPU
                rand = torch.rand(x.shape, generator=rng, device="cpu").to(device)
                mask = (rand < 0.15)
                corrupted = torch.where(mask, torch.ones_like(x), x)

                if has_reliability:
                    logits, r_scores = model(corrupted, mask_override=False, return_reliability=True)
                    all_rel_scores.extend(r_scores.detach().cpu().numpy().flatten().tolist())
                else:
                    logits = model(corrupted, mask_override=False)
                    r_scores = None

                logits_flat = logits.view(-1, logits.size(-1))
                targets_flat = x.view(-1)
                mask_flat = mask.view(-1)

                masked_logits = logits_flat[mask_flat]
                masked_targets = targets_flat[mask_flat]

                loss = F.cross_entropy(masked_logits, masked_targets, reduction="sum")
                n_tok = mask_flat.sum().item()

                _, pred_top5 = masked_logits.topk(5, dim=-1)
                pred_top1 = pred_top5[:, 0]
                correct_1 = (pred_top1 == masked_targets)
                correct_top1 += correct_1.sum().item()
                correct_5 = (pred_top5 == masked_targets.unsqueeze(-1)).any(dim=-1)
                correct_top5 += correct_5.sum().item()

            total_loss += loss.item()
            total_tokens += n_tok

    avg_ce = total_loss / max(1, total_tokens)
    ppl = math.exp(min(avg_ce, 20.0))
    top1_pct = (correct_top1 / max(1, total_tokens)) * 100.0
    top5_pct = (correct_top5 / max(1, total_tokens)) * 100.0

    result = {
        "val_loss": round(avg_ce, 4),
        "perplexity": round(ppl, 2),
        "top1_acc": round(top1_pct, 2),
        "top5_acc": round(top5_pct, 2),
        "tokens_evaluated": total_tokens
    }

    if all_rel_scores:
        result["mean_reliability"] = round(float(np.mean(all_rel_scores)), 4)
        if high_rel_total > 0:
            result["high_rel_acc"] = round((high_rel_correct / high_rel_total) * 100.0, 2)
            result["high_rel_count"] = high_rel_total
        if low_rel_total > 0:
            result["low_rel_acc"] = round((low_rel_correct / low_rel_total) * 100.0, 2)
            result["low_rel_count"] = low_rel_total

    return result


def main():
    print("=" * 84)
    print("  TELOS 50M HEAD-TO-HEAD COMPREHENSIVE BENCHMARK")
    print("  Comparing: 50M AR 300M | 50M Phase A (1.0B) | 50M Phase B (2.0B) | 50M AR 2.0B")
    print("=" * 84)

    # Device selection: use MPS if available for acceleration, else CPU
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  Using hardware device: {device.upper()}")

    # 1. Load Validation Data (held-out tail of corpus)
    dataset_path = "data/python_corpus_2.5b.bin"
    raw_data = np.memmap(dataset_path, dtype=np.uint16, mode="r")
    seq_len = 512
    total_seqs = len(raw_data) // seq_len
    # Take last 200 sequences (102,400 tokens) strictly held out
    val_matrix = np.array([raw_data[idx * seq_len : (idx + 1) * seq_len] for idx in range(total_seqs - 200, total_seqs)])
    print(f"✓ Loaded {len(val_matrix)} held-out validation sequences ({len(val_matrix) * seq_len:,} tokens)")

    tok = load_tokenizer()

    models_to_eval = [
        {
            "id": "ar_300m",
            "name": "50M AR Baseline (300M tok)",
            "tokens": "300M",
            "path": "checkpoints/ar/50m_300m",
        },
        {
            "id": "phase_a_1b",
            "name": "50M CoroSRED Phase A (1.0B tok)",
            "tokens": "1.0B",
            "path": "checkpoints/corosred/50m/phase_a",
        },
        {
            "id": "phase_b_2b",
            "name": "50M CoroSRED Phase B (2.0B tok total)",
            "tokens": "2.0B (1B A + 1B B)",
            "path": "checkpoints/corosred/50m/phase_b",
        },
        {
            "id": "ar_2b",
            "name": "50M AR Pure Baseline (2.0B tok)",
            "tokens": "2.0B",
            "path": "checkpoints/ar/50m_2b",
        },
    ]

    all_results = {}

    for m_info in models_to_eval:
        name = m_info["name"]
        path = m_info["path"]
        print("\n" + "-" * 84)
        print(f"  EVALUATING: {name}")
        print(f"  Path: {path} | Total Tokens: {m_info['tokens']}")
        print("-" * 84)

        if not Path(path).exists():
            print(f"  [Error] Checkpoint path does not exist: {path}")
            continue

        model, backend, vocab_size = load_model_from_checkpoint(path)
        model.to(device)
        model.eval()

        # 1. Causal Next-Token Validation Loss
        print("  1. Evaluating Causal Next-Token Validation (Held-out)...")
        val_causal = evaluate_dataset_validation(model, val_matrix, batch_size=16, num_batches=10, is_causal=True, device=device)
        print(f"     ✓ Val CE: {val_causal['val_loss']} | PPL: {val_causal['perplexity']} | Top-1: {val_causal['top1_acc']}% | Top-5: {val_causal['top5_acc']}%")
        if "mean_reliability" in val_causal:
            print(f"     ✓ Reliability Head: Mean={val_causal['mean_reliability']} | High-Rel Acc={val_causal.get('high_rel_acc', 'N/A')}% | Low-Rel Acc={val_causal.get('low_rel_acc', 'N/A')}%")

        # 2. Bidirectional Masked Denoising Validation Loss (15% random mask, fixed seed)
        print("  2. Evaluating Bidirectional Denoising Validation (15% mask, seed=42)...")
        val_bidir = evaluate_dataset_validation(model, val_matrix, batch_size=16, num_batches=10, is_causal=False, device=device, seed=42)
        print(f"     ✓ Denoise CE: {val_bidir['val_loss']} | Denoise PPL: {val_bidir['perplexity']} | Denoise Top-1: {val_bidir['top1_acc']}% | Denoise Top-5: {val_bidir['top5_acc']}%")

        # 3. Contextual Probes - Causal Mode
        print("  3. Evaluating Contextual Probes Suite in Causal Mode (100 probes)...")
        probes_causal = evaluate_probes_mode(model, tok, mode="causal", device=device)
        print(f"     ✓ Causal Probes: Avg Rank: {probes_causal['avg_rank']} | Avg CE: {probes_causal['avg_ce']} | Top-1: {probes_causal['top1_pct']}% | Top-5: {probes_causal['top5_pct']}%")

        # 4. Contextual Probes - Bidirectional Mode
        print("  4. Evaluating Contextual Probes Suite in Bidirectional Mode (100 probes)...")
        probes_bidir = evaluate_probes_mode(model, tok, mode="bidirectional", device=device)
        print(f"     ✓ Bidir Probes:  Avg Rank: {probes_bidir['avg_rank']} | Avg CE: {probes_bidir['avg_ce']} | Top-1: {probes_bidir['top1_pct']}% | Top-5: {probes_bidir['top5_pct']}%")

        all_results[name] = {
            "tokens": m_info["tokens"],
            "path": path,
            "val_causal": val_causal,
            "val_bidirectional": val_bidir,
            "probes_causal": probes_causal,
            "probes_bidirectional": probes_bidir,
        }

    # Save complete JSON report
    out_dir = Path("logs")
    out_dir.mkdir(exist_ok=True)
    report_file = out_dir / f"eval_50m_comparison_2b_{int(time.time())}.json"
    with open(report_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # -------------------------------------------------------------------------
    # PRINT SUMMARY BENCHMARK TABLES
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("  TABLE 1: CAUSAL NEXT-TOKEN PREDICTION (PURE AR MODE: mask_override=True)")
    print("=" * 90)
    print(f"{'Model Architecture':<38} | {'Tokens':<12} | {'Val CE':<8} | {'Val PPL':<8} | {'Top-1':<7} | {'Top-5':<7}")
    print("-" * 90)
    for name, r in all_results.items():
        v = r["val_causal"]
        print(f"{name:<38} | {r['tokens']:<12} | {v['val_loss']:<8.4f} | {v['perplexity']:<8.2f} | {v['top1_acc']:>5.2f}% | {v['top5_acc']:>5.2f}%")
    print("=" * 90)

    print("\n" + "=" * 90)
    print("  TABLE 2: BIDIRECTIONAL MASKED DENOISING (15% RANDOM MASK: mask_override=False)")
    print("=" * 90)
    print(f"{'Model Architecture':<38} | {'Tokens':<12} | {'Den. CE':<8} | {'Den. PPL':<8} | {'Top-1':<7} | {'Top-5':<7}")
    print("-" * 90)
    for name, r in all_results.items():
        v = r["val_bidirectional"]
        print(f"{name:<38} | {r['tokens']:<12} | {v['val_loss']:<8.4f} | {v['perplexity']:<8.2f} | {v['top1_acc']:>5.2f}% | {v['top5_acc']:>5.2f}%")
    print("=" * 90)

    print("\n" + "=" * 90)
    print("  TABLE 3: CONTEXTUAL PROBES BENCHMARK (100 PROBES)")
    print("=" * 90)
    print(f"{'Model Architecture':<38} | {'Causal Rank':<12} | {'Causal CE':<10} | {'Bidir Rank':<12} | {'Bidir CE':<10}")
    print("-" * 90)
    for name, r in all_results.items():
        pc = r["probes_causal"]
        pb = r["probes_bidirectional"]
        print(f"{name:<38} | {pc['avg_rank']:<12.1f} | {pc['avg_ce']:<10.2f} | {pb['avg_rank']:<12.1f} | {pb['avg_ce']:<10.2f}")
    print("=" * 90)

    print(f"\n✓ Saved complete evaluation results to {report_file}\n")


if __name__ == "__main__":
    main()
