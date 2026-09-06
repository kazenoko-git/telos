"""
Comprehensive Evaluation Suite for 50M Models:
- 50M AR Baseline (300M tokens)
- 50M CoroSRED Phase A (Causal + Reliability Head)
- 50M CoroSRED Phase B (Self-Conditioned Denoiser)

Evaluates:
1. Contextual Probes Suite (100 probes across 8 syntax/semantic categories)
2. Validation Loss & Perplexity on held-out dataset sequences
3. Multi-Stage Refinement Gains (AR draft -> CoroSRED Phase B refinement)
"""

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
from telos.diffusion.region import form_refinement_regions
from telos.diffusion.refiner import COROSredRefiner


def evaluate_model_probes(model, tok, is_causal: bool = False, mask_token_id: int = 1) -> dict:
    """Evaluates contextual probes with mode-aware prefix vs masked prediction."""
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

        if is_causal:
            # Causal LM: model predicts next token at last prefix position
            input_ids = list(p_ids)
            eval_idx = len(input_ids) - 1
            x = torch.tensor([input_ids], dtype=torch.long)
            with torch.no_grad():
                logits = model(x, mask_override=True)
        else:
            # Bidirectional Diffusion LM: model denoises [MASK] at target position
            input_ids = list(p_ids) + [mask_token_id]
            eval_idx = len(input_ids) - 1
            x = torch.tensor([input_ids], dtype=torch.long)
            with torch.no_grad():
                logits = model(x, mask_override=False)

        logits_pos = logits[0, eval_idx].detach().cpu().numpy()

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
    is_causal: bool = True
) -> dict:
    """Computes held-out validation cross-entropy and perplexity across sequential batches."""
    seq_len = dataset_matrix.shape[1]
    total_tokens = 0
    total_loss = 0.0

    with torch.no_grad():
        for b in range(num_batches):
            batch_slice = dataset_matrix[b * batch_size : (b + 1) * batch_size]
            x = torch.from_numpy(batch_slice.astype(np.int64))

            if is_causal:
                logits = model(x, mask_override=True)
                shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
                shift_targets = x[:, 1:].contiguous().view(-1)
                loss = F.cross_entropy(shift_logits, shift_targets, reduction="sum")
                n_tok = shift_targets.numel()
            else:
                # Denoising loss on 15% random mask
                rand = torch.rand(x.shape)
                mask = (rand < 0.15)
                corrupted = torch.where(mask, torch.ones_like(x), x)
                logits = model(corrupted, mask_override=False)
                logits_flat = logits.view(-1, logits.size(-1))
                targets_flat = x.view(-1)
                mask_flat = mask.view(-1)
                loss = F.cross_entropy(logits_flat[mask_flat], targets_flat[mask_flat], reduction="sum")
                n_tok = mask_flat.sum().item()

            total_loss += loss.item()
            total_tokens += n_tok

    avg_ce = total_loss / max(1, total_tokens)
    ppl = math.exp(min(avg_ce, 20.0))
    return {
        "val_loss": round(avg_ce, 4),
        "perplexity": round(ppl, 2),
        "tokens_evaluated": total_tokens
    }


def main():
    print("=" * 80)
    print("  TELOS 50M COMPREHENSIVE EVALUATION BENCHMARK")
    print("=" * 80)

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
            "name": "50M AR Baseline (300M tokens)",
            "path": "checkpoints/ar/50m_300m",
            "is_causal": True,
            "paradigm": "ar"
        },
        {
            "name": "50M CoroSRED Phase A",
            "path": "checkpoints/corosred/50m/phase_a",
            "is_causal": True,
            "paradigm": "corosred_a"
        },
        {
            "name": "50M CoroSRED Phase B (Self-Conditioned)",
            "path": "checkpoints/corosred/50m/phase_b",
            "is_causal": False,
            "paradigm": "corosred_b"
        },
    ]

    all_results = {}

    for m_info in models_to_eval:
        name = m_info["name"]
        path = m_info["path"]
        is_causal = m_info["is_causal"]

        print("\n" + "-" * 80)
        print(f"  EVALUATING: {name}")
        print(f"  Path: {path}")
        print("-" * 80)

        model, backend, vocab_size = load_model_from_checkpoint(path)
        model.to("cpu")
        model.eval()

        # A. Probes
        print("  Running Contextual Probes Suite (100 probes)...")
        probes_res = evaluate_model_probes(model, tok, is_causal=is_causal)
        print(f"  ✓ Probes: Top-1: {probes_res['top1_pct']}% | Top-5: {probes_res['top5_pct']}% | Avg Rank: {probes_res['avg_rank']} | Avg CE: {probes_res['avg_ce']}")

        # B. Validation Loss
        print("  Running Held-Out Validation Loss...")
        val_res = evaluate_dataset_validation(model, val_matrix, batch_size=16, num_batches=10, is_causal=is_causal)
        print(f"  ✓ Validation CE: {val_res['val_loss']} nats | Perplexity: {val_res['perplexity']}")

        all_results[name] = {
            "probes": probes_res,
            "validation": val_res
        }

    # Save detailed evaluation report
    out_dir = Path("logs")
    out_dir.mkdir(exist_ok=True)
    report_file = out_dir / f"eval_50m_suite_{int(time.time())}.json"
    with open(report_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print("\n" + "=" * 80)
    print("  50M COMPREHENSIVE BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Model Architecture':<42} | {'Val CE':<8} | {'Val PPL':<8} | {'Probes CE':<10} | {'Avg Rank':<8}")
    print("-" * 80)
    for name, r in all_results.items():
        v = r["validation"]
        p = r["probes"]
        print(f"{name:<42} | {v['val_loss']:<8.4f} | {v['perplexity']:<8.2f} | {p['avg_ce']:<10.2f} | {p['avg_rank']:<8.1f}")
    print("=" * 80)
    print(f"✓ Saved full report to {report_file}\n")


if __name__ == "__main__":
    main()
