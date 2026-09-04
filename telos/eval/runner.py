"""
High-End Evaluation Engine for Télos Models.
Supports Contextual Probes Benchmark (100 probes across 8 categories),
Qualitative Code Sampling, and Validation Perplexity computation.
Compatible with both MLX (.safetensors) and PyTorch (.pt) checkpoints.
"""

import os
import sys
import json
import time
import math
import argparse
from pathlib import Path
import numpy as np

from .probes import PROBE_SUITE_100
from telos.data.tokenizer import load_tokenizer
from telos.models import MLXTelosTransformer, TelosTransformer, TelosConfig


def load_model_from_checkpoint(checkpoint_path: str | Path, config: dict | None = None):
    """Loads an MLX or PyTorch model along with its tokenizer from a checkpoint path."""
    cp = Path(checkpoint_path)
    if cp.is_dir():
        mlx_file = cp / "model.safetensors"
        pt_file = cp / "checkpoint_final.pt"
        if not pt_file.exists():
            pt_files = sorted(cp.glob("checkpoint_step_*.pt"))
            pt_file = pt_files[-1] if pt_files else pt_file

        if mlx_file.exists():
            cp = mlx_file
        elif pt_file.exists():
            cp = pt_file
        else:
            raise FileNotFoundError(f"No checkpoint file found in directory {checkpoint_path}")

    # Load config JSON if present
    cfg_file = cp.parent / "config.json"
    m_cfg = {}
    if cfg_file.exists():
        with open(cfg_file, "r") as f:
            m_cfg = json.load(f)
    elif config:
        m_cfg = config.get("model", config)

    vocab_size = m_cfg.get("vocab_size", 8192)
    d_model = m_cfg.get("d_model", 512)
    n_layers = m_cfg.get("n_layers", 12)
    n_heads = m_cfg.get("n_heads", 16)
    n_kv_heads = m_cfg.get("n_kv_heads", n_heads)

    if cp.suffix == ".safetensors":
        import mlx.core as mx
        model = MLXTelosTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            tied_embeddings=True
        )
        model.load_weights(str(cp))
        return model, "mlx", vocab_size

    elif cp.suffix in [".pt", ".bin"]:
        import torch
        tc = TelosConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            tied_embeddings=True
        )
        model = TelosTransformer(tc)
        state = torch.load(str(cp), map_location="cpu")
        sd = state.get("model_state_dict", state)
        model.load_state_dict(sd, strict=False)
        model.eval()
        return model, "pytorch", vocab_size

    else:
        raise ValueError(f"Unrecognized checkpoint format: {cp.name}")


def evaluate_probes(model, tokenizer, backend: str, mask_token_id: int = 1) -> dict:
    """Executes the 100 contextual probes benchmark and computes publication metrics."""
    results = []
    category_stats = {}

    for probe in PROBE_SUITE_100:
        cat = probe["category"]
        prompt = probe["prompt"]
        target_str = probe["target"]

        if cat not in category_stats:
            category_stats[cat] = {"count": 0, "top1": 0, "top5": 0, "ce": [], "rank": []}

        # Tokenize prompt and target
        p_ids = tokenizer.encode(prompt).ids
        target_ids = tokenizer.encode(target_str).ids
        if not target_ids:
            target_ids = [0]
        target_tok = target_ids[0]

        # Prepare input with [MASK] at prediction position
        input_ids = list(p_ids) + [mask_token_id]
        mask_idx = len(input_ids) - 1

        if backend == "mlx":
            import mlx.core as mx
            x = mx.array([input_ids], dtype=mx.int32)
            logits = model(x)
            logits_pos = np.array(logits[0, mask_idx])
        else:
            import torch
            x = torch.tensor([input_ids], dtype=torch.long)
            with torch.no_grad():
                logits = model(x)
            logits_pos = logits[0, mask_idx].detach().cpu().numpy()

        # Compute softmax probabilities & target rank
        shifted = logits_pos - np.max(logits_pos)
        probs = np.exp(shifted) / np.sum(np.exp(shifted))
        
        target_prob = max(float(probs[target_tok]), 1e-12)
        target_ce = -math.log(target_prob)

        # Rank of target token (1-based)
        sorted_indices = np.argsort(logits_pos)[::-1]
        rank = int(np.where(sorted_indices == target_tok)[0][0]) + 1
        is_top1 = (rank == 1)
        is_top5 = (rank <= 5)

        category_stats[cat]["count"] += 1
        if is_top1:
            category_stats[cat]["top1"] += 1
        if is_top5:
            category_stats[cat]["top5"] += 1
        category_stats[cat]["ce"].append(target_ce)
        category_stats[cat]["rank"].append(rank)

        results.append({
            "category": cat,
            "prompt": prompt,
            "target": target_str,
            "rank": rank,
            "target_ce": target_ce,
            "top1": is_top1,
            "top5": is_top5,
        })

    # Summary calculations
    total_count = len(results)
    overall_top1 = sum(r["top1"] for r in results) / total_count * 100.0
    overall_top5 = sum(r["top5"] for r in results) / total_count * 100.0
    overall_ce = float(np.mean([r["target_ce"] for r in results]))
    overall_rank = float(np.mean([r["rank"] for r in results]))

    print("\n" + "=" * 78)
    print("  TÉLOS CONTEXTUAL PROBES BENCHMARK REPORT (100 PROBES)")
    print("=" * 78)
    print(f"  {'Category':<24} | {'Count':<5} | {'Top-1 (%)':<9} | {'Top-5 (%)':<9} | {'Avg Rank':<8} | {'Avg CE':<6}")
    print("-" * 78)

    cat_breakdown = {}
    for cat, s in category_stats.items():
        cnt = s["count"]
        top1_pct = (s["top1"] / cnt) * 100.0 if cnt else 0.0
        top5_pct = (s["top5"] / cnt) * 100.0 if cnt else 0.0
        avg_r = float(np.mean(s["rank"])) if cnt else 0.0
        avg_ce = float(np.mean(s["ce"])) if cnt else 0.0
        cat_breakdown[cat] = {"top1_pct": top1_pct, "top5_pct": top5_pct, "avg_rank": avg_r, "avg_ce": avg_ce}
        print(f"  {cat:<24} | {cnt:<5d} | {top1_pct:>8.1f}% | {top5_pct:>8.1f}% | {avg_r:>8.1f} | {avg_ce:>6.2f}")

    print("-" * 78)
    print(f"  {'OVERALL SUMMARY':<24} | {total_count:<5d} | {overall_top1:>8.1f}% | {overall_top5:>8.1f}% | {overall_rank:>8.1f} | {overall_ce:>6.2f}")
    print("=" * 78 + "\n")

    report_payload = {
        "overall": {
            "top1_acc_pct": overall_top1,
            "top5_acc_pct": overall_top5,
            "mean_rank": overall_rank,
            "mean_ce": overall_ce,
            "total_probes": total_count,
        },
        "categories": cat_breakdown,
        "probes": results,
    }

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    report_file = log_dir / f"eval_probes_{int(time.time())}.json"
    with open(report_file, "w") as f:
        json.dump(report_payload, f, indent=2)
    print(f"  Saved detailed evaluation probe metrics to {report_file}\n")

    return report_payload


def evaluate_sample(model, tokenizer, backend: str, prompts: list[str] | None = None, steps: int = 16):
    """Generates code completions for sample evaluation prompts."""
    default_prompts = [
        "def fibonacci(n: int) -> int:\n    if n <= 1:\n        return",
        "def quicksort(arr: list[int]) -> list[int]:\n    if len(arr) <= 1:\n",
        "class Node:\n    def __init__(self, value):\n        self.value = value\n        self.",
    ]
    prompts = prompts or default_prompts

    print("\n" + "=" * 76)
    print("  TÉLOS QUALITATIVE CODE COMPLETION EVALUATION")
    print("=" * 76)

    for i, p in enumerate(prompts, 1):
        print(f"\n--- [Prompt {i}] ---\n{p}")
        p_ids = tokenizer.encode(p).ids
        # Autocomplete next tokens
        curr_ids = list(p_ids)
        for _ in range(steps):
            if backend == "mlx":
                import mlx.core as mx
                x = mx.array([curr_ids], dtype=mx.int32)
                logits = model(x)
                next_tok = int(np.argmax(np.array(logits[0, -1])))
            else:
                import torch
                x = torch.tensor([curr_ids], dtype=torch.long)
                with torch.no_grad():
                    logits = model(x)
                next_tok = int(torch.argmax(logits[0, -1]).item())

            curr_ids.append(next_tok)
            if next_tok in [0, 3]:  # EOS / PAD
                break

        completion = tokenizer.decode(curr_ids)
        print(f"--- [Completion] ---\n{completion}\n")
    print("=" * 76 + "\n")


def evaluate(
    checkpoint: str | Path,
    mode: str = "probes",
    tokenizer_path: str | Path | None = None,
    prompts: list[str] | None = None,
    **kwargs
) -> dict:
    """Master programmatic entrypoint for Télos evaluation."""
    model, backend, vocab_size = load_model_from_checkpoint(checkpoint)
    tok = load_tokenizer(str(tokenizer_path or "configs/shared/tokenizer_0.json"))

    if mode == "probes":
        return evaluate_probes(model, tok, backend)
    elif mode == "sample":
        evaluate_sample(model, tok, backend, prompts=prompts)
        return {"mode": "sample", "status": "completed"}
    else:
        raise ValueError(f"Unknown evaluation mode: {mode}. Choose 'probes' or 'sample'.")


def main():
    parser = argparse.ArgumentParser(description="Télos Model Evaluation Suite")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file (.safetensors, .pt) or directory")
    parser.add_argument("--mode", type=str, default="probes", choices=["probes", "sample"], help="Evaluation mode")
    parser.add_argument("--tokenizer", type=str, default=None, help="Path to BPE tokenizer JSON")
    args = parser.parse_args()

    evaluate(checkpoint=args.checkpoint, mode=args.mode, tokenizer_path=args.tokenizer)


if __name__ == "__main__":
    main()
