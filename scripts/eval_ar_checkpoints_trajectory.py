"""
Evaluates validation cross-entropy across all intermediate checkpoints of 50M AR 2B:
checkpoints: 1017, 2034, 3051, 4068, 5085, 6102, 7119, 8136, 9153, 10170, final.
Downloads from Hugging Face: Kazenowoko/telos-corosred-50m-ar
"""
import os
import sys
import json
import time
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from telos.eval.runner import load_model_from_checkpoint
from scripts.eval_50m_head_to_head import evaluate_dataset_validation

REPO_ID = "Kazenowoko/telos-corosred-50m-ar"
STEPS = [1017, 2034, 3051, 4068, 5085, 6102, 7119, 8136, 9153, 10170, "final"]

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load validation data (last 200 sequences = 102,400 tokens)
    dataset_path = "data/python_corpus_2.5b.bin"
    raw_data = np.memmap(dataset_path, dtype=np.uint16, mode="r")
    seq_len = 512
    total_seqs = len(raw_data) // seq_len
    val_matrix = np.array([raw_data[idx * seq_len : (idx + 1) * seq_len] for idx in range(total_seqs - 200, total_seqs)])
    print(f"Loaded {len(val_matrix)} held-out validation sequences ({len(val_matrix) * seq_len:,} tokens)")

    results = []
    output_file = Path("logs/eval_ar_50m_trajectory.json")
    output_file.parent.mkdir(exist_ok=True)

    # If partial results exist, load them to resume
    if output_file.exists():
        try:
            with open(output_file, "r") as f:
                results = json.load(f)
            evaluated_steps = {r["step"] for r in results}
            print(f"Resuming from existing results. Already evaluated steps: {evaluated_steps}")
        except Exception:
            results = []

    evaluated_steps = {r["step"] for r in results}

    for step in STEPS:
        if step in evaluated_steps:
            print(f"Step {step} already evaluated, skipping.")
            continue

        filename = f"checkpoint_step_{step}.pt" if step != "final" else "checkpoint_final.pt"
        print(f"\n--- Processing Step {step} ({filename}) ---")

        # Download checkpoint from HF
        t0 = time.time()
        print(f"Downloading {filename} from {REPO_ID}...")
        try:
            ckpt_path = hf_hub_download(repo_id=REPO_ID, filename=filename)
            print(f"✓ Downloaded in {time.time() - t0:.1f}s to {ckpt_path}")
        except Exception as e:
            print(f"✗ Failed to download {filename}: {e}")
            continue

        # Load model and evaluate
        t_load = time.time()
        try:
            model, backend, vocab_size = load_model_from_checkpoint(ckpt_path)
            model.to(device)
            model.eval()
            print(f"✓ Model loaded in {time.time() - t_load:.2f}s")
        except Exception as e:
            print(f"✗ Failed to load model from {ckpt_path}: {e}")
            continue

        t_eval = time.time()
        eval_metrics = evaluate_dataset_validation(
            model,
            val_matrix,
            batch_size=16,
            num_batches=10,
            is_causal=True,
            device=device
        )
        t_elapsed = time.time() - t_eval

        # Approximate tokens trained
        step_num = 10173 if step == "final" else int(step)
        tokens_trained = step_num * 384 * 512

        record = {
            "step": step,
            "step_num": step_num,
            "tokens_trained": tokens_trained,
            "tokens_trained_str": f"{tokens_trained / 1e9:.2f}B",
            "val_loss": eval_metrics["val_loss"],
            "perplexity": eval_metrics["perplexity"],
            "top1_acc": eval_metrics["top1_acc"],
            "top5_acc": eval_metrics["top5_acc"],
            "eval_time_sec": round(t_elapsed, 2)
        }
        results.append(record)
        print(f"✓ Result for Step {step} ({record['tokens_trained_str']}): Val Loss: {record['val_loss']:.4f} | PPL: {record['perplexity']:.2f} | Top-1: {record['top1_acc']:.2f}% | Top-5: {record['top5_acc']:.2f}%")

        # Save incremental results
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        # Free GPU memory
        del model
        if device == "mps":
            torch.mps.empty_cache()

    print("\n" + "=" * 80)
    print("  50M AR BASELINE VALIDATION TRAJECTORY (0 to 2.0B TOKENS)")
    print("=" * 80)
    print(f"{'Step':<10} | {'Tokens':<10} | {'Val CE':<10} | {'Val PPL':<10} | {'Top-1':<10} | {'Top-5':<10}")
    print("-" * 80)
    for r in sorted(results, key=lambda x: x["step_num"]):
        print(f"{str(r['step']):<10} | {r['tokens_trained_str']:<10} | {r['val_loss']:<10.4f} | {r['perplexity']:<10.2f} | {r['top1_acc']:>6.2f}%   | {r['top5_acc']:>6.2f}%")
    print("=" * 80)

if __name__ == "__main__":
    main()
