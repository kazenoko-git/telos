"""Unified Master Evaluation CLI for télos MDLM.

Evaluates:
1. Overall Model Loss & Unweighted Cross-Entropy (CE)
2. Category Breakdown CE (Newlines, Indentation, Punctuation, Keywords, Identifiers, Numbers)
3. Contextual Probes & Target Token Ranks (e.g. `def add(a, b): return a + [MASK]`)
4. Top-k Token Probability Distribution Analysis

Usage:
  python scripts/eval.py --checkpoint checkpoints/phase_c_tpu_125m/checkpoint_tpu_125M_final_step_238.pt --mode full
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import torch
import torch.nn.functional as F
import numpy as np

from telos.hub.inference import TelosModel
from telos.diffusion.loss import mdlm_loss
from telos.diffusion.forward_process import apply_masking


PROBE_SUITE = [
    {"prompt": "return a +", "target": "Ġb", "category": "Identifier"},
    {"prompt": "if x ==", "target": "ĠNone", "category": "Literal"},
    {"prompt": "x = 10\nprint(", "target": "x", "category": "Identifier"},
    {"prompt": "for i in", "target": "Ġrange", "category": "Keyword"},
    {"prompt": "def __init__(self,", "target": "Ġname", "category": "Identifier"},
    {"prompt": "import", "target": "Ġos", "category": "Module"},
]

CATEGORY_PATTERNS = {
    "Newlines": [1, 2],  # '\n', '\r'
    "Indentation": [198, 263],  # '    ', '  '
    "Punctuation": [7, 8, 9, 10, 11, 12, 13, 14, 15, 16],  # ':', '(', ')', '=', etc.
    "Keywords": [30, 31, 32, 33, 34, 35, 36],  # 'def', 'return', 'if', 'else', 'import', 'for', 'class'
}


def run_overall_and_category_ce(model_obj: TelosModel, checkpoint_path: str):
    """Evaluates Overall Loss, Unweighted CE, and Category-based CE Breakdown."""
    print("\n" + "=" * 80)
    print(f"EVALUATING MODEL LOSS & CATEGORY CE: {checkpoint_path}")
    print("=" * 80)

    model = model_obj.model
    device = model_obj.device
    tokenizer = model_obj.tokenizer
    mask_token_id = tokenizer.token_to_id("[MASK]") or 4

    dataset_path = Path("data/python_corpus_mac.bin")
    if not dataset_path.exists():
        dataset_path = Path("data/python_corpus_1.7b.bin")

    if not dataset_path.exists():
        print("Warning: Corpus file not found for evaluation.")
        return

    seq_len = model.config.max_seq_len
    dataset = np.memmap(dataset_path, dtype=np.uint16, mode="r")
    num_samples = dataset.shape[0] // seq_len
    dataset = dataset[:num_samples * seq_len].reshape(num_samples, seq_len)

    # Sample eval batch of 32 sequences
    eval_batch = dataset[:32]
    token_ids = torch.from_numpy(np.array(eval_batch, dtype=np.int64)).to(device)

    masked_ids, mask_positions, t_values = apply_masking(
        input_ids=token_ids,
        mask_token_id=mask_token_id
    )

    model.eval()
    with torch.no_grad():
        logits = model(masked_ids)
        loss, metrics = mdlm_loss(logits=logits, targets=token_ids, mask_positions=mask_positions, t_values=t_values)

        unweighted_ce = metrics.get("ce_loss", loss.item()) if isinstance(metrics, dict) else metrics
        print(f"Overall ELBO Loss : {loss.item():.4f}")
        print(f"Unweighted CE     : {unweighted_ce:.4f}")

        # Category Breakdown CE
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = token_ids.view(-1)
        mask_flat = mask_positions.view(-1)

        masked_logits = logits_flat[mask_flat]
        masked_targets = targets_flat[mask_flat]

        if masked_logits.size(0) > 0:
            per_token_ce = F.cross_entropy(masked_logits, masked_targets, reduction="none")

            print("\nCategory Breakdown CE:")
            print("-" * 50)
            for cat_name, token_list in CATEGORY_PATTERNS.items():
                cat_mask = torch.isin(masked_targets, torch.tensor(token_list, device=device))
                if cat_mask.any():
                    cat_ce = per_token_ce[cat_mask].mean().item()
                    print(f"  {cat_name:<16} : {cat_ce:.4f} ({cat_mask.sum().item()} tokens)")
                else:
                    print(f"  {cat_name:<16} : N/A")


def run_contextual_probes(model_obj: TelosModel, checkpoint_path: str):
    """Evaluates contextual probes, target token ranks, and top-5 predicted tokens."""
    print("\n" + "=" * 80)
    print(f"EVALUATING CONTEXTUAL PROBES: {checkpoint_path}")
    print("=" * 80)

    model = model_obj.model
    tokenizer = model_obj.tokenizer
    device = model_obj.device
    mask_token_id = tokenizer.token_to_id("[MASK]") or 4

    model.eval()

    print(f"{'PROMPT':<25} | {'TARGET':<10} | {'RANK':<6} | {'TOP 5 PREDICTED TOKENS'}")
    print("-" * 80)

    for probe in PROBE_SUITE:
        prompt = probe["prompt"]
        target_str = probe["target"]

        # Encode prompt + [MASK]
        prompt_enc = tokenizer.encode(prompt)
        prompt_ids = prompt_enc.ids

        seq_ids = prompt_ids + [mask_token_id]
        input_tensor = torch.tensor([seq_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(input_tensor)
            mask_pos_logits = logits[0, -1].clone()
            mask_pos_logits[mask_token_id] = -float("inf")
            probs = F.softmax(mask_pos_logits, dim=-1)

        # Target token ID
        target_token_id = tokenizer.token_to_id(target_str)

        if target_token_id is not None:
            sorted_indices = torch.argsort(probs, descending=True)
            rank = (sorted_indices == target_token_id).nonzero(as_tuple=True)[0].item() + 1
            rank_str = f"#{rank}"
        else:
            rank_str = "N/A"

        # Top 5 tokens
        top5_probs, top5_ids = torch.topk(probs, k=5)
        top5_tokens = [tokenizer.id_to_token(tid.item()).replace("Ġ", " ") for tid in top5_ids]

        top5_formatted = ", ".join([f"'{t}' ({p:.2f})" for t, p in zip(top5_tokens, top5_probs.tolist())])
        print(f"{prompt:<25} | {target_str:<10} | {rank_str:<6} | {top5_formatted}")


def main():
    parser = argparse.ArgumentParser(description="Master Evaluation CLI for télos MDLM")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--mode", type=str, choices=["category-ce", "probes", "full"], default="full", help="Eval mode")
    args = parser.parse_args()

    model_obj = TelosModel.from_pretrained(args.checkpoint)

    if args.mode in ["category-ce", "full"]:
        run_overall_and_category_ce(model_obj, args.checkpoint)

    if args.mode in ["probes", "full"]:
        run_contextual_probes(model_obj, args.checkpoint)


if __name__ == "__main__":
    main()
