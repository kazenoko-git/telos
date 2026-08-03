"""Script to measure Exact Match Mask Reconstruction Accuracy across various mask ratios.

Evaluates models on real Python code snippets masked at exact ratios:
r = [0.20, 0.40, 0.60, 0.80, 0.95, 0.99]

Measures:
Accuracy = (Correctly predicted masked tokens / Total masked tokens) * 100%
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from telos.hub.inference import TelosModel

CHECKPOINTS = {
    "25M TPU Model": "checkpoints/phase_b_25m_tpu_bundle",
    "85M TPU 1:1 Ratio": "checkpoints/phase_b_85m_tpu_ratio_study/checkpoint_ratio_1_1_step_162.pt",
    "85M TPU 1:3 Ratio": "checkpoints/phase_b_85m_tpu_ratio_study/checkpoint_ratio_1_3_step_486.pt",
    "85M TPU 1:5 Ratio": "checkpoints/phase_b_85m_tpu_ratio_study/checkpoint_ratio_1_5_step_811.pt",
    "85M TPU 1:10 Ratio": "checkpoints/phase_b_85m_tpu_ratio_study/checkpoint_ratio_1_10_step_1621.pt",
    "85M TPU 1:17 Ratio": "checkpoints/phase_b_85m_tpu_ratio_study/checkpoint_ratio_1_17_step_2741.pt",
}

MASK_RATIOS = [0.20, 0.40, 0.60, 0.80, 0.95, 0.99]

# Clean reference Python code blocks for evaluation
REAL_CODE_SNIPPETS = [
    "import math\n\ndef calculate_std_dev(data):\n    mean = sum(data) / len(data)\n    variance = sum((x - mean) ** 2 for x in data) / len(data)\n    return math.sqrt(variance)\n",
    "def is_prime(n: int) -> bool:\n    if n <= 1:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n",
    "def binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return -1\n"
]


def evaluate_reconstruction():
    """Evaluates mask reconstruction accuracy across mask ratios."""
    print("=" * 90)
    print("MDLM MASK RECONSTRUCTION ACCURACY BENCHMARK ACROSS MASK RATING (20% -> 99%)")
    print("=" * 90 + "\n")

    torch.manual_seed(42)  # Fixed reproducible seed for masking mask generation

    results_table = {}

    for label, ckpt_path in CHECKPOINTS.items():
        if not Path(ckpt_path).exists():
            continue

        try:
            model_wrapper = TelosModel.from_pretrained(ckpt_path)
            tokenizer = model_wrapper.tokenizer
            model = model_wrapper.model
            device = model_wrapper.device

            mask_token_id = tokenizer.token_to_id("[MASK]")
            if mask_token_id is None:
                mask_token_id = 4

            ratio_accuracies = {}

            for mask_ratio in MASK_RATIOS:
                total_masked_tokens = 0
                correct_predictions = 0

                for snippet in REAL_CODE_SNIPPETS:
                    enc = tokenizer.encode(snippet, add_special_tokens=False)
                    token_ids = enc.ids
                    seq_len = len(token_ids)

                    if seq_len == 0:
                        continue

                    # Create masked sequence according to mask_ratio
                    input_tensor = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
                    target_tensor = input_tensor.clone()

                    # Determine masked positions (random mask at exact ratio)
                    num_to_mask = int(math.ceil(seq_len * mask_ratio))
                    perm = torch.randperm(seq_len)
                    masked_indices = perm[:num_to_mask]

                    masked_input = target_tensor.clone()
                    masked_input[0, masked_indices] = mask_token_id

                    # Model forward pass
                    with torch.no_grad():
                        logits = model(masked_input)

                    logits[:, :, mask_token_id] = -float("inf")
                    pred_tokens = torch.argmax(logits[0], dim=-1)

                    # Evaluate exact match accuracy on masked positions
                    masked_targets = target_tensor[0, masked_indices]
                    masked_preds = pred_tokens[masked_indices]

                    correct = (masked_preds == masked_targets).sum().item()
                    correct_predictions += correct
                    total_masked_tokens += num_to_mask

                acc_pct = (correct_predictions / max(1, total_masked_tokens)) * 100.0
                ratio_accuracies[mask_ratio] = acc_pct

            results_table[label] = ratio_accuracies

        except Exception as e:
            print(f"Error evaluating {label}: {e}")

    # Output formatted summary table
    print("\n" + "=" * 90)
    print("📊 EXACT MATCH MASK RECONSTRUCTION ACCURACY (%) SUMMARY TABLE")
    print("=" * 90)
    
    header = f"{'Model Checkpoint':<25} | " + " | ".join([f"{int(r*100)}% Mask" for r in MASK_RATIOS])
    print(header)
    print("-" * len(header))

    for label, acc_dict in results_table.items():
        row_vals = " | ".join([f"{acc_dict.get(r, 0.0):>7.2f}%" for r in MASK_RATIOS])
        print(f"{label:<25} | {row_vals}")

    print("=" * 90 + "\n")


if __name__ == "__main__":
    evaluate_reconstruction()
