"""Script to benchmark and compare Sampler A (Probability Margin) vs Sampler B (Non-Monotonic Re-Masking).

Evaluates both samplers side-by-side across all 5 ratio study model checkpoints.
"""

import sys
from pathlib import Path
import torch
from telos.hub.inference import TelosModel

CHECKPOINTS = {
    "1:1 Ratio (85M)": "checkpoints/phase_b_85m_tpu_ratio_study/checkpoint_ratio_1_1_step_162.pt",
    "1:3 Ratio (85M)": "checkpoints/phase_b_85m_tpu_ratio_study/checkpoint_ratio_1_3_step_486.pt",
    "1:5 Ratio (85M)": "checkpoints/phase_b_85m_tpu_ratio_study/checkpoint_ratio_1_5_step_811.pt",
    "1:10 Ratio (85M)": "checkpoints/phase_b_85m_tpu_ratio_study/checkpoint_ratio_1_10_step_1621.pt",
    "1:17 Ratio (85M)": "checkpoints/phase_b_85m_tpu_ratio_study/checkpoint_ratio_1_17_step_2741.pt",
}

TEST_PROMPTS = [
    "def fibonacci(n: int) -> int:\n",
    "import math\n\ndef calculate_std_dev(data):\n    mean = sum(data) / len(data)\n",
    "def is_prime(n: int) -> bool:\n",
    "class Node:\n    def __init__(self, val=0, next=None):\n        self.val = val\n        self.next = next\n\ndef reverse_list(head):\n",
    "def count_frequency(items):\n    counts = {}\n",
    "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n"
]


def compare_samplers():
    """Runs comprehensive comparative benchmark of Sampler A vs Sampler B with full untruncated output."""
    print("=" * 90)
    print("FULL SAMPLER COMPARISON REPORT: MARGIN MONOTONIC (A) VS NON-MONOTONIC RE-MASKING (B)")
    print("=" * 90 + "\n")

    for p_idx, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n==========================================================================================")
        print(f"PROMPT #{p_idx}:\n{prompt}")
        print(f"==========================================================================================\n")

        for label, ckpt_path in CHECKPOINTS.items():
            if not Path(ckpt_path).exists():
                continue

            try:
                model_wrapper = TelosModel.from_pretrained(ckpt_path)

                # Sampler A: Probability Margin Monotonic Sampler (temp=0.0, rep_penalty=1.0)
                out_a = model_wrapper.complete(
                    prompt=prompt,
                    max_tokens=64,
                    num_steps=64,
                    temperature=0.0,
                    repetition_penalty=1.0,
                    schedule="cosine"
                )

                # Sampler B: Non-Monotonic Re-Masking Sampler (temp=0.0, rep_penalty=1.0)
                out_b = model_wrapper.complete_non_monotonic(
                    prompt=prompt,
                    max_tokens=64,
                    num_steps=64,
                    temperature=0.0,
                    repetition_penalty=1.0,
                    schedule="cosine",
                    remask_threshold=0.15
                )

                print(f"┌─────────────────────────────────────────────────────────────────────────┐")
                print(f"│ MODEL CHECKPOINT: {label:<53} │")
                print(f"└─────────────────────────────────────────────────────────────────────────┘")
                print(f"[SAMPLER A - Margin Monotonic Full Output]:\n{out_a}\n")
                print(f"[SAMPLER B - Non-Monotonic Re-Masking Full Output]:\n{out_b}\n")
                print("-" * 90 + "\n")

            except Exception as e:
                print(f"Error evaluating {label}: {e}\n")


if __name__ == "__main__":
    compare_samplers()
