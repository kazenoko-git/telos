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
    "import math\n\ndef calculate_std_dev(data):\n    mean = sum(data) / len(data)\n",
    "def is_prime(n: int) -> bool:\n",
]


def compare_samplers():
    """Runs comparative benchmark of Sampler A vs Sampler B."""
    print("=" * 80)
    print("SAMPLER COMPARISON: MARGIN MONOTONIC (A) VS NON-MONOTONIC RE-MASKING (B)")
    print("=" * 80 + "\n")

    for prompt in TEST_PROMPTS:
        print(f"\n================================================================================")
        print(f"EVALUATING PROMPT:\n{prompt!r}")
        print(f"================================================================================\n")

        for label, ckpt_path in CHECKPOINTS.items():
            if not Path(ckpt_path).exists():
                continue

            try:
                model_wrapper = TelosModel.from_pretrained(ckpt_path)

                # Sampler A: Probability Margin Monotonic Sampler
                out_a = model_wrapper.complete(
                    prompt=prompt,
                    max_tokens=64,
                    num_steps=64,
                    temperature=0.0,
                    repetition_penalty=1.0,
                    schedule="cosine"
                )

                # Sampler B: Non-Monotonic Re-Masking Sampler
                out_b = model_wrapper.complete_non_monotonic(
                    prompt=prompt,
                    max_tokens=64,
                    num_steps=64,
                    temperature=0.0,
                    repetition_penalty=1.0,
                    schedule="cosine",
                    remask_threshold=0.15
                )

                print(f"--- {label} ---")
                print("SAMPLER A (Probability Margin Monotonic):")
                print(repr(out_a))
                print("\nSAMPLER B (Non-Monotonic Dynamic Re-Masking):")
                print(repr(out_b))
                print("-" * 65 + "\n")

            except Exception as e:
                print(f"Error evaluating {label}: {e}\n")


if __name__ == "__main__":
    compare_samplers()
