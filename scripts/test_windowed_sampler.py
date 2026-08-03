"""Script to test the Windowed Progressive Localized Sampler (Sampler C) against Samplers A & B.
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
]


def test_windowed_sampler():
    """Runs comparative evaluation including the Windowed Localized Sampler (C)."""
    print("=" * 90)
    print("3-WAY SAMPLER COMPARISON: GLOBAL MONOTONIC (A) VS NON-MONOTONIC (B) VS WINDOWED (C)")
    print("=" * 90 + "\n")

    for prompt in TEST_PROMPTS:
        print(f"\n==========================================================================================")
        print(f"PROMPT:\n{prompt}")
        print(f"==========================================================================================\n")

        for label, ckpt_path in CHECKPOINTS.items():
            if not Path(ckpt_path).exists():
                continue

            try:
                model_wrapper = TelosModel.from_pretrained(ckpt_path)

                # Sampler A: Global Margin Monotonic Sampler
                out_a = model_wrapper.complete(
                    prompt=prompt,
                    max_tokens=64,
                    num_steps=64,
                    temperature=0.0,
                    repetition_penalty=1.0,
                    schedule="cosine"
                )

                # Sampler B: Global Non-Monotonic Re-Masking Sampler
                out_b = model_wrapper.complete_non_monotonic(
                    prompt=prompt,
                    max_tokens=64,
                    num_steps=64,
                    temperature=0.0,
                    repetition_penalty=1.0,
                    schedule="cosine",
                    remask_threshold=0.15
                )

                # Sampler C: Windowed Progressive Localized Sampler (W=32)
                out_c = model_wrapper.complete_windowed(
                    prompt=prompt,
                    max_tokens=64,
                    window_size=32,
                    num_steps_per_window=16,
                    temperature=0.0,
                    remask_threshold=0.15
                )

                print(f"┌─────────────────────────────────────────────────────────────────────────┐")
                print(f"│ MODEL CHECKPOINT: {label:<53} │")
                print(f"└─────────────────────────────────────────────────────────────────────────┘")
                print(f"[SAMPLER A - Global Margin Monotonic]:\n{out_a!r}\n")
                print(f"[SAMPLER B - Global Non-Monotonic Re-Masking]:\n{out_b!r}\n")
                print(f"[SAMPLER C - Windowed Progressive Localized (W=32)]:\n{out_c!r}\n")
                print("-" * 90 + "\n")

            except Exception as e:
                print(f"Error evaluating {label}: {e}\n")


if __name__ == "__main__":
    test_windowed_sampler()
