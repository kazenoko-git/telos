"""Script to test the 'Boring Decoder' hypothesis against standard sampler across all 5 ratio checkpoints.

Compares:
1. Standard Sampler (temp=0.3, rep_penalty=1.2, multinomial, Gumbel noise)
2. Boring Decoder (temp=0.0, rep_penalty=1.0, argmax, zero noise)
3. Non-Monotonic Re-masking Decoder (allows revising unmasked tokens if confidence changes)
"""

import sys
from pathlib import Path
import torch
from tokenizers import Tokenizer
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


def test_decoders():
    """Runs comparative evaluation across all 5 models."""
    print("=" * 80)
    print("BORING DECODER VS STANDARD SAMPLER EVALUATION")
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

                # 1. Standard Sampler (temp=0.3, rep_penalty=1.2)
                out_std = model_wrapper.complete(
                    prompt=prompt,
                    max_tokens=64,
                    num_steps=64,
                    temperature=0.3,
                    repetition_penalty=1.2
                )

                # 2. Boring Decoder (temp=0.0, rep_penalty=1.0)
                out_boring = model_wrapper.complete(
                    prompt=prompt,
                    max_tokens=64,
                    num_steps=64,
                    temperature=0.0,
                    repetition_penalty=1.0
                )

                print(f"--- {label} ---")
                print("STANDARD SAMPLER (temp=0.3, rep=1.2):")
                print(repr(out_std))
                print("\nBORING DECODER (temp=0.0, rep=1.0, argmax):")
                print(repr(out_boring))
                print("-" * 65 + "\n")

            except Exception as e:
                print(f"Error evaluating {label}: {e}\n")


if __name__ == "__main__":
    test_decoders()
