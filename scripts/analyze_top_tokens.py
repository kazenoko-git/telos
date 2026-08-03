"""Script to analyze and print Top 20 and Top 50 predicted tokens across 6 model checkpoints.

Evaluates next-token probability distributions right after input prompt across:
- 25M TPU Model
- 85M TPU Ratio 1:1
- 85M TPU Ratio 1:3
- 85M TPU Ratio 1:5
- 85M TPU Ratio 1:10
- 85M TPU Ratio 1:17
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from telos.hub.inference import TelosModel

# 6 Model Checkpoints to compare
CHECKPOINTS = {
    "25M TPU Model": "checkpoints/phase_b_25m_tpu_bundle",
    "85M TPU 1:1 Ratio": "checkpoints/phase_b_85m_tpu_ratio_study/checkpoint_ratio_1_1_step_162.pt",
    "85M TPU 1:3 Ratio": "checkpoints/phase_b_85m_tpu_ratio_study/checkpoint_ratio_1_3_step_486.pt",
    "85M TPU 1:5 Ratio": "checkpoints/phase_b_85m_tpu_ratio_study/checkpoint_ratio_1_5_step_811.pt",
    "85M TPU 1:10 Ratio": "checkpoints/phase_b_85m_tpu_ratio_study/checkpoint_ratio_1_10_step_1621.pt",
    "85M TPU 1:17 Ratio": "checkpoints/phase_b_85m_tpu_ratio_study/checkpoint_ratio_1_17_step_2741.pt",
}


def analyze_prompt_next_token(prompt: str, top_k: int = 50):
    """Prints top-k predicted tokens for the first masked position following prompt."""
    print(f"\n================================================================================")
    print(f"PROMPT UNDER EVALUATION:\n{prompt!r}")
    print(f"================================================================================\n")

    for label, ckpt_path in CHECKPOINTS.items():
        if not Path(ckpt_path).exists():
            print(f"Skipping {label}: file not found at {ckpt_path}\n")
            continue

        try:
            # Load model
            model_wrapper = TelosModel.from_pretrained(ckpt_path)
            tokenizer = model_wrapper.tokenizer
            model = model_wrapper.model
            device = model_wrapper.device

            # Encode prompt
            enc = tokenizer.encode(prompt, add_special_tokens=False)
            prompt_ids = enc.ids
            prompt_len = len(prompt_ids)

            # Construct input tensor [1, seq_len] with prompt + [MASK] tokens up to max_seq_len
            max_len = min(prompt_len + 32, 512)
            seq = torch.full((1, max_len), fill_value=model_wrapper.config.mask_token_id, dtype=torch.long, device=device)
            seq[0, :prompt_len] = torch.tensor(prompt_ids, dtype=torch.long, device=device)

            # Model forward pass
            with torch.no_grad():
                logits = model(seq)  # [1, seq_len, vocab_size]

            # Next token position is right after prompt (index prompt_len)
            next_pos_logits = logits[0, prompt_len, :]  # [vocab_size]

            # Zero out mask_token_id so it is never predicted
            next_pos_logits[model_wrapper.config.mask_token_id] = -float("inf")

            # Softmax to get probabilities
            probs = F.softmax(next_pos_logits, dim=-1)

            # Top-k tokens
            top_probs, top_indices = torch.topk(probs, k=top_k)

            print(f"--- {label} ---")
            print(f"{'Rank':<6} | {'Token ID':<9} | {'Prob (%)':<9} | {'Decoded Token String':<30}")
            print("-" * 65)

            for rank in range(top_k):
                tok_id = top_indices[rank].item()
                prob_pct = top_probs[rank].item() * 100
                tok_str = repr(tokenizer.decode([tok_id]))
                print(f"#{rank+1:<5} | {tok_id:<9} | {prob_pct:>8.2f}% | {tok_str:<30}")

            print("\n")

        except Exception as e:
            print(f"Error evaluating {label}: {e}\n")


if __name__ == "__main__":
    test_prompt = "import math\n\ndef calculate_std_dev(data):\n    mean = sum(data) / len(data)\n"
    if len(sys.argv) > 1:
        test_prompt = sys.argv[1]

    analyze_prompt_next_token(test_prompt, top_k=50)
