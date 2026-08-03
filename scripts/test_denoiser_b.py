"""Script to test single-token mask denoising accuracy for `def add(a, b):\n    return a + [MASK]`

Evaluates whether the models allocate high probability (>95%) to predicting token 'b'.
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

PROMPT_BEFORE_MASK = "def add(a, b):\n    return a + "


def evaluate_add_b():
    """Evaluates top token predictions for `def add(a, b):\n    return a + [MASK]`."""
    print("=" * 80)
    print("DENOISER PROOF BENCHMARK: def add(a, b): return a + [MASK]")
    print("=" * 80 + "\n")

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

            # Encode prefix before mask
            enc = tokenizer.encode(PROMPT_BEFORE_MASK, add_special_tokens=False)
            prompt_ids = enc.ids
            prompt_len = len(prompt_ids)

            # Construct sequence with 1 [MASK] token at the end
            seq = torch.full((1, prompt_len + 1), fill_value=mask_token_id, dtype=torch.long, device=device)
            seq[0, :prompt_len] = torch.tensor(prompt_ids, dtype=torch.long, device=device)

            with torch.no_grad():
                logits = model(seq)

            mask_logits = logits[0, prompt_len, :].clone()
            mask_logits[mask_token_id] = -float("inf")
            probs = F.softmax(mask_logits, dim=-1)

            top_probs, top_indices = torch.topk(probs, k=15)

            target_b_id = tokenizer.encode(" b", add_special_tokens=False).ids[0]
            target_b_prob = probs[target_b_id].item() * 100

            print(f"--- {label} ---")
            print(f"Target token ' b' (ID {target_b_id}) Prob: {target_b_prob:.4f}%\n")
            print(f"{'Rank':<6} | {'Token ID':<9} | {'Prob (%)':<10} | {'Decoded Token String':<25}")
            print("-" * 60)

            for rank in range(15):
                tok_id = top_indices[rank].item()
                prob_pct = top_probs[rank].item() * 100
                tok_str = tokenizer.decode([tok_id])
                print(f"#{rank+1:<5} | {tok_id:<9} | {prob_pct:>8.2f}%  | {repr(tok_str):<25}")

            print("\n")

        except Exception as e:
            print(f"Error evaluating {label}: {e}\n")


if __name__ == "__main__":
    evaluate_add_b()
