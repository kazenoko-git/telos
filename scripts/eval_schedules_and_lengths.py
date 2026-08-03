"""Script to evaluate 4 Timestep Schedules x 3 Mask Suffix Lengths across all 5 model ratio checkpoints.

Schedules:
1. Cosine: 1 - cos(progress * pi / 2)
2. Front-loaded: progress**0.5 (unmasks fast early)
3. Back-loaded: progress**2.0 (unmasks fast late)
4. Sigmoid: 1 / (1 + exp(-10 * (progress - 0.5)))

Mask Suffix Lengths:
1. 500 masks (Current full blank slate)
2. 64 masks (Short completion target)
3. 128 masks (Medium completion target)
"""

import sys
import math
from pathlib import Path
import torch
import torch.nn.functional as F
from telos.hub.inference import TelosModel

CHECKPOINTS = {
    "1:1 Ratio (85M)": "checkpoints/phase_b_85m_tpu_ratio_study/checkpoint_ratio_1_1_step_162.pt",
    "1:3 Ratio (85M)": "checkpoints/phase_b_85m_tpu_ratio_study/checkpoint_ratio_1_3_step_486.pt",
    "1:5 Ratio (85M)": "checkpoints/phase_b_85m_tpu_ratio_study/checkpoint_ratio_1_5_step_811.pt",
    "1:10 Ratio (85M)": "checkpoints/phase_b_85m_tpu_ratio_study/checkpoint_ratio_1_10_step_1621.pt",
    "1:17 Ratio (85M)": "checkpoints/phase_b_85m_tpu_ratio_study/checkpoint_ratio_1_17_step_2741.pt",
}

SCHEDULES = ["cosine", "front_loaded", "back_loaded", "sigmoid"]
MASK_LENGTHS = [485, 64, 128]

TEST_PROMPT = "import math\n\ndef calculate_std_dev(data):\n    mean = sum(data) / len(data)\n"


def get_unmask_ratio(step: int, num_steps: int, schedule_name: str) -> float:
    """Computes target unmask ratio at step according to chosen schedule."""
    progress = min(max((step + 1) / num_steps, 0.0), 1.0)

    if schedule_name == "cosine":
        return 1.0 - math.cos(progress * math.pi / 2.0)
    elif schedule_name == "front_loaded":
        return math.sqrt(progress)  # unmasks lots early
    elif schedule_name == "back_loaded":
        return progress ** 2.0  # unmasks slow early, fast late
    elif schedule_name == "sigmoid":
        # S-curve scaled to [0, 1]
        sig = 1.0 / (1.0 + math.exp(-10.0 * (progress - 0.5)))
        sig_0 = 1.0 / (1.0 + math.exp(5.0))
        sig_1 = 1.0 / (1.0 + math.exp(-5.0))
        return (sig - sig_0) / (sig_1 - sig_0)
    else:
        return progress


def run_custom_sampler(
    model_wrapper: TelosModel,
    prompt: str,
    target_mask_len: int,
    num_steps: int = 64,
    schedule_name: str = "cosine"
) -> str:
    """Runs customized diffusion sampler with specified schedule and mask length."""
    model = model_wrapper.model
    tokenizer = model_wrapper.tokenizer
    device = model_wrapper.device

    mask_token_id = tokenizer.token_to_id("[MASK]")
    if mask_token_id is None:
        mask_token_id = 4

    enc = tokenizer.encode(prompt, add_special_tokens=False)
    prompt_ids = enc.ids
    prompt_len = len(prompt_ids)

    seq_len = prompt_len + target_mask_len
    seq = torch.full((1, seq_len), mask_token_id, dtype=torch.long, device=device)
    seq[0, :prompt_len] = torch.tensor(prompt_ids, dtype=torch.long, device=device)

    total_masked = target_mask_len
    already_unmasked = 0

    for step in range(num_steps):
        is_final = (step == num_steps - 1)

        current_mask = (seq[0] == mask_token_id)
        current_mask[:prompt_len] = False

        num_masked = current_mask.sum().item()
        if num_masked == 0:
            break

        logits = model(seq).clone()
        logits[:, :, mask_token_id] = -float("inf")

        probs = F.softmax(logits, dim=-1)

        top2_probs, top2_indices = torch.topk(probs, k=2, dim=-1)
        margins = top2_probs[0, :, 0] - top2_probs[0, :, 1]
        top1_tokens = top2_indices[0, :, 0]

        if is_final:
            seq[0, current_mask] = top1_tokens[current_mask]
            break

        ratio = get_unmask_ratio(step, num_steps, schedule_name)
        target_unmasked = math.ceil(ratio * total_masked)
        num_to_unmask = max(1, target_unmasked - already_unmasked)

        scores = margins.clone()
        scores[~current_mask] = -float("inf")

        k = min(num_to_unmask, num_masked)
        if k > 0:
            _, topk_indices = torch.topk(scores, k=k)
            seq[0, topk_indices] = top1_tokens[topk_indices]
            already_unmasked += k

    full_text = tokenizer.decode(seq[0].tolist(), skip_special_tokens=True)
    return full_text.rstrip()


def evaluate_all():
    """Executes 4 Schedules x 3 Mask Lengths matrix evaluation."""
    print("=" * 90)
    print("TIMESTEP SCHEDULE (4) X MASK SUFFIX LENGTH (3) MATRIX BENCHMARK")
    print("=" * 90 + "\n")

    print(f"PROMPT UNDER EVALUATION:\n{TEST_PROMPT!r}\n")

    for label, ckpt_path in CHECKPOINTS.items():
        if not Path(ckpt_path).exists():
            continue

        print(f"\n==========================================================================================")
        print(f"MODEL CHECKPOINT: {label}")
        print(f"==========================================================================================\n")

        try:
            model_wrapper = TelosModel.from_pretrained(ckpt_path)

            for target_len in MASK_LENGTHS:
                print(f"--- MASK LENGTH: Prompt + [MASK] x {target_len} ---")

                for sched in SCHEDULES:
                    out_str = run_custom_sampler(
                        model_wrapper,
                        prompt=TEST_PROMPT,
                        target_mask_len=target_len,
                        num_steps=64,
                        schedule_name=sched
                    )
                    print(f"  [{sched.upper():<12}]: {out_str!r}")

                print("-" * 75)

        except Exception as e:
            print(f"Error evaluating {label}: {e}")


if __name__ == "__main__":
    evaluate_all()
