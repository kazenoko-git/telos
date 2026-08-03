"""Script to evaluate contextual target prediction across categories:
- Top-1 Accuracy
- Top-5 Accuracy
- Rank of Correct Token

Suite of targeted contextual probes:
1. 'def add(a, b):\n    return a + ' -> Target: ' b' (Identifier)
2. 'if x == ' -> Target: ' None' (Literal)
3. 'x = 10\nprint(' -> Target: 'x' (Identifier)
4. 'for i in ' -> Target: ' range' (Keyword/Builtin)
5. 'class Foo(' -> Target: 'object' (Identifier)
6. 'def foo(x):\n' -> Target: '    return' (Keyword)
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

PROBE_SUITE = [
    {
        "name": "return a + [MASK]",
        "prompt": "def add(a, b):\n    return a + ",
        "target": " b",
        "category": "Identifier"
    },
    {
        "name": "if x == [MASK]",
        "prompt": "if x == ",
        "target": " None",
        "category": "Literal"
    },
    {
        "name": "print([MASK])",
        "prompt": "x = 10\nprint(",
        "target": "x",
        "category": "Identifier"
    },
    {
        "name": "for i in [MASK]",
        "prompt": "for i in ",
        "target": " range",
        "category": "Keyword"
    },
    {
        "name": "class Foo([MASK])",
        "prompt": "class Foo(",
        "target": "object",
        "category": "Identifier"
    },
    {
        "name": "def foo(x):\\n [MASK]",
        "prompt": "def foo(x):\n",
        "target": "    return",
        "category": "Keyword"
    }
]


def get_checkpoints(cli_path: str | None = None) -> dict[str, str]:
    """Dynamically resolves checkpoints from CLI path or searches checkpoints directory recursively."""
    if cli_path:
        p = Path(cli_path)
        if p.is_file():
            return {p.stem: str(p)}
        elif p.is_dir():
            found = {pt.stem: str(pt) for pt in sorted(p.rglob("*.pt"))}
            if found:
                return found

    # Fallback to recursive discovery in checkpoints/
    found = {}
    ck_root = Path("checkpoints")
    if ck_root.exists():
        for pt in sorted(ck_root.rglob("*.pt")):
            found[pt.stem] = str(pt)
    return found


def run_contextual_probe(ckpt_path: str | None = None):
    """Runs the targeted contextual probe benchmark."""
    print("=" * 90)
    print("CONTEXTUAL MASK PROBE BENCHMARK: TOP-1, TOP-5 & RANK ACCURACY")
    print("=" * 90 + "\n")

    checkpoints = get_checkpoints(ckpt_path)
    if not checkpoints:
        print("No model checkpoints found to evaluate! Specify --checkpoint or place .pt files in checkpoints/")
        return

    for label, ckpt_file in checkpoints.items():
        if not Path(ckpt_file).exists():
            continue

        print(f"\n==========================================================================================")
        print(f"MODEL CHECKPOINT: {label}")
        print(f"==========================================================================================\n")

        try:
            model_wrapper = TelosModel.from_pretrained(ckpt_file)
            tokenizer = model_wrapper.tokenizer
            model = model_wrapper.model
            device = model_wrapper.device

            mask_token_id = tokenizer.token_to_id("[MASK]")
            if mask_token_id is None:
                mask_token_id = 4

            print(f"{'Probe Prompt':<25} | {'Category':<12} | {'Target Token':<12} | {'Top-1':<6} | {'Top-5':<6} | {'Rank':<6} | {'Target Prob':<11}")
            print("-" * 90)

            for probe in PROBE_SUITE:
                prompt = probe["prompt"]
                target_str = probe["target"]
                cat = probe["category"]
                probe_name = probe["name"]

                enc_prompt = tokenizer.encode(prompt, add_special_tokens=False).ids
                enc_target = tokenizer.encode(target_str, add_special_tokens=False).ids

                if len(enc_target) == 0:
                    continue
                target_id = enc_target[0]

                seq_len = len(enc_prompt) + 1
                input_tensor = torch.full((1, seq_len), fill_value=mask_token_id, dtype=torch.long, device=device)
                input_tensor[0, :len(enc_prompt)] = torch.tensor(enc_prompt, dtype=torch.long, device=device)

                with torch.no_grad():
                    logits = model(input_tensor)

                mask_logits = logits[0, len(enc_prompt), :].clone()
                mask_logits[mask_token_id] = -float("inf")
                probs = F.softmax(mask_logits, dim=-1)

                sorted_probs, sorted_indices = torch.sort(probs, descending=True)

                # Find rank of target_id
                target_rank = (sorted_indices == target_id).nonzero(as_tuple=True)[0].item() + 1
                target_prob_pct = probs[target_id].item() * 100.0

                top1_hit = "YES" if target_rank == 1 else "NO"
                top5_hit = "YES" if target_rank <= 5 else "NO"

                print(f"{probe_name:<25} | {cat:<12} | {repr(target_str):<12} | {top1_hit:<6} | {top5_hit:<6} | #{target_rank:<5} | {target_prob_pct:>9.4f}%")

        except Exception as e:
            print(f"Error evaluating {label}: {e}")

    print("\n" + "=" * 90 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Contextual Mask Probe Benchmark")
    parser.add_argument("--checkpoint", "--ckpt-dir", type=str, default=None, help="Path to checkpoint file (.pt) or directory")
    args = parser.parse_args()

    run_contextual_probe(args.checkpoint)
