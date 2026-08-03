"""Script to measure Cross-Entropy (CE) Loss broken down by Token Category.

Categories:
1. Newlines ('\n', '\r\n')
2. Indentation (leading/standalone space tokens)
3. Punctuation & Operators ('(', ')', ':', ',', '.', '=', '+', '-', '*', '/', '[', ']', '{', '}', '#', '"', '\'')
4. Keywords ('def', 'return', 'import', 'from', 'if', 'else', 'for', 'while', 'class', 'in', 'is', 'as', 'try', 'except', 'with', 'and', 'or', 'not', 'None', 'True', 'False')
5. Identifiers ('data', 'mean', 'len', 'sum', 'a', 'b', 'n', 'x', 'self', 'math', etc.)
6. Numbers ('0'-'9', numeric subwords)
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

PYTHON_KEYWORDS = {
    "def", "return", "import", "from", "if", "else", "elif", "for", "while",
    "class", "in", "is", "as", "try", "except", "with", "and", "or", "not",
    "None", "True", "False", "lambda", "yield", "pass", "break", "continue"
}

PUNCTUATION_CHARS = set("():,.-=+*/[]{}#\"'<>!%&|^~;")

TEST_CODE_SNIPPETS = [
    "import math\n\ndef calculate_std_dev(data):\n    mean = sum(data) / len(data)\n    variance = sum((x - mean) ** 2 for x in data) / len(data)\n    return math.sqrt(variance)\n",
    "def is_prime(n: int) -> bool:\n    if n <= 1:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n",
    "def binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return -1\n"
]


def classify_token(tok_str: str) -> str:
    """Categorizes a decoded token string into one of 6 categories."""
    raw = tok_str
    clean = tok_str.strip()

    if "\n" in raw or "\r" in raw:
        return "Newlines"

    if clean == "" and len(raw) > 0:
        return "Indentation"

    if clean in PYTHON_KEYWORDS or raw in PYTHON_KEYWORDS or f" {clean}" in PYTHON_KEYWORDS:
        return "Keywords"

    if all(c in PUNCTUATION_CHARS for c in clean) and len(clean) > 0:
        return "Punctuation"

    if clean.isdigit() or (clean.replace(".", "", 1).isdigit() and len(clean) > 0):
        return "Numbers"

    if clean.isidentifier() or len(clean) > 0:
        return "Identifiers"

    return "Punctuation"


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


def run_ce_evaluation(ckpt_path: str | None = None):
    """Runs cross-entropy evaluation broken down by token category."""
    print("=" * 80)
    print("CROSS-ENTROPY LOSS BY TOKEN CATEGORY EVALUATION")
    print("=" * 80 + "\n")

    checkpoints = get_checkpoints(ckpt_path)
    if not checkpoints:
        print("No model checkpoints found to evaluate! Specify --checkpoint or place .pt files in checkpoints/")
        return

    for label, ckpt_file in checkpoints.items():
        if not Path(ckpt_file).exists():
            continue

        print(f"\n==========================================================================================")
        print(f"EVALUATING MODEL: {label}")
        print(f"==========================================================================================\n")

        try:
            model_wrapper = TelosModel.from_pretrained(ckpt_file)
            metrics = evaluate_ce_by_category(model_wrapper, TEST_CODE_SNIPPETS)

            print(f"{'Category':<15} | {'Count':<8} | {'Avg CE Loss':<12}")
            print("-" * 45)
            for cat, data in metrics.items():
                print(f"{cat:<15} | {data['count']:<8} | {data['loss']:<12.4f}")

        except Exception as e:
            print(f"Error evaluating {label}: {e}")

    print("\n" + "=" * 80 + "\n")


def evaluate_ce_by_category():
    """Measures cross-entropy loss by token category for all models."""
    print("=" * 90)
    print("CROSS-ENTROPY LOSS BY TOKEN CATEGORY BENCHMARK")
    print("=" * 90 + "\n")

    summary_table = {}

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

            cat_losses = {
                "Newlines": [],
                "Indentation": [],
                "Punctuation": [],
                "Keywords": [],
                "Identifiers": [],
                "Numbers": []
            }

            for snippet in TEST_CODE_SNIPPETS:
                enc = tokenizer.encode(snippet, add_special_tokens=False)
                token_ids = enc.ids
                seq_len = len(token_ids)

                if seq_len == 0:
                    continue

                input_tensor = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)

                # Predict every token position given the rest (or unmasked sequence)
                # To get true conditional CE at each position, evaluate forward pass with 50% mask
                masked_input = input_tensor.clone()
                torch.manual_seed(42)
                perm = torch.randperm(seq_len)
                masked_indices = perm[:seq_len // 2]
                masked_input[0, masked_indices] = mask_token_id

                with torch.no_grad():
                    logits = model(masked_input)

                logits[:, :, mask_token_id] = -float("inf")
                ce_losses = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    input_tensor.view(-1),
                    reduction="none"
                ).view(1, seq_len)

                for idx in range(seq_len):
                    if idx in masked_indices:
                        tok_id = token_ids[idx]
                        tok_str = tokenizer.decode([tok_id])
                        cat = classify_token(tok_str)
                        loss_val = ce_losses[0, idx].item()
                        cat_losses[cat].append(loss_val)

            avg_cat_ce = {}
            for cat, losses in cat_losses.items():
                if len(losses) > 0:
                    avg_cat_ce[cat] = sum(losses) / len(losses)
                else:
                    avg_cat_ce[cat] = 0.0

            summary_table[label] = avg_cat_ce

        except Exception as e:
            print(f"Error evaluating {label}: {e}")

    # Output formatted table
    print("\n" + "=" * 90)
    print("📊 CROSS-ENTROPY LOSS BY TOKEN CATEGORY SUMMARY TABLE (LOWER IS BETTER)")
    print("=" * 90)

    cats = ["Newlines", "Indentation", "Punctuation", "Keywords", "Identifiers", "Numbers"]
    header = f"{'Model Checkpoint':<23} | " + " | ".join([f"{c:<11}" for c in cats])
    print(header)
    print("-" * len(header))

    for label, cat_dict in summary_table.items():
        row_vals = " | ".join([f"{cat_dict.get(c, 0.0):>11.4f}" for c in cats])
        print(f"{label:<23} | {row_vals}")

    print("=" * 90 + "\n")


if __name__ == "__main__":
    evaluate_ce_by_category()
