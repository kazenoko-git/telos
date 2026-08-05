"""Sampling & Contextual Probes Suite for télos MDLM Models.

Includes:
1. 100 Contextual Probes organized across 8 categories:
   - Identifier recovery
   - Function names
   - Keywords
   - Operators
   - Literals
   - Imports
   - Class names
   - Attribute names
2. Aggregate Summary Metrics:
   - Average Target CE (overall & per category)
   - Average Rank
   - Top-1 Accuracy (%)
   - Top-5 Accuracy (%)
"""

import sys
import math
import argparse
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from telos.data.tokenizer import load_tokenizer
from telos.eval.sample import run_qualitative_evaluation


PROBE_SUITE_100 = [
    # --- 1. Identifier Recovery (13) ---
    {"category": "Identifier recovery", "prompt": "return a +", "target": "b", "target_bpe": "b"},
    {"category": "Identifier recovery", "prompt": "x = 10\nprint(", "target": "x", "target_bpe": "x"},
    {"category": "Identifier recovery", "prompt": "def __init__(self,", "target": "name", "target_bpe": "Ġname"},
    {"category": "Identifier recovery", "prompt": "self.name =", "target": "name", "target_bpe": "Ġname"},
    {"category": "Identifier recovery", "prompt": "total = sum(", "target": "items", "target_bpe": "Ġitems"},
    {"category": "Identifier recovery", "prompt": "for elem in", "target": "lst", "target_bpe": "Ġlst"},
    {"category": "Identifier recovery", "prompt": "res = val *", "target": "factor", "target_bpe": "Ġfactor"},
    {"category": "Identifier recovery", "prompt": "msg = str(", "target": "err", "target_bpe": "Ġerr"},
    {"category": "Identifier recovery", "prompt": "dx = x2 -", "target": "x1", "target_bpe": "Ġx1"},
    {"category": "Identifier recovery", "prompt": "data = json.loads(", "target": "text", "target_bpe": "Ġtext"},
    {"category": "Identifier recovery", "prompt": "res = []\nfor x in", "target": "items", "target_bpe": "Ġitems"},
    {"category": "Identifier recovery", "prompt": "left +", "target": "right", "target_bpe": "Ġright"},
    {"category": "Identifier recovery", "prompt": "width *", "target": "height", "target_bpe": "Ġheight"},

    # --- 2. Function Names (13) ---
    {"category": "Function names", "prompt": "def get_", "target": "name", "target_bpe": "name"},
    {"category": "Function names", "prompt": "def set_", "target": "val", "target_bpe": "val"},
    {"category": "Function names", "prompt": "def parse_", "target": "data", "target_bpe": "data"},
    {"category": "Function names", "prompt": "def build_", "target": "model", "target_bpe": "model"},
    {"category": "Function names", "prompt": "def test_", "target": "func", "target_bpe": "func"},
    {"category": "Function names", "prompt": "def process_", "target": "request", "target_bpe": "request"},
    {"category": "Function names", "prompt": "def validate_", "target": "input", "target_bpe": "input"},
    {"category": "Function names", "prompt": "def load_", "target": "config", "target_bpe": "config"},
    {"category": "Function names", "prompt": "def save_", "target": "file", "target_bpe": "file"},
    {"category": "Function names", "prompt": "def calculate_", "target": "total", "target_bpe": "total"},
    {"category": "Function names", "prompt": "def convert_", "target": "type", "target_bpe": "type"},
    {"category": "Function names", "prompt": "def read_", "target": "bytes", "target_bpe": "bytes"},
    {"category": "Function names", "prompt": "def create_", "target": "instance", "target_bpe": "instance"},

    # --- 3. Keywords (13) ---
    {"category": "Keywords", "prompt": "if x == 1:\n    pass\n", "target": "else", "target_bpe": "else"},
    {"category": "Keywords", "prompt": "try:\n    pass\n", "target": "except", "target_bpe": "except"},
    {"category": "Keywords", "prompt": "for i in", "target": "range", "target_bpe": "Ġrange"},
    {"category": "Keywords", "prompt": "with open(path) ", "target": "as", "target_bpe": "Ġas"},
    {"category": "Keywords", "prompt": "if not", "target": "found", "target_bpe": "Ġfound"},
    {"category": "Keywords", "prompt": "while", "target": "True", "target_bpe": "ĠTrue"},
    {"category": "Keywords", "prompt": "from typing", "target": "import", "target_bpe": "Ġimport"},
    {"category": "Keywords", "prompt": "assert x is", "target": "not", "target_bpe": "Ġnot"},
    {"category": "Keywords", "prompt": "if item", "target": "in", "target_bpe": "Ġin"},
    {"category": "Keywords", "prompt": "def func():\n   ", "target": "return", "target_bpe": "Ġreturn"},
    {"category": "Keywords", "prompt": "raise ValueError(", "target": "msg", "target_bpe": "msg"},
    {"category": "Keywords", "prompt": "class MyClass(", "target": "object", "target_bpe": "object"},
    {"category": "Keywords", "prompt": "yield", "target": "from", "target_bpe": "Ġfrom"},

    # --- 4. Operators (12) ---
    {"category": "Operators", "prompt": "x = a +", "target": "b", "target_bpe": "Ġb"},
    {"category": "Operators", "prompt": "if a ==", "target": "b", "target_bpe": "Ġb"},
    {"category": "Operators", "prompt": "x +=", "target": "1", "target_bpe": "Ġ1"},
    {"category": "Operators", "prompt": "a >", "target": "0", "target_bpe": "Ġ0"},
    {"category": "Operators", "prompt": "x = y *", "target": "z", "target_bpe": "Ġz"},
    {"category": "Operators", "prompt": "a !=", "target": "None", "target_bpe": "ĠNone"},
    {"category": "Operators", "prompt": "count = len(arr) -", "target": "1", "target_bpe": "Ġ1"},
    {"category": "Operators", "prompt": "if x <=", "target": "max_val", "target_bpe": "Ġmax_val"},
    {"category": "Operators", "prompt": "idx = (i + 1) %", "target": "n", "target_bpe": "Ġn"},
    {"category": "Operators", "prompt": "res = a &", "target": "b", "target_bpe": "Ġb"},
    {"category": "Operators", "prompt": "flags = A |", "target": "B", "target_bpe": "ĠB"},
    {"category": "Operators", "prompt": "val = x **", "target": "2", "target_bpe": "Ġ2"},

    # --- 5. Literals (12) ---
    {"category": "Literals", "prompt": "if x is", "target": "None", "target_bpe": "ĠNone"},
    {"category": "Literals", "prompt": "flag =", "target": "True", "target_bpe": "ĠTrue"},
    {"category": "Literals", "prompt": "status =", "target": "False", "target_bpe": "ĠFalse"},
    {"category": "Literals", "prompt": "count =", "target": "0", "target_bpe": "Ġ0"},
    {"category": "Literals", "prompt": "name =", "target": "\"\"", "target_bpe": "Ġ\"\""},
    {"category": "Literals", "prompt": "items =", "target": "[]", "target_bpe": "Ġ[]"},
    {"category": "Literals", "prompt": "data =", "target": "{}", "target_bpe": "Ġ{}"},
    {"category": "Literals", "prompt": "rate =", "target": "0.0", "target_bpe": "Ġ0.0"},
    {"category": "Literals", "prompt": "idx =", "target": "-1", "target_bpe": "Ġ-1"},
    {"category": "Literals", "prompt": "pi =", "target": "3.14", "target_bpe": "Ġ3.14"},
    {"category": "Literals", "prompt": "res =", "target": "1", "target_bpe": "Ġ1"},
    {"category": "Literals", "prompt": "msg =", "target": "\"hello\"", "target_bpe": "Ġ\"hello\""},

    # --- 6. Imports (13) ---
    {"category": "Imports", "prompt": "import", "target": "os", "target_bpe": "Ġos"},
    {"category": "Imports", "prompt": "import", "target": "sys", "target_bpe": "Ġsys"},
    {"category": "Imports", "prompt": "import", "target": "json", "target_bpe": "Ġjson"},
    {"category": "Imports", "prompt": "import", "target": "time", "target_bpe": "Ġtime"},
    {"category": "Imports", "prompt": "import", "target": "math", "target_bpe": "Ġmath"},
    {"category": "Imports", "prompt": "import", "target": "re", "target_bpe": "Ġre"},
    {"category": "Imports", "prompt": "import", "target": "random", "target_bpe": "Ġrandom"},
    {"category": "Imports", "prompt": "from typing import", "target": "List", "target_bpe": "ĠList"},
    {"category": "Imports", "prompt": "from pathlib import", "target": "Path", "target_bpe": "ĠPath"},
    {"category": "Imports", "prompt": "import numpy as", "target": "np", "target_bpe": "Ġnp"},
    {"category": "Imports", "prompt": "import torch.nn as", "target": "nn", "target_bpe": "Ġnn"},
    {"category": "Imports", "prompt": "from collections import", "target": "defaultdict", "target_bpe": "Ġdefaultdict"},
    {"category": "Imports", "prompt": "import logging", "target": "as", "target_bpe": "Ġas"},

    # --- 7. Class Names (12) ---
    {"category": "Class names", "prompt": "class", "target": "Base", "target_bpe": "ĠBase"},
    {"category": "Class names", "prompt": "class", "target": "Model", "target_bpe": "ĠModel"},
    {"category": "Class names", "prompt": "class", "target": "Config", "target_bpe": "ĠConfig"},
    {"category": "Class names", "prompt": "class", "target": "Trainer", "target_bpe": "ĠTrainer"},
    {"category": "Class names", "prompt": "class", "target": "User", "target_bpe": "ĠUser"},
    {"category": "Class names", "prompt": "class", "target": "Dataset", "target_bpe": "ĠDataset"},
    {"category": "Class names", "prompt": "class", "target": "Engine", "target_bpe": "ĠEngine"},
    {"category": "Class names", "prompt": "class", "target": "Handler", "target_bpe": "ĠHandler"},
    {"category": "Class names", "prompt": "class", "target": "Session", "target_bpe": "ĠSession"},
    {"category": "Class names", "prompt": "class", "target": "Exception", "target_bpe": "ĠException"},
    {"category": "Class names", "prompt": "class", "target": "Node", "target_bpe": "ĠNode"},
    {"category": "Class names", "prompt": "class", "target": "Server", "target_bpe": "ĠServer"},

    # --- 8. Attribute Names (13) ---
    {"category": "Attribute names", "prompt": "self.", "target": "name", "target_bpe": "name"},
    {"category": "Attribute names", "prompt": "self.", "target": "value", "target_bpe": "value"},
    {"category": "Attribute names", "prompt": "self.", "target": "config", "target_bpe": "config"},
    {"category": "Attribute names", "prompt": "self.", "target": "device", "target_bpe": "device"},
    {"category": "Attribute names", "prompt": "self.", "target": "logger", "target_bpe": "logger"},
    {"category": "Attribute names", "prompt": "self.", "target": "state", "target_bpe": "state"},
    {"category": "Attribute names", "prompt": "obj.", "target": "data", "target_bpe": "data"},
    {"category": "Attribute names", "prompt": "req.", "target": "json", "target_bpe": "json"},
    {"category": "Attribute names", "prompt": "path.", "target": "exists", "target_bpe": "exists"},
    {"category": "Attribute names", "prompt": "res.", "target": "status_code", "target_bpe": "status_code"},
    {"category": "Attribute names", "prompt": "torch.", "target": "cuda", "target_bpe": "cuda"},
    {"category": "Attribute names", "prompt": "os.", "target": "path", "target_bpe": "path"},
    {"category": "Attribute names", "prompt": "sys.", "target": "path", "target_bpe": "path"},
]


def run_mlx_probes(model, tokenizer, mask_token_id, model_name: str):
    """Evaluates 100 contextual probes and outputs structured summary metrics."""
    import mlx.core as mx
    import mlx.nn as nn

    print(f"\n{'=' * 115}")
    print(f" 100 CONTEXTUAL PROBES SUITE: {model_name}")
    print(f"{'=' * 115}")
    print(f"{'CATEGORY':<20} | {'PROMPT':<28} | {'TARGET':<8} | {'RANK':<6} | {'P(target)':<9} | {'CE(-logP)':<9} | {'TOP 3 PREDICTED'}")
    print("-" * 115)

    results = []

    for probe in PROBE_SUITE_100:
        prompt_ids = tokenizer.encode(probe["prompt"]).ids
        seq_ids = prompt_ids + [mask_token_id]
        seq_mx = mx.array([seq_ids], dtype=mx.int32)

        logits = model(seq_mx)
        raw = logits[0, -1].tolist()
        raw[mask_token_id] = -1e9
        probs_mx = nn.softmax(mx.array(raw), axis=-1)
        mx.eval(probs_mx)
        probs = probs_mx.tolist()

        target_id = tokenizer.token_to_id(probe["target_bpe"])
        if target_id is None:
            target_id = tokenizer.token_to_id(probe["target"])

        if target_id is not None:
            sorted_ids = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
            rank = sorted_ids.index(target_id) + 1
            p_target = max(probs[target_id], 1e-9)
            ce = -math.log(p_target)
        else:
            rank = 9999
            p_target = 0.0
            ce = 15.0

        top3_ids = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:3]
        top3_tokens = [tokenizer.id_to_token(i).replace("Ġ", " ") for i in top3_ids]
        top3_fmt = ", ".join([f"'{t}'" for t in top3_tokens])

        results.append({
            "category": probe["category"],
            "prompt": probe["prompt"],
            "target": probe["target"],
            "rank": rank,
            "p_target": p_target,
            "ce": ce
        })

        clean_prompt = probe["prompt"].replace("\n", "\\n")
        print(f"{probe['category']:<20} | {clean_prompt:<28} | {probe['target']:<8} | #{rank:<5} | {p_target:.5f}  | {ce:.4f}   | {top3_fmt}")

    # --- AGGREGATE SUMMARY REPORT ---
    print("\n" + "=" * 80)
    print(f" AGGREGATE PROBE SUMMARY REPORT — {model_name}")
    print("=" * 80)

    ranks = [r["rank"] for r in results]
    ces = [r["ce"] for r in results]
    top1_acc = sum(1 for r in ranks if r == 1) / len(ranks) * 100.0
    top5_acc = sum(1 for r in ranks if r <= 5) / len(ranks) * 100.0

    print(f"  Total Probes Evaluated : {len(results)}")
    print(f"  Overall Average Target CE : {np.mean(ces):.4f}")
    print(f"  Overall Average Rank      : {np.mean(ranks):.1f}")
    print(f"  Top-1 Accuracy            : {top1_acc:.2f}% ({sum(1 for r in ranks if r == 1)}/100)")
    print(f"  Top-5 Accuracy            : {top5_acc:.2f}% ({sum(1 for r in ranks if r <= 5)}/100)")
    print("-" * 80)

    # Per Category Breakdown
    print("  Category-Specific Breakdown:")
    categories = sorted(list(set(r["category"] for r in results)))
    for cat in categories:
        cat_items = [r for r in results if r["category"] == cat]
        cat_ce = np.mean([r["ce"] for r in cat_items])
        cat_rank = np.mean([r["rank"] for r in cat_items])
        cat_top5 = sum(1 for r in cat_items if r["rank"] <= 5) / len(cat_items) * 100.0
        print(f"    - {cat:<22} : Target CE = {cat_ce:.4f} | Avg Rank = {cat_rank:>5.1f} | Top-5 Acc = {cat_top5:.1f}%")

    print("=" * 80 + "\n")


def sample_mlx_model(ckpt_dir: Path, num_steps: int = 64, mode: str = "sample"):
    """Qualitative code completion sampling and/or probes for Apple MLX models."""
    import mlx.core as mx
    import mlx.nn as nn
    from scripts.train_mlx import MLXTelosTransformer

    weights_path = ckpt_dir / "model.safetensors"
    if not weights_path.exists():
        print(f"Error: Weights {weights_path} not found.")
        return

    tok_path = Path("configs/tokenizer_mac.json")
    if not tok_path.exists():
        tok_path = Path("configs/tokenizer_0.json")
    tokenizer = load_tokenizer(str(tok_path))

    model = MLXTelosTransformer(vocab_size=8192, d_model=512, n_layers=8, n_heads=8, n_kv_heads=8)
    model.load_weights(str(weights_path))
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())

    mask_token_id = tokenizer.token_to_id("[MASK]") or 4

    if mode in ("probes", "full"):
        run_mlx_probes(model, tokenizer, mask_token_id, ckpt_dir.name)

    if mode in ("sample", "full"):
        prompts = [
            "def add(a: int, b: int) -> int:\n    return a +",
            "import os\n\ndef get_current_dir():\n    return os.path.dirname(",
            "class User:\n    def __init__(self, username: str):\n        self.username = username\n\n    def get_name(self):\n        return self.",
            "def binary_search(arr, target):",
            "class UserSession:"
        ]

        print(f"\n{'=' * 70}")
        print(f" QUALITATIVE CODE COMPLETION SAMPLING (MLX): {ckpt_dir.name}")
        print(f"{'=' * 70}")

        for temp in [0.0, 0.5]:
            print(f"\n--- Temperature = {temp} ---")
            for idx, prompt_text in enumerate(prompts):
                prompt_ids = tokenizer.encode(prompt_text).ids
                gen_len = len(prompt_ids) + 10

                seq = np.full((1, gen_len), mask_token_id, dtype=np.int32)
                seq[0, :len(prompt_ids)] = prompt_ids
                seq_mx = mx.array(seq)
                prompt_mask = mx.zeros((1, gen_len), dtype=mx.bool_)
                prompt_mask[:, :len(prompt_ids)] = True

                for step in range(num_steps):
                    logits = model(seq_mx)
                    if temp > 0.0:
                        probs = nn.softmax(logits / temp, axis=-1)
                        sampled = mx.random.categorical(mx.log(probs + 1e-10))
                    else:
                        sampled = mx.argmax(logits, axis=-1)
                    seq_mx = mx.where(prompt_mask, seq_mx, sampled)

                mx.eval(seq_mx)
                completed_text = tokenizer.decode(seq_mx[0].tolist())
                clean_text = completed_text.replace("\n", "\\n")
                print(f"[Prompt #{idx+1}]: '{prompt_text}'")
                print(f"  -> Completion: '{clean_text}'\n")


def main():
    parser = argparse.ArgumentParser(description="Sample code completions from télos model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/phase_b_25m_1to1_mlx",
                        help="Path to checkpoint dir (MLX safetensors) or .pt file (PyTorch)")
    parser.add_argument("--steps", type=int, default=64, help="Number of denoising steps")
    parser.add_argument("--mode", type=str, choices=["sample", "probes", "full"], default="probes",
                        help="Evaluation mode: 'sample' for completions, 'probes' for 100 contextual probes, 'full' for both")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)

    if ckpt_path.is_dir() or ckpt_path.suffix == ".safetensors":
        target_dir = ckpt_path if ckpt_path.is_dir() else ckpt_path.parent
        sample_mlx_model(target_dir, num_steps=args.steps, mode=args.mode)
    else:
        from telos.hub.inference import TelosModel
        print(f"Loading {args.checkpoint}...")
        model_obj = TelosModel.from_pretrained(args.checkpoint)
        print(f"Running generations on {model_obj.device}...")
        run_qualitative_evaluation(model_obj.model, model_obj.tokenizer, num_steps=args.steps, device=model_obj.device)


if __name__ == "__main__":
    main()
