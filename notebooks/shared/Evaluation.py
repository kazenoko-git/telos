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
3. Top 20 Logits Inspection Mode (--mode top20):
   - Inspects top 20 candidate predictions at [MASK] positions immediately following code prompts.
"""

import sys
import math
import argparse
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mdiff.data.tokenizer import load_tokenizer
from mdiff.eval.sample import run_qualitative_evaluation

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

def run_top20_logits_inspection(model, tokenizer, mask_token_id, model_name: str, top_k: int = 50):
    """Inspects top K candidate logits and probabilities at [MASK] positions following code prompts."""
    import mlx.core as mx
    import mlx.nn as nn

    test_prompts = [
        {"name": "return a +", "text": "def add(a: int, b: int) -> int:\n    return a +", "num_masks": 3},
        {"name": "os.path.dirname(", "text": "import os\n\ndef get_current_dir():\n    return os.path.dirname(", "num_masks": 3},
        {"name": "return self.", "text": "class User:\n    def __init__(self, username: str):\n        self.username = username\n\n    def get_name(self):\n        return self.", "num_masks": 3},
        {"name": "def binary_search(", "text": "def binary_search(arr, target):", "num_masks": 3},
    ]

    print(f"\n{'=' * 105}")
    print(f" TOP {top_k} LOGITS INSPECTION (MDLM FIRST [MASK] POSITION): {model_name}")
    print(f"{'=' * 105}")

    for p in test_prompts:
        prompt_ids = tokenizer.encode(p["text"]).ids
        seq_ids = prompt_ids + [mask_token_id] * p["num_masks"]
        seq_mx = mx.array([seq_ids], dtype=mx.int32)

        logits = model(seq_mx)
        # First [MASK] position is at index len(prompt_ids)
        mask_idx = len(prompt_ids)
        raw_logits_np = logits[0, mask_idx].tolist()
        
        # Suppress [MASK] token (token ID mask_token_id)
        raw_logits_np[mask_token_id] = -1e9

        probs_mx = nn.softmax(mx.array(raw_logits_np), axis=-1)
        mx.eval(probs_mx)
        probs_np = probs_mx.tolist()

        sorted_indices = sorted(range(len(probs_np)), key=lambda i: probs_np[i], reverse=True)[:top_k]

        clean_prompt = p["text"].replace("\n", "\\n")
        print(f"\nPrompt: '{clean_prompt}'")
        print(f"Inspecting 1st [MASK] position (Index {mask_idx}) | Top {top_k} Candidate Predictions:")
        print(f"{'RANK':<6} | {'TOKEN':<20} | {'TOKEN ID':<10} | {'RAW LOGIT':<12} | {'PROBABILITY P':<14} | {'CE (-log P)':<11}")
        print("-" * 90)

        for rank_idx, tid in enumerate(sorted_indices, start=1):
            tok_str = tokenizer.id_to_token(tid).replace("Ġ", " ")
            clean_tok = tok_str.replace("\n", "\\n")
            logit_val = raw_logits_np[tid]
            prob_val = probs_np[tid]
            ce_val = -math.log(max(prob_val, 1e-9))
            print(f"#{rank_idx:<5} | '{clean_tok}':<20 | {tid:<10} | {logit_val:<12.4f} | {prob_val:<14.5f} | {ce_val:<11.4f}")

def run_mlx_probes(model, tokenizer, mask_token_id, model_name: str, save_to_disk: bool = True):
    """Evaluates 100 contextual probes, prints all 100 probe rows, and outputs structured summary reports to disk."""
    import math
    from pathlib import Path
    import numpy as np
    import mlx.core as mx
    import mlx.nn as nn

    report_lines = []
    header_banner = "=" * 115
    title_line = f" 100 CONTEXTUAL PROBES SUITE: {model_name}"
    cols_header = f"{"CATEGORY":<20} | {"PROMPT":<28} | {"TARGET":<8} | {"RANK":<6} | {"P(target)":<9} | {"CE(-logP)":<9} | TOP 3 PREDICTED"
    divider = "-" * 115

    report_lines.extend(["", header_banner, title_line, header_banner, cols_header, divider])
    print(f"\n{header_banner}")
    print(title_line)
    print(header_banner)
    print(cols_header)
    print(divider)

    results = []
    is_undlm = "uniform" in model_name.lower() or "undlm" in model_name.lower()
    is_ar = "ar" in model_name.lower()
    vocab_size = getattr(model, "vocab_size", 8192)

    for probe in PROBE_SUITE_100:
        prompt_ids = tokenizer.encode(probe["prompt"]).ids

        if is_undlm:
            # Distribution-aligned protocol for UNDLM:
            # Pad to 512 tokens with uniform random vocab tokens (matching training distribution)
            # and marginalize over K=32 independent noise draws.
            K = 32
            SEQ_LEN = 512
            target_pos = len(prompt_ids)  # Position right after prompt
            all_probs = []
            for _ in range(K):
                # Build a 512-token sequence: prompt + random tokens filling the rest
                seq = list(prompt_ids) + [int(x) for x in np.random.randint(0, vocab_size, size=SEQ_LEN - len(prompt_ids))]
                seq = seq[:SEQ_LEN]  # Truncate if prompt is very long
                logits = model(mx.array([seq], dtype=mx.int32))
                p = nn.softmax(logits[0, target_pos].astype(mx.float32), axis=-1)
                mx.eval(p)
                all_probs.append(p)
            # Average probabilities across all noise draws
            probs_mx = mx.mean(mx.stack(all_probs), axis=0)
            mx.eval(probs_mx)
            probs = probs_mx.tolist()
        elif is_ar:
            # Autoregressive (AR) protocol: causal model, no padding needed.
            # The representation at position -1 of prompt_ids predicts the next token.
            seq_mx = mx.array([prompt_ids], dtype=mx.int32)
            logits = model(seq_mx)
            raw = logits[0, -1].astype(mx.float32).tolist()
            probs_mx = nn.softmax(mx.array(raw), axis=-1)
            mx.eval(probs_mx)
            probs = probs_mx.tolist()
        else:
            # Distribution-aligned protocol for MDLM:
            # Pad to 512 tokens with [MASK] tokens to match the training distribution.
            # During training, the model sees 512-token sequences with distributed masks.
            # A bare 5-token probe is out-of-distribution and causes misleading CE regression.
            SEQ_LEN = 512
            target_pos = len(prompt_ids)  # Position where [MASK] replaces the target
            # Build: prompt_ids + [MASK at target] + [MASK padding to 512]
            seq_ids = list(prompt_ids) + [mask_token_id] * (SEQ_LEN - len(prompt_ids))
            seq_ids = seq_ids[:SEQ_LEN]  # Truncate if prompt exceeds 512
            seq_mx = mx.array([seq_ids], dtype=mx.int32)

            logits = model(seq_mx)
            raw = logits[0, target_pos].astype(mx.float32).tolist()
            raw[mask_token_id] = -1e9  # Suppress [MASK] token from predictions
            probs_mx = nn.softmax(mx.array(raw), axis=-1)
            mx.eval(probs_mx)
            probs = probs_mx.tolist()

        # BPE leading-space token target resolution
        target_token_str = probe.get("target_bpe", probe["target"])
        target_ids_space = tokenizer.encode(" " + probe["target"]).ids
        target_id = target_ids_space[0] if len(target_ids_space) > 0 else tokenizer.token_to_id(target_token_str)
        if target_id is None:
            target_id = tokenizer.token_to_id(probe["target"])

        sorted_ids = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)

        if target_id is not None and target_id < len(probs):
            rank = sorted_ids.index(target_id) + 1
            p_target = max(probs[target_id], 1e-9)
            ce = -math.log(p_target)
        else:
            rank = 9999
            p_target = 0.0
            ce = 15.0

        top3_ids = sorted_ids[:3]
        top3_tokens = [f"\"{tokenizer.id_to_token(i).replace('Ġ', ' ')}\"({probs[i]:.3f})" for i in top3_ids]
        top3_fmt = ", ".join(top3_tokens)

        results.append({
            "category": probe["category"],
            "prompt": probe["prompt"],
            "target": probe["target"],
            "rank": rank,
            "p_target": p_target,
            "ce": ce
        })

        clean_prompt = probe["prompt"].replace("\n", "\\n")
        row_str = f"{probe['category']:<20} | {clean_prompt:<28} | {probe['target']:<8} | #{rank:<5} | {p_target:<9.5f} | {ce:<9.4f} | {top3_fmt}"
        print(row_str)
        report_lines.append(row_str)

    # --- AGGREGATE SUMMARY REPORT ---
    banner80 = "=" * 80
    div80 = "-" * 80
    sum_title = f" AGGREGATE PROBE SUMMARY REPORT — {model_name}"
    
    print("\n" + banner80)
    print(sum_title)
    print(banner80)
    report_lines.extend(["\n" + banner80, sum_title, banner80])

    ranks = [r["rank"] for r in results]
    ces = [r["ce"] for r in results]
    top1_acc = sum(1 for r in ranks if r == 1) / len(ranks)
    top5_acc = sum(1 for r in ranks if r <= 5) / len(ranks)

    l1 = f"  Total Probes Evaluated : {len(results)}"
    l2 = f"  Overall Average Target CE : {np.mean(ces):.4f}"
    l3 = f"  Overall Average Rank      : {np.mean(ranks):.1f}"
    l4 = f"  Top-1 Accuracy            : {top1_acc * 100.0:.2f}% ({sum(1 for r in ranks if r == 1)}/{len(ranks)})"
    l5 = f"  Top-5 Accuracy            : {top5_acc * 100.0:.2f}% ({sum(1 for r in ranks if r <= 5)}/{len(ranks)})"

    for l in [l1, l2, l3, l4, l5, div80]:
        print(l)
        report_lines.append(l)

    print("  Category-Specific Breakdown:")
    report_lines.append("  Category-Specific Breakdown:")
    categories = sorted(list(set(r["category"] for r in results)))
    for cat in categories:
        cat_items = [r for r in results if r["category"] == cat]
        cat_ce = np.mean([r["ce"] for r in cat_items])
        cat_rank = np.mean([r["rank"] for r in cat_items])
        cat_top5 = sum(1 for r in cat_items if r["rank"] <= 5) / len(cat_items)
        cat_line = f"    - {cat:<22} : Target CE = {cat_ce:.4f} | Avg Rank = {cat_rank:>5.1f} | Top-5 Acc = {cat_top5 * 100.0:.1f}%"
        print(cat_line)
        report_lines.append(cat_line)

    print(banner80 + "\n")
    report_lines.append(banner80 + "\n")

    if save_to_disk:
        out_dir = Path("evals/masked/probes") if Path("evals/masked/probes").exists() else Path("probes_output")
        out_dir.mkdir(parents=True, exist_ok=True)
        full_report_text = "\n".join(report_lines) + "\n"
        clean_name = model_name.replace(" ", "_").replace(":", "_")
        with open(out_dir / f"{clean_name}_probes.txt", "w") as f:
            f.write(full_report_text)
        with open(out_dir / "master_overnight_probes_summary.txt", "a") as f_sum:
            f_sum.write(full_report_text + "\n")
        print(f"  [Saved Probes Report] -> {out_dir}/{clean_name}_probes.txt")


def sample_mlx_model(ckpt_dir: Path, num_steps: int = 64, mode: str = "sample"):
    """Qualitative code completion sampling and/or probes for Apple MLX models."""
    import mlx.core as mx
    import mlx.nn as nn
    from mdiff.model.mlx_components import MLXTelosTransformer

    weights_path = ckpt_dir / "model.safetensors"
    if not weights_path.exists():
        print(f"Error: Weights {weights_path} not found.")
        return

    tok_path = Path("configs/tokenizer_mac.json")
    if not tok_path.exists():
        tok_path = Path("configs/tokenizer_0.json")
    tokenizer = load_tokenizer(str(tok_path))

    cfg_path = ckpt_dir / "config.json"
    if cfg_path.exists():
        import json
        with open(cfg_path) as f:
            cfg = json.load(f)
        model_kwargs = {
            "vocab_size": cfg.get("vocab_size", 8192),
            "d_model": cfg.get("d_model", 512),
            "n_layers": cfg.get("n_layers", 8),
            "n_heads": cfg.get("n_heads", 8),
            "n_kv_heads": cfg.get("n_kv_heads", 8)
        }
    else:
        model_kwargs = {"vocab_size": 8192, "d_model": 512, "n_layers": 8, "n_heads": 8, "n_kv_heads": 8}

    is_ar = "/ar/" in str(ckpt_dir).lower() or ckpt_dir.name.startswith("ar_")
    if is_ar:
        from ar.model.mlx_components import MLXCausalTransformer
        model = MLXCausalTransformer(**model_kwargs)
    else:
        model = MLXTelosTransformer(**model_kwargs)

    model.load_weights(str(weights_path), strict=False)
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())

    mask_token_id = tokenizer.token_to_id("[MASK]") or 4

    if mode in ("top20", "top50", "full"):
        top_k = 50 if mode in ("top50", "full") else 20
        run_top20_logits_inspection(model, tokenizer, mask_token_id, ckpt_dir.name, top_k=top_k)

    if mode in ("probes", "full"):
        run_mlx_probes(model, tokenizer, mask_token_id, f"{ckpt_dir.parent.parent.name}_{ckpt_dir.name}")

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

def run_prompt_rank_matrix(model_dict: dict, save_to_disk: bool = True):
    """Evaluates all 101 probes across all loaded models and generates per-prompt average rank comparison table."""
    import math
    from pathlib import Path
    import numpy as np
    import mlx.core as mx
    import mlx.nn as nn

    model_names = list(model_dict.keys())
    # Header
    header = f"{"CAT":<18} | {"PROMPT":<26} | {"TARGET":<8}"
    for m_name in model_names:
        header += f" | {m_name[:12]:<12}"
    header += " | AVG RANK"
    
    lines = ["=" * 135, " PER-PROMPT TARGET RANK MATRIX & ACROSS-MODEL AVERAGES", "=" * 135, header, "-" * 135]
    print("\n" + lines[0])
    print(lines[1])
    print(lines[2])
    print(lines[3])
    print(lines[4])

    prompt_stats = []
    
    for idx, probe in enumerate(PROBE_SUITE_100):
        ranks = []
        row_str = f"{probe['category']:<18} | {probe['prompt'].replace('\n', '\\n'):<26} | {probe['target']:<8}"
        
        for m_name in model_names:
            model, tokenizer, mask_id = model_dict[m_name]
            prompt_ids = tokenizer.encode(probe["prompt"]).ids
            seq_ids = prompt_ids + [mask_id]
            logits = model(mx.array([seq_ids], dtype=mx.int32))
            raw = logits[0, -1].tolist()
            raw[mask_id] = -1e9
            probs_mx = nn.softmax(mx.array(raw), axis=-1)
            mx.eval(probs_mx)
            probs = probs_mx.tolist()

            target_token_str = probe.get("target_bpe", probe["target"])
            target_ids_space = tokenizer.encode(" " + probe["target"]).ids
            target_id = target_ids_space[0] if len(target_ids_space) > 0 else tokenizer.token_to_id(target_token_str)
            if target_id is None:
                target_id = tokenizer.token_to_id(probe["target"])

            sorted_ids = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
            if target_id is not None and target_id < len(probs):
                rank = sorted_ids.index(target_id) + 1
            else:
                rank = 9999
            ranks.append(rank)
            row_str += f" | #{rank:<11}"
        
        avg_r = np.mean(ranks)
        row_str += f" | #{avg_r:<8.1f}"
        print(row_str)
        lines.append(row_str)
        prompt_stats.append({"probe": probe, "ranks": ranks, "avg_rank": avg_r})

    # Overall Category Average Ranks
    lines.extend(["-" * 135, " CATEGORY-LEVEL AVERAGE RANKS ACROSS ALL MODELS:", "-" * 135])
    print("\n" + lines[-2])
    print(lines[-1])
    
    cats = sorted(list(set(p["probe"]["category"] for p in prompt_stats)))
    for cat in cats:
        cat_prompts = [p for p in prompt_stats if p["probe"]["category"] == cat]
        cat_avg_rank = np.mean([p["avg_rank"] for p in cat_prompts])
        cat_line = f"  - {cat:<24} : Across-Model Average Rank = #{cat_avg_rank:.1f}"
        print(cat_line)
        report_lines = cat_line
        lines.append(cat_line)

    lines.append("=" * 135 + "\n")
    print("=" * 135 + "\n")

    if save_to_disk:
        out_dir = Path("evals/masked/probes")
        out_dir.mkdir(parents=True, exist_ok=True)
        full_txt = "\n".join(lines) + "\n"
        with open(out_dir / "master_prompt_rank_matrix.txt", "w") as f:
            f.write(full_txt)
        print(f"  [Saved Matrix] -> {out_dir}/master_prompt_rank_matrix.txt")


def main():
    parser = argparse.ArgumentParser(description="Sample code completions from télos model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/phase_b_25m_1to1_mlx",
                        help="Path to checkpoint dir (MLX safetensors) or .pt file (PyTorch)")
    parser.add_argument("--steps", type=int, default=64, help="Number of denoising steps")
    parser.add_argument("--mode", type=str, choices=["sample", "probes", "top20", "top50", "full"], default="probes",
                        help="Evaluation mode: 'sample', 'probes', 'top20', 'top50' for top 50 logits inspection, 'full'")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)

    if ckpt_path.is_dir() or ckpt_path.suffix == ".safetensors":
        target_dir = ckpt_path if ckpt_path.is_dir() else ckpt_path.parent
        sample_mlx_model(target_dir, num_steps=args.steps, mode=args.mode)
    else:
        from mdiff.hub.inference import TelosModel
        print(f"Loading {args.checkpoint}...")
        model_obj = TelosModel.from_pretrained(args.checkpoint)
        print(f"Running generations on {model_obj.device}...")
        run_qualitative_evaluation(model_obj.model, model_obj.tokenizer, num_steps=args.steps, device=model_obj.device)

if __name__ == "__main__":
    main()

def run_comprehensive_benchmark_suite(save_to_disk: bool = True):
    """Runs the complete TELOS MDLM Probes Benchmark Suite:
    - Category Mean, Median, and StdDev metrics
    - Overall Model Leaderboard (CE, Rank, Top-K %, Wins)
    - Best Model per Prompt & Relative Improvement (25M 1:1 -> Best)
    - Visual ASCII Heatmap Matrix
    - TELOS PROBES SCORECARD
    """
    import math
    import json
    from pathlib import Path
    import numpy as np
    import mlx.core as mx
    import mlx.nn as nn
    from mdiff.model.mlx_components import MLXTelosTransformer
    from mdiff.data.tokenizer import load_tokenizer

    models_to_eval = [
        ("25M 1:1", "checkpoints/phase_b_25m_1to1_mlx"),
        ("25M 1:5", "checkpoints/phase_b_25m_1to5_mlx"),
        ("25M 1:10", "checkpoints/phase_b_25m_1to10_mlx"),
        ("25M 1:15", "checkpoints/phase_b_25m_1to15_mlx"),
        ("25M 1:20", "checkpoints/phase_b_25m_1to20_mlx"),
        ("25M 1:25", "checkpoints/phase_b_25m_1to25_mlx"),
        ("50M 1:1", "checkpoints/phase_b_50m_1to1_mlx"),
        ("50M 1:10", "checkpoints/phase_b_50m_1to10_mlx"),
        ("50M 1:20", "checkpoints/phase_b_50m_1to20_mlx"),
        ("50M 1:25", "checkpoints/phase_b_50m_1to25_mlx"),
    ]

    model_data = {}
    for name, ckpt_str in models_to_eval:
        p = Path(ckpt_str)
        if not (p / "model.safetensors").exists():
            continue
        tok = load_tokenizer(str(p / "tokenizer.json"))
        with open(p / "config.json") as f: cfg = json.load(f)
        m = MLXTelosTransformer(**cfg)
        m.load_weights(str(p / "model.safetensors"))
        m.set_dtype(mx.bfloat16)
        mask_id = tok.token_to_id("[MASK]") or 1

        probe_results = []
        for probe in PROBE_SUITE_100:
            prompt_ids = tok.encode(probe["prompt"]).ids
            seq_ids = prompt_ids + [mask_id]
            logits = m(mx.array([seq_ids], dtype=mx.int32))
            raw = logits[0, -1].tolist()
            raw[mask_id] = -1e9
            probs_mx = nn.softmax(mx.array(raw), axis=-1)
            mx.eval(probs_mx)
            probs = probs_mx.tolist()

            target_token_str = probe.get("target_bpe", probe["target"])
            target_ids_space = tok.encode(" " + probe["target"]).ids
            target_id = target_ids_space[0] if len(target_ids_space) > 0 else tok.token_to_id(target_token_str)
            if target_id is None:
                target_id = tok.token_to_id(probe["target"])

            sorted_ids = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
            if target_id is not None and target_id < len(probs):
                rank = sorted_ids.index(target_id) + 1
                p_target = max(probs[target_id], 1e-9)
                ce = -math.log(p_target)
            else:
                rank = 9999
                p_target = 0.0
                ce = 15.0

            probe_results.append({
                "probe": probe,
                "rank": rank,
                "p_target": p_target,
                "ce": ce,
                "top1": rank == 1,
                "top5": rank <= 5,
                "top10": rank <= 10,
                "top50": rank <= 50,
                "top100": rank <= 100
            })
        model_data[name] = probe_results

    model_names = list(model_data.keys())
    n_probes = len(PROBE_SUITE_100)
    lines = []

    wins = {m: 0 for m in model_names}
    prompt_metrics = []

    for p_idx in range(n_probes):
        probe = PROBE_SUITE_100[p_idx]
        ranks_by_model = {m: model_data[m][p_idx]["rank"] for m in model_names}
        ces_by_model = {m: model_data[m][p_idx]["ce"] for m in model_names}
        
        best_m = min(model_names, key=lambda m: (ces_by_model[m], ranks_by_model[m]))
        wins[best_m] += 1
        
        base_rank = ranks_by_model.get("25M 1:1", 9999)
        best_rank = ranks_by_model[best_m]
        
        if base_rank > best_rank:
            pct_imp = (base_rank - best_rank) / base_rank * 100.0
            ratio_imp = base_rank / max(best_rank, 1)
            imp_str = f"{base_rank} -> {best_rank} (+{pct_imp:.0f}%, {ratio_imp:.2f}x)"
        elif base_rank == best_rank:
            pct_imp = 0.0
            ratio_imp = 1.0
            imp_str = f"{base_rank} (Same)"
        else:
            pct_imp = (base_rank - best_rank) / base_rank * 100.0
            ratio_imp = base_rank / best_rank
            imp_str = f"{base_rank} -> {best_rank} ({pct_imp:.0f}%)"

        prompt_metrics.append({
            "probe": probe,
            "best_model": best_m,
            "best_rank": best_rank,
            "best_ce": ces_by_model[best_m],
            "base_rank": base_rank,
            "imp_pct": pct_imp,
            "imp_ratio": ratio_imp,
            "imp_str": imp_str,
            "ranks": ranks_by_model,
            "ces": ces_by_model,
            "avg_rank": np.mean(list(ranks_by_model.values())),
            "med_rank": np.median(list(ranks_by_model.values())),
            "std_rank": np.std(list(ranks_by_model.values())),
            "avg_ce": np.mean(list(ces_by_model.values())),
            "med_ce": np.median(list(ces_by_model.values())),
            "std_ce": np.std(list(ces_by_model.values())),
        })

    # 1. Overall Leaderboard
    leaderboard = []
    for m in model_names:
        m_ranks = [r["rank"] for r in model_data[m]]
        m_ces = [r["ce"] for r in model_data[m]]
        leaderboard.append({
            "model": m,
            "wins": wins[m],
            "mean_ce": np.mean(m_ces),
            "med_ce": np.median(m_ces),
            "std_ce": np.std(m_ces),
            "mean_rank": np.mean(m_ranks),
            "med_rank": np.median(m_ranks),
            "std_rank": np.std(m_ranks),
            "top1": sum(1 for r in model_data[m] if r["top1"]) / n_probes,
            "top5": sum(1 for r in model_data[m] if r["top5"]) / n_probes,
            "top10": sum(1 for r in model_data[m] if r["top10"]) / n_probes,
            "top50": sum(1 for r in model_data[m] if r["top50"]) / n_probes,
            "top100": sum(1 for r in model_data[m] if r["top100"]) / n_probes,
        })
    leaderboard.sort(key=lambda x: (x["mean_ce"], x["mean_rank"]))

    b80 = "=" * 135
    l_head = f"{"POS":<4} | {"MODEL":<10} | {"WINS":<5} | {"MEAN CE":<8} | {"MED CE":<8} | {"STD CE":<8} | {"MEAN RNK":<9} | {"MED RNK":<8} | {"STD RNK":<8} | {"TOP1%":<5} | {"TOP5%":<5} | {"TOP10%":<6} | {"TOP100%":<7}"
    lines.extend([b80, " OVERALL MODEL LEADERBOARD (Primary Metric: Target Cross Entropy | Secondary Metric: Mean Rank)", b80, l_head, "-" * 135])
    
    for pos, l in enumerate(leaderboard, 1):
        m_name = l["model"]
        w = l["wins"]
        mce, medce, stdce = l["mean_ce"], l["med_ce"], l["std_ce"]
        mr, medr, stdr = l["mean_rank"], l["med_rank"], l["std_rank"]
        t1, t5, t10, t100 = l["top1"], l["top5"], l["top10"], l["top100"]
        row = f"#{pos:<3} | {m_name:<10} | {w:<5} | {mce:<8.4f} | {medce:<8.4f} | {stdce:<8.4f} | {mr:<9.1f} | {medr:<8.1f} | {stdr:<8.1f} | {t1 * 100.0:<5.1f}% | {t5 * 100.0:<5.1f}% | {t10 * 100.0:<6.1f}% | {t100 * 100.0:<7.1f}%"
        lines.append(row)
    lines.append(b80 + "\n")

    # 2. Category Stability & Performance (Mean, Median, StdDev)
    cats = sorted(list(set(p["category"] for p in PROBE_SUITE_100)))
    lines.extend([b80, " CATEGORY STABILITY & PERFORMANCE BREAKDOWN (Mean, Median, StdDev across Prompts)", b80])
    cat_head = f"{"CATEGORY":<22} | {"MEAN CE":<8} | {"MED CE":<8} | {"STD CE":<8} | {"MEAN RNK":<9} | {"MED RNK":<8} | {"STD RNK":<8} | {"TOP10%":<6} | {"TOP100%":<7}"
    lines.extend([cat_head, "-" * 135])

    category_bests = {}
    for cat in cats:
        lines.append(f" === {cat} ===")
        cat_model_scores = {}
        for m in model_names:
            c_items = [model_data[m][i] for i in range(n_probes) if PROBE_SUITE_100[i]["category"] == cat]
            ces = [it["ce"] for it in c_items]
            ranks = [it["rank"] for it in c_items]
            t10 = sum(1 for it in c_items if it["top10"]) / len(c_items)
            t100 = sum(1 for it in c_items if it["top100"]) / len(c_items)
            cat_model_scores[m] = np.mean(ces)
            crow = f"  {m:<20} | {np.mean(ces):<8.4f} | {np.median(ces):<8.4f} | {np.std(ces):<8.4f} | {np.mean(ranks):<9.1f} | {np.median(ranks):<8.1f} | {np.std(ranks):<8.1f} | {t10 * 100:<6.1f}% | {t100 * 100:<7.1f}%"
            lines.append(crow)
        category_bests[cat] = min(cat_model_scores.keys(), key=lambda k: cat_model_scores[k])
        lines.append("-" * 135)

    # 3. Prompt-by-Prompt Best Model, Relative Improvement & Heatmap Matrix
    mat_head = f"{"CAT":<18} | {"PROMPT":<24} | {"TGT":<6} | {"BEST MODEL":<10} | {"IMPROVEMENT (25M 1:1 -> BEST)":<32} | HEATMAP (Ranks: [1:1, 1:5, 1:10, 1:15, 1:20, 1:25 | 50M 1:1..])"
    lines.extend([b80, " DETAILED PROMPT MATRIX & HEATMAP", b80, mat_head, "-" * 135])

    for pm in prompt_metrics:
        pr = pm["probe"]
        h_cells = []
        for m in model_names:
            rk = pm["ranks"][m]
            if rk <= 10:
                h_cells.append(f"[G:{rk:>2}]")
            elif rk <= 100:
                h_cells.append(f"[Y:{rk:>3}]")
            else:
                h_cells.append(f"[R:{rk:>4}]")
        h_str = " ".join(h_cells)
        p_clean = pr["prompt"].replace("\n", "\\n")
        p_row = f"{pr['category']:<18} | {p_clean:<24} | {pr['target']:<6} | {pm['best_model']:<10} | {pm['imp_str']:<32} | {h_str}"
        lines.append(p_row)
    lines.append(b80 + "\n")

    # 4. TELOS PROBES SCORECARD
    best_overall = leaderboard[0]["model"]
    best_25m = [l for l in leaderboard if "25M" in l["model"]][0]["model"]
    hardest_p = max(prompt_metrics, key=lambda x: (x["avg_ce"], x["avg_rank"]))
    easiest_p = min(prompt_metrics, key=lambda x: (x["avg_ce"], x["avg_rank"]))
    most_imp_p = max(prompt_metrics, key=lambda x: x["imp_pct"])
    regress_prompts = [x for x in prompt_metrics if x["imp_pct"] < 0]
    worst_reg_p = min(prompt_metrics, key=lambda x: x["imp_pct"]) if len(regress_prompts) > 0 else None

    hardest_prompt_str = hardest_p["probe"]["prompt"].replace("\n", "\\n")
    easiest_prompt_str = easiest_p["probe"]["prompt"].replace("\n", "\\n")
    most_imp_prompt_str = most_imp_p["probe"]["prompt"].replace("\n", "\\n")
    worst_reg_prompt_str = worst_reg_p["probe"]["prompt"].replace("\n", "\\n") if worst_reg_p else "None"

    sc_lines = [
        "=" * 80,
        "        TELOS PROBES SCORECARD",
        "=" * 80,
        f"  Best overall checkpoint : 🥇 {best_overall}",
        f"  Best 25M checkpoint     : 🥇 {best_25m}",
        "",
        "  Category Leaders:",
    ]
    for cat, b_m in category_bests.items():
        sc_lines.append(f"    - Best {cat:<18} : {b_m}")
    sc_lines.extend([
        "",
        f"  Hardest Prompt          : \"{hardest_prompt_str}\" -> target \"{hardest_p['probe']['target']}\" (Avg CE = {hardest_p['avg_ce']:.4f}, Avg Rank = #{hardest_p['avg_rank']:.1f})",
        f"  Easiest Prompt          : \"{easiest_prompt_str}\" -> target \"{easiest_p['probe']['target']}\" (Avg CE = {easiest_p['avg_ce']:.4f}, Avg Rank = #{easiest_p['avg_rank']:.1f})",
        f"  Most Improved Prompt    : \"{most_imp_prompt_str}\" ({most_imp_p['imp_str']})",
        f"  Largest Regression      : \"{worst_reg_prompt_str}\" ({worst_reg_p['imp_str']})" if worst_reg_p else "  Largest Regression      : None",
        "=" * 80 + "\n"
    ])
    lines.extend(sc_lines)

    full_txt = "\n".join(lines) + "\n"
    print(full_txt)

    if save_to_disk:
        out_dir = Path("evals/masked/probes") if Path("evals/masked/probes").exists() else Path("probes_output")
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "master_comprehensive_scorecard_and_benchmark.txt", "w") as f:
            f.write(full_txt)
        with open(out_dir / "TELOS_PROBES_SCORECARD.txt", "w") as f_sc:
            f_sc.write("\n".join(sc_lines) + "\n")
        print(f"  [Saved Comprehensive Benchmark Report] -> {out_dir}/master_comprehensive_scorecard_and_benchmark.txt")
        print(f"  [Saved Scorecard Report]               -> {out_dir}/TELOS_PROBES_SCORECARD.txt")

