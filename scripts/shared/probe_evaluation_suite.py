"""
Comprehensive Tri-Paradigm Probe Evaluation Suite for télos.

Evaluates AR, MDLM, UNDLM models across all token ratios (12.5M and 25M scales)
with distribution-aligned probe protocols and generates publication-quality figures.

Protocols:
  - AR: Causal next-token prediction (no padding needed)
  - MDLM: 512-token [MASK]-padded sequences (matching training distribution)
  - UNDLM: 512-token uniform-random-padded sequences, marginalized over K=32 draws

Outputs:
  - JSON results in evals/probe_results/
  - Publication figures in figures/
"""

import sys
import math
import ast
import json
import time
import difflib
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlx.core as mx
import mlx.nn as nn
from mdiff.model.mlx_components import MLXTelosTransformer
from ar.model.mlx_components import MLXCausalTransformer
from mdiff.data.tokenizer import load_tokenizer
from mdiff.diffusion.sampler import MLXMDLMSampler
from undiff.diffusion.sampler import UNDLMSampler
from notebooks.shared.Evaluation import PROBE_SUITE_100

# ============================================================================
# MULTI-TOKEN ITERATIVE GENERATION PROBES (25 probes, 5 categories)
# ============================================================================
MULTI_TOKEN_PROBES_25 = [
    # 1. Function Return Expressions (5)
    {"category": "Function Returns", "prompt": "def add(a: int, b: int) -> int:\n    return ", "target": "a + b"},
    {"category": "Function Returns", "prompt": "def get_length(s: str) -> int:\n    return ", "target": "len(s)"},
    {"category": "Function Returns", "prompt": "def is_even(n: int) -> bool:\n    return ", "target": "n % 2 == 0"},
    {"category": "Function Returns", "prompt": "def square(x: float) -> float:\n    return ", "target": "x ** 2"},
    {"category": "Function Returns", "prompt": "def join_words(words: list[str]) -> str:\n    return ", "target": "' '.join(words)"},
    # 2. Conditionals & Logic (5)
    {"category": "Conditionals & Logic", "prompt": "if x is None:\n    ", "target": "return False"},
    {"category": "Conditionals & Logic", "prompt": "if not os.path.exists(path):\n    ", "target": "raise FileNotFoundError(path)"},
    {"category": "Conditionals & Logic", "prompt": "if a > b:\n    max_val = a\nelse:\n    ", "target": "max_val = b"},
    {"category": "Conditionals & Logic", "prompt": "if count >= 10:\n    ", "target": "break"},
    {"category": "Conditionals & Logic", "prompt": "if item not in visited:\n    ", "target": "visited.add(item)"},
    # 3. Loop & Iteration Constructs (5)
    {"category": "Loops & Iterations", "prompt": "total = 0\nfor x in values:\n    ", "target": "total += x"},
    {"category": "Loops & Iterations", "prompt": "result = []\nfor item in collection:\n    ", "target": "result.append(item)"},
    {"category": "Loops & Iterations", "prompt": "for i in range(len(arr)):\n    ", "target": "print(arr[i])"},
    {"category": "Loops & Iterations", "prompt": "while queue:\n    node = ", "target": "queue.pop(0)"},
    {"category": "Loops & Iterations", "prompt": "squares = [", "target": "x ** 2 for x in nums]"},
    # 4. Class Methods & Attributes (5)
    {"category": "Class Attributes", "prompt": "class Point:\n    def __init__(self, x: float, y: float):\n        ", "target": "self.x = x\n        self.y = y"},
    {"category": "Class Attributes", "prompt": "class User:\n    def get_name(self) -> str:\n        return ", "target": "self.name"},
    {"category": "Class Attributes", "prompt": "class Counter:\n    def increment(self):\n        ", "target": "self.count += 1"},
    {"category": "Class Attributes", "prompt": "class Stack:\n    def pop(self):\n        return ", "target": "self.items.pop()"},
    {"category": "Class Attributes", "prompt": "class Node:\n    def __init__(self, val):\n        self.val = val\n        ", "target": "self.next = None"},
    # 5. Imports & Exceptions (5)
    {"category": "Imports & Exceptions", "prompt": "import json\nimport ", "target": "os"},
    {"category": "Imports & Exceptions", "prompt": "from pathlib import ", "target": "Path"},
    {"category": "Imports & Exceptions", "prompt": "try:\n    value = int(text)\nexcept ", "target": "ValueError:\n    value = 0"},
    {"category": "Imports & Exceptions", "prompt": "with open(filepath, 'r') as f:\n    ", "target": "content = f.read()"},
    {"category": "Imports & Exceptions", "prompt": "from typing import ", "target": "List, Dict, Optional"},
]


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_tokenizer_global():
    """Loads the shared tokenizer used across all paradigms."""
    tok_path = PROJECT_ROOT / "configs" / "tokenizer_mac.json"
    if not tok_path.exists():
        tok_path = PROJECT_ROOT / "configs" / "tokenizer_0.json"
    return load_tokenizer(str(tok_path))


def load_model(paradigm: str, ckpt_dir: Path):
    """Loads a model checkpoint for a given paradigm.
    
    Args:
        paradigm: 'ar', 'masked', or 'uniform'
        ckpt_dir: Path to checkpoint directory containing model.safetensors + config.json
    
    Returns:
        Loaded MLX model in bfloat16
    """
    weights_path = ckpt_dir / "model.safetensors"
    config_path = ckpt_dir / "config.json"
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    
    with open(config_path) as f:
        cfg = json.load(f)
    
    if paradigm == "ar":
        model = MLXCausalTransformer(**cfg)
    else:
        # Both MDLM and UNDLM use the same bidirectional transformer architecture
        model = MLXTelosTransformer(**cfg)
    
    model.load_weights(str(weights_path), strict=False)
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    return model


def discover_checkpoints():
    """Auto-discovers all available model checkpoints in the checkpoints/ tree.
    
    Returns:
        dict: {scale: {paradigm: {ratio: ckpt_path}}}
              e.g. {"12m": {"ar": {1: Path(...), 5: Path(...), ...}, "masked": {...}, "uniform": {...}},
                     "25m": {...}}
    """
    ckpt_root = PROJECT_ROOT / "checkpoints"
    result = {}
    
    for paradigm in ["ar", "masked", "uniform"]:
        paradigm_dir = ckpt_root / paradigm
        if not paradigm_dir.exists():
            continue
        for scale_dir in sorted(paradigm_dir.iterdir()):
            if not scale_dir.is_dir() or scale_dir.name in ("old",):
                continue
            scale = scale_dir.name  # "12m" or "25m"
            if scale not in result:
                result[scale] = {}
            if paradigm not in result[scale]:
                result[scale][paradigm] = {}
            
            for ckpt_dir in sorted(scale_dir.iterdir()):
                if not ckpt_dir.is_dir() or ckpt_dir.name in ("old",):
                    continue
                # Extract ratio from directory name like "telos_12m_r10"
                name = ckpt_dir.name
                if "_r" in name:
                    try:
                        ratio = int(name.split("_r")[-1])
                        if (ckpt_dir / "model.safetensors").exists():
                            result[scale][paradigm][ratio] = ckpt_dir
                    except ValueError:
                        continue
    
    return result


# ============================================================================
# SINGLE-STEP DISTRIBUTION-ALIGNED PROBE EVALUATION
# ============================================================================

def evaluate_single_step(model, tokenizer, paradigm: str):
    """Evaluates 100 single-step contextual probes with distribution-aligned protocols.
    
    Protocols:
      - AR: Forward prompt_ids, read logits at position -1
      - MDLM: Pad to 512 with [MASK], read logits at target_pos (right after prompt)
      - UNDLM: Pad to 512 with random tokens, marginalize over K=32 noise draws
    
    Returns:
        dict with overall_ce, overall_rank, top1_acc, top5_acc, categories
    """
    vocab_size = 8192
    mask_token_id = tokenizer.token_to_id("[MASK]") or 1
    SEQ_LEN = 512

    ce_list, rank_list, top1_list, top5_list = [], [], [], []
    cat_metrics = {}

    for probe in PROBE_SUITE_100:
        prompt_ids = tokenizer.encode(probe["prompt"]).ids
        target_pos = len(prompt_ids)  # Position of the target token

        if paradigm == "uniform":
            # UNDLM: pad to 512 with random tokens, marginalize K=32 draws
            K = 32
            all_probs = []
            for _ in range(K):
                seq = list(prompt_ids) + [int(x) for x in np.random.randint(0, vocab_size, size=SEQ_LEN - len(prompt_ids))]
                seq = seq[:SEQ_LEN]
                logits = model(mx.array([seq], dtype=mx.int32))
                p = nn.softmax(logits[0, target_pos].astype(mx.float32), axis=-1)
                mx.eval(p)
                all_probs.append(p)
            probs_mx = mx.mean(mx.stack(all_probs), axis=0)
            mx.eval(probs_mx)
            probs = probs_mx.tolist()

        elif paradigm == "ar":
            # AR: causal, no padding needed
            seq_mx = mx.array([prompt_ids], dtype=mx.int32)
            logits = model(seq_mx)
            raw = logits[0, -1].astype(mx.float32).tolist()
            probs_mx = nn.softmax(mx.array(raw), axis=-1)
            mx.eval(probs_mx)
            probs = probs_mx.tolist()

        else:
            # MDLM: pad to 512 with [MASK] tokens
            seq_ids = list(prompt_ids) + [mask_token_id] * (SEQ_LEN - len(prompt_ids))
            seq_ids = seq_ids[:SEQ_LEN]
            seq_mx = mx.array([seq_ids], dtype=mx.int32)
            logits = model(seq_mx)
            raw = logits[0, target_pos].astype(mx.float32).tolist()
            raw[mask_token_id] = -1e9
            probs_mx = nn.softmax(mx.array(raw), axis=-1)
            mx.eval(probs_mx)
            probs = probs_mx.tolist()

        # Resolve target token ID (handles BPE leading-space tokens)
        target_ids_space = tokenizer.encode(" " + probe["target"]).ids
        target_id = target_ids_space[0] if len(target_ids_space) > 0 else tokenizer.token_to_id(probe.get("target_bpe", probe["target"]))
        if target_id is None:
            target_id = tokenizer.token_to_id(probe["target"])

        # Compute rank and CE
        sorted_ids = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
        if target_id is not None and target_id < len(probs):
            rank = sorted_ids.index(target_id) + 1
            p_target = max(probs[target_id], 1e-9)
            ce = -math.log(p_target)
        else:
            rank = 9999
            ce = 15.0

        ce_list.append(ce)
        rank_list.append(rank)
        top1_list.append(1.0 if rank == 1 else 0.0)
        top5_list.append(1.0 if rank <= 5 else 0.0)

        # Per-category tracking
        cat = probe["category"]
        if cat not in cat_metrics:
            cat_metrics[cat] = {"ce": [], "rank": [], "top5": []}
        cat_metrics[cat]["ce"].append(ce)
        cat_metrics[cat]["rank"].append(rank)
        cat_metrics[cat]["top5"].append(1.0 if rank <= 5 else 0.0)

    # Build category summaries
    summary_cats = {}
    for cat, vals in cat_metrics.items():
        summary_cats[cat] = {
            "ce": float(np.mean(vals["ce"])),
            "rank": float(np.mean(vals["rank"])),
            "top5": float(np.mean(vals["top5"]) * 100.0)
        }

    return {
        "overall_ce": float(np.mean(ce_list)),
        "overall_rank": float(np.mean(rank_list)),
        "top1_acc": float(np.mean(top1_list) * 100.0),
        "top5_acc": float(np.mean(top5_list) * 100.0),
        "categories": summary_cats
    }


# ============================================================================
# MULTI-STEP ITERATIVE GENERATION PROBE EVALUATION
# ============================================================================

def check_syntax(code: str) -> bool:
    """Tests if code parses as valid Python (with indentation fallback)."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        try:
            ast.parse("def _dummy_fn():\n" + "\n".join("    " + l for l in code.split("\n")))
            return True
        except SyntaxError:
            return False


def edit_similarity(s1: str, s2: str) -> float:
    """Character-level normalized sequence matcher similarity in [0, 1]."""
    return difflib.SequenceMatcher(None, s1, s2).ratio()


def evaluate_multi_step(model, tokenizer, paradigm: str, num_steps: int = 32):
    """Evaluates 25 multi-token iterative code generation probes.
    
    Metrics:
      - Exact Match: generated tokens exactly match target tokens
      - Token Accuracy: fraction of correctly generated tokens  
      - AST Validity: generated code parses as valid Python
      - Edit Similarity: character-level similarity to target
    """
    mask_token_id = tokenizer.token_to_id("[MASK]") or 1
    vocab_size = 8192

    # Create samplers for diffusion paradigms
    if paradigm == "masked":
        sampler = MLXMDLMSampler(model, mask_token_id=mask_token_id, num_steps=num_steps, schedule="cosine")
    elif paradigm == "uniform":
        sampler = UNDLMSampler(model, vocab_size=vocab_size, num_steps=num_steps, mode="self_correction")

    exact_matches, token_accs, syntax_valid_list, edit_sims = [], [], [], []

    for probe in MULTI_TOKEN_PROBES_25:
        prompt_ids = tokenizer.encode(probe["prompt"]).ids
        target_ids = tokenizer.encode(probe["target"]).ids
        target_len = len(target_ids)
        total_len = len(prompt_ids) + target_len

        if paradigm == "ar":
            # Autoregressive greedy generation
            seq = mx.array([prompt_ids], dtype=mx.int32)
            generated = []
            for _ in range(target_len):
                logits = model(seq)
                next_token = mx.argmax(logits[0, -1], axis=-1)
                mx.eval(next_token)
                generated.append(next_token.item())
                seq = mx.concatenate([seq, next_token.reshape(1, 1)], axis=1)
            gen_ids = generated

        elif paradigm == "masked":
            # MDLM: confidence-based iterative unmasking
            prompt_mx = mx.array([prompt_ids], dtype=mx.int32)
            output = sampler.sample(total_len, prompt_ids=prompt_mx)
            mx.eval(output)
            gen_ids = output[0, len(prompt_ids):total_len].tolist()

        else:
            # UNDLM: self-correcting iterative denoising
            prompt_mx = mx.array([prompt_ids], dtype=mx.int32)
            output = sampler.sample(total_len, prompt_ids=prompt_mx)
            mx.eval(output)
            gen_ids = output[0, len(prompt_ids):total_len].tolist()

        # Pad/truncate generated ids to match target length
        gen_ids = gen_ids[:target_len]
        while len(gen_ids) < target_len:
            gen_ids.append(0)

        # Compute metrics
        exact = gen_ids == target_ids
        tok_acc = sum(1 for g, t in zip(gen_ids, target_ids) if g == t) / max(len(target_ids), 1)
        gen_text = tokenizer.decode(gen_ids)
        full_code = probe["prompt"] + gen_text
        syntax_ok = check_syntax(full_code)
        sim = edit_similarity(gen_text, probe["target"])

        exact_matches.append(1.0 if exact else 0.0)
        token_accs.append(tok_acc)
        syntax_valid_list.append(1.0 if syntax_ok else 0.0)
        edit_sims.append(sim)

    return {
        "exact_match": float(np.mean(exact_matches) * 100.0),
        "token_accuracy": float(np.mean(token_accs) * 100.0),
        "ast_validity": float(np.mean(syntax_valid_list) * 100.0),
        "edit_similarity": float(np.mean(edit_sims) * 100.0)
    }


# ============================================================================
# FIGURE GENERATION
# ============================================================================

def generate_figures(all_results: dict, output_dir: Path):
    """Generates publication-quality matplotlib figures from evaluation results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    output_dir.mkdir(parents=True, exist_ok=True)

    # Color palette: consistent paradigm colors across all figures
    PARADIGM_COLORS = {"ar": "#2196F3", "masked": "#FF5722", "uniform": "#4CAF50"}
    PARADIGM_LABELS = {"ar": "AR (Autoregressive)", "masked": "MDLM (Masked Diffusion)", "uniform": "UNDLM (Uniform Noise DLM)"}
    PARADIGM_MARKERS = {"ar": "o", "masked": "s", "uniform": "D"}

    # Global matplotlib style
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    })

    for scale in sorted(all_results.keys()):
        scale_data = all_results[scale]
        
        # ====================================================================
        # FIGURE 1: Single-Step Scaling Trajectory (CE, Rank, Top-5)
        # ====================================================================
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        fig.suptitle(f"télos {scale.upper()} — Single-Step Probe Scaling Trajectory", fontsize=14, fontweight="bold", y=1.02)

        metrics = [
            ("overall_ce", "Target Cross-Entropy (nats)", "lower is better"),
            ("overall_rank", "Average Target Rank", "lower is better"),
            ("top5_acc", "Top-5 Accuracy (%)", "higher is better"),
        ]

        for ax_idx, (metric_key, ylabel, note) in enumerate(metrics):
            ax = axes[ax_idx]
            for paradigm in ["ar", "masked", "uniform"]:
                if paradigm not in scale_data:
                    continue
                ratios = sorted(scale_data[paradigm].keys())
                vals = [scale_data[paradigm][r]["single_step"][metric_key] for r in ratios]
                ax.plot(ratios, vals,
                        marker=PARADIGM_MARKERS[paradigm], markersize=7, linewidth=2.2,
                        color=PARADIGM_COLORS[paradigm], label=PARADIGM_LABELS[paradigm])

            ax.set_xlabel("Token Ratio (1:N)", fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(f"{ylabel}\n({note})", fontsize=10)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            if ax_idx == 0:
                ax.legend(fontsize=9, loc="best")

        plt.tight_layout()
        fig_path = output_dir / f"{scale}_single_step_trajectory.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [Saved] {fig_path}")

        # ====================================================================
        # FIGURE 2: Multi-Step Generation Trajectory
        # ====================================================================
        fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
        fig.suptitle(f"télos {scale.upper()} — Multi-Step Iterative Generation Probes", fontsize=14, fontweight="bold", y=1.02)

        gen_metrics = [
            ("exact_match", "Exact Match (%)"),
            ("token_accuracy", "Token Accuracy (%)"),
            ("ast_validity", "AST Validity (%)"),
            ("edit_similarity", "Edit Similarity (%)"),
        ]

        for ax_idx, (metric_key, ylabel) in enumerate(gen_metrics):
            ax = axes[ax_idx]
            for paradigm in ["ar", "masked", "uniform"]:
                if paradigm not in scale_data:
                    continue
                ratios = sorted(scale_data[paradigm].keys())
                vals = [scale_data[paradigm][r]["multi_step"][metric_key] for r in ratios]
                ax.plot(ratios, vals,
                        marker=PARADIGM_MARKERS[paradigm], markersize=7, linewidth=2.2,
                        color=PARADIGM_COLORS[paradigm], label=PARADIGM_LABELS[paradigm])

            ax.set_xlabel("Token Ratio (1:N)", fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(ylabel, fontsize=10)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            if ax_idx == 0:
                ax.legend(fontsize=9, loc="best")

        plt.tight_layout()
        fig_path = output_dir / f"{scale}_multi_step_trajectory.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [Saved] {fig_path}")

        # ====================================================================
        # FIGURE 3: Category Breakdown Heatmap (Single-Step CE by Category)
        # ====================================================================
        paradigms_present = [p for p in ["ar", "masked", "uniform"] if p in scale_data]
        if paradigms_present:
            # Collect all categories across all models for this scale
            all_cats = set()
            for paradigm in paradigms_present:
                for ratio in scale_data[paradigm]:
                    all_cats.update(scale_data[paradigm][ratio]["single_step"]["categories"].keys())
            all_cats = sorted(all_cats)

            # Build model labels and CE matrix
            model_labels = []
            ce_matrix = []
            for paradigm in paradigms_present:
                for ratio in sorted(scale_data[paradigm].keys()):
                    label = f"{paradigm.upper()[:4]} 1:{ratio}"
                    model_labels.append(label)
                    cats = scale_data[paradigm][ratio]["single_step"]["categories"]
                    row = [cats.get(c, {}).get("ce", 12.0) for c in all_cats]
                    ce_matrix.append(row)

            ce_arr = np.array(ce_matrix)
            if ce_arr.size > 0:
                fig, ax = plt.subplots(figsize=(max(12, len(all_cats) * 1.5), max(6, len(model_labels) * 0.5)))
                im = ax.imshow(ce_arr, cmap="RdYlGn_r", aspect="auto", vmin=3, vmax=10)

                ax.set_xticks(range(len(all_cats)))
                ax.set_xticklabels(all_cats, rotation=45, ha="right", fontsize=9)
                ax.set_yticks(range(len(model_labels)))
                ax.set_yticklabels(model_labels, fontsize=9)

                # Annotate cells with CE values
                for i in range(len(model_labels)):
                    for j in range(len(all_cats)):
                        ax.text(j, i, f"{ce_arr[i, j]:.1f}", ha="center", va="center", fontsize=8,
                                color="white" if ce_arr[i, j] > 7 else "black")

                plt.colorbar(im, ax=ax, label="Target CE (nats)", shrink=0.8)
                ax.set_title(f"télos {scale.upper()} — Per-Category Target CE Heatmap", fontsize=13, fontweight="bold")
                plt.tight_layout()
                fig_path = output_dir / f"{scale}_category_heatmap.png"
                fig.savefig(fig_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"  [Saved] {fig_path}")

    # ====================================================================
    # FIGURE 4: Cross-Scale Comparison (12.5M vs 25M at matching ratios)
    # ====================================================================
    scales = sorted(all_results.keys())
    if len(scales) >= 2:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        fig.suptitle("télos — Parameter Scale Comparison (12.5M vs 25M)", fontsize=14, fontweight="bold", y=1.02)

        scale_styles = {"12m": "--", "25m": "-"}
        scale_alphas = {"12m": 0.6, "25m": 1.0}

        for ax_idx, (metric_key, ylabel, _) in enumerate(metrics):
            ax = axes[ax_idx]
            for paradigm in ["ar", "masked", "uniform"]:
                for scale in scales:
                    if paradigm not in all_results.get(scale, {}):
                        continue
                    ratios = sorted(all_results[scale][paradigm].keys())
                    vals = [all_results[scale][paradigm][r]["single_step"][metric_key] for r in ratios]
                    ax.plot(ratios, vals,
                            marker=PARADIGM_MARKERS[paradigm], markersize=6, linewidth=2,
                            linestyle=scale_styles.get(scale, "-"),
                            alpha=scale_alphas.get(scale, 1.0),
                            color=PARADIGM_COLORS[paradigm],
                            label=f"{PARADIGM_LABELS[paradigm]} ({scale.upper()})")

            ax.set_xlabel("Token Ratio (1:N)", fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            if ax_idx == 0:
                ax.legend(fontsize=7, loc="best", ncol=2)

        plt.tight_layout()
        fig_path = output_dir / "cross_scale_comparison.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [Saved] {fig_path}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    print("=" * 80)
    print("  télos COMPREHENSIVE TRI-PARADIGM PROBE EVALUATION SUITE")
    print("  Distribution-Aligned Protocols | Single-Step + Multi-Step")
    print("=" * 80)

    tokenizer = load_tokenizer_global()
    checkpoints = discover_checkpoints()

    print(f"\n  Discovered checkpoints:")
    for scale in sorted(checkpoints.keys()):
        for paradigm in sorted(checkpoints[scale].keys()):
            ratios = sorted(checkpoints[scale][paradigm].keys())
            print(f"    {scale.upper()} {paradigm}: ratios {ratios}")

    # Results container: {scale: {paradigm: {ratio: {single_step: {...}, multi_step: {...}}}}}
    all_results = {}
    total_models = sum(len(ratios) for scale in checkpoints.values() for ratios in scale.values())
    model_idx = 0

    for scale in sorted(checkpoints.keys()):
        all_results[scale] = {}
        for paradigm in sorted(checkpoints[scale].keys()):
            all_results[scale][paradigm] = {}
            for ratio in sorted(checkpoints[scale][paradigm].keys()):
                model_idx += 1
                ckpt_dir = checkpoints[scale][paradigm][ratio]
                model_tag = f"{scale.upper()} {paradigm.upper()} 1:{ratio}"
                print(f"\n  [{model_idx}/{total_models}] Evaluating {model_tag}...")
                print(f"    Checkpoint: {ckpt_dir}")

                t0 = time.time()
                model = load_model(paradigm, ckpt_dir)
                print(f"    Model loaded in {time.time() - t0:.1f}s")

                # Single-step probes
                t1 = time.time()
                ss_results = evaluate_single_step(model, tokenizer, paradigm)
                print(f"    Single-Step: CE={ss_results['overall_ce']:.4f} | Rank={ss_results['overall_rank']:.1f} | "
                      f"Top-1={ss_results['top1_acc']:.1f}% | Top-5={ss_results['top5_acc']:.1f}% ({time.time()-t1:.1f}s)")

                # Multi-step probes
                t2 = time.time()
                ms_results = evaluate_multi_step(model, tokenizer, paradigm, num_steps=32)
                print(f"    Multi-Step:  EM={ms_results['exact_match']:.1f}% | TokAcc={ms_results['token_accuracy']:.1f}% | "
                      f"AST={ms_results['ast_validity']:.1f}% | EditSim={ms_results['edit_similarity']:.1f}% ({time.time()-t2:.1f}s)")

                all_results[scale][paradigm][ratio] = {
                    "single_step": ss_results,
                    "multi_step": ms_results
                }

                # Free model memory before loading next
                del model
                mx.metal.clear_cache() if hasattr(mx, "metal") else None

    # Save raw JSON results
    results_dir = PROJECT_ROOT / "evals" / "probe_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "comprehensive_probe_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  [Saved JSON] {results_path}")

    # Generate figures
    figures_dir = PROJECT_ROOT / "figures"
    print(f"\n  Generating publication figures...")
    generate_figures(all_results, figures_dir)

    # Print final summary table
    print("\n" + "=" * 100)
    print("  FINAL SUMMARY TABLE — Single-Step Target CE")
    print("=" * 100)
    header = f"  {'Model':<25} | {'CE':>8} | {'Rank':>8} | {'Top-1':>6} | {'Top-5':>6}"
    print(header)
    print("  " + "-" * 95)

    for scale in sorted(all_results.keys()):
        for paradigm in sorted(all_results[scale].keys()):
            for ratio in sorted(all_results[scale][paradigm].keys()):
                ss = all_results[scale][paradigm][ratio]["single_step"]
                tag = f"{scale.upper()} {paradigm.upper()} 1:{ratio}"
                print(f"  {tag:<25} | {ss['overall_ce']:>8.4f} | {ss['overall_rank']:>8.1f} | "
                      f"{ss['top1_acc']:>5.1f}% | {ss['top5_acc']:>5.1f}%")

    print("=" * 100)
    print("  Evaluation complete.")


if __name__ == "__main__":
    main()
