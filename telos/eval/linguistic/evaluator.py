"""
English Linguistic Benchmarking & Evaluation Engine for Télos.

Supports:
1. English Deterministic Probes:
   - 100 contextual probes across 5 linguistic categories (Grammar, Collocations, Connectives, Knowledge, Suffix-Infill).
   - Evaluates Top-1 Accuracy, Top-5 Accuracy, Mean Cross-Entropy, Average Target Rank, and 95% Bootstrap CI.
2. English Perplexity Evaluation:
   - Measures token-level cross-entropy and perplexity (PPL) on high-quality English evaluation text.
3. Qualitative Linguistic Sampling:
   - Open-ended English generation sampling with greedy decoding.
4. Paradigm Differentiation:
   - Strictly distinguishes causal AR models from bidirectional infilling models (COROSred, MDLM, UNDLM).
"""

import math
import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from .probes_en import load_english_probes, ENGLISH_PROBES_100
from telos.eval.stats import bootstrap_confidence_interval

# Curated high-quality representative English evaluation passages for perplexity benchmarking
ENGLISH_PERPLEXITY_PASSAGES: List[str] = [
    (
        "Natural language processing has undergone a profound revolution with the advent of "
        "large-scale neural language models. These models learn syntactic structures, semantic associations, "
        "and factual world knowledge by predicting masked or subsequent tokens across diverse textual corpora."
    ),
    (
        "The scientific method relies on systematic observation, measurement, and experiment, as well as the "
        "formulation, testing, and modification of hypotheses. Peer review serves as a cornerstone of rigorous "
        "inquiry, ensuring that published findings withstand independent scrutiny and replication."
    ),
    (
        "Throughout history, architecture has reflected both functional engineering requirements and artistic "
        "cultural expressions. From classical Greco-Roman columns to contemporary high-rise skyscrapers with "
        "energy-efficient glass facades, the built environment embodies the technological aspirations of society."
    ),
    (
        "The biological cell is the fundamental structural and functional unit of all known living organisms. "
        "Within eukaryotic cells, specialized organelles such as the nucleus and mitochondria coordinate metabolic "
        "pathways, cellular respiration, and the transcription of genetic instructions encoded in DNA."
    ),
]


def _evaluate_single_linguistic_probe_type(
    model,
    tokenizer,
    backend: str,
    probes_list: List[Dict[str, Any]],
    probe_type: str = "infill",
    mask_token_id: int = 1
) -> Dict[str, Any]:
    """
    Evaluates a single pass (infill or causal next-token) over English linguistic probes.
    
    Args:
        model: MLX or PyTorch transformer model.
        tokenizer: BPE Tokenizer instance.
        backend: 'mlx' or 'pytorch'.
        probes_list: List of probe dictionaries.
        probe_type: 'infill' or 'causal'.
        mask_token_id: Special token ID for [MASK] (default: 1).
    """
    results = []
    category_stats: Dict[str, Dict[str, Any]] = {}

    for probe in probes_list:
        cat = probe.get("category", "General")
        prompt = probe["prompt"]
        target_str = probe["target"]

        if cat not in category_stats:
            category_stats[cat] = {"count": 0, "top1": 0, "top5": 0, "ce": [], "rank": []}

        # Context-aware tokenization handling leading-space byte ('Ġ')
        p_ids = tokenizer.encode(prompt).ids
        target_bpe = probe.get("target_bpe", "")
        target_tok = None
        if target_bpe:
            target_tok = tokenizer.token_to_id(target_bpe)

        if target_tok is None:
            sep = " " if ("Ġ" in target_bpe and not prompt.endswith(" ")) else ""
            full_text = prompt + sep + target_str
            full_ids = tokenizer.encode(full_text).ids
            if len(full_ids) > len(p_ids):
                target_tok = full_ids[len(p_ids)]
            else:
                target_ids = tokenizer.encode(target_str).ids
                target_tok = target_ids[0] if target_ids else 0

        # Construct input sequence based on probe type
        if probe_type == "infill":
            suffix = probe.get("suffix", "")
            s_ids = tokenizer.encode(suffix).ids if suffix else []
            # Bidirectional input: prefix + [MASK] + suffix
            input_ids = list(p_ids) + [mask_token_id] + s_ids
            eval_idx = len(p_ids)
            mask_override = False
        else:
            # Causal continuation input: prefix only, evaluate at last token position
            input_ids = list(p_ids)
            eval_idx = len(p_ids) - 1
            mask_override = True

        # Forward pass through model backend
        if backend == "mlx":
            import mlx.core as mx
            x = mx.array([input_ids], dtype=mx.int32)
            logits = model(x, mask_override=mask_override)
            logits_pos = np.array(logits[0, eval_idx].astype(mx.float32))
        else:
            import torch
            device = next(model.parameters()).device if hasattr(model, "parameters") else "cpu"
            x = torch.tensor([input_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                logits = model(x, mask_override=mask_override)
            logits_pos = logits[0, eval_idx].detach().cpu().numpy()

        if probe_type == "infill":
            # Mask out the [MASK] token itself to prevent degenerate identity prediction
            logits_pos[mask_token_id] = -1e9

        # Numerically stable softmax probability distribution
        shifted = logits_pos - np.max(logits_pos)
        probs = np.exp(shifted) / np.sum(np.exp(shifted))

        target_prob = max(float(probs[target_tok]), 1e-12)
        target_ce = -math.log(target_prob)

        sorted_indices = np.argsort(logits_pos)[::-1]
        rank_matches = np.where(sorted_indices == target_tok)[0]
        rank = int(rank_matches[0]) + 1 if len(rank_matches) > 0 else len(logits_pos)
        is_top1 = (rank == 1)
        is_top5 = (rank <= 5)

        category_stats[cat]["count"] += 1
        if is_top1:
            category_stats[cat]["top1"] += 1
        if is_top5:
            category_stats[cat]["top5"] += 1
        category_stats[cat]["ce"].append(target_ce)
        category_stats[cat]["rank"].append(rank)

        results.append({
            "id": probe.get("id"),
            "category": cat,
            "prompt": prompt,
            "target": target_str,
            "rank": rank,
            "target_ce": round(target_ce, 3),
            "top1": is_top1,
            "top5": is_top5,
        })

    total_count = len(results)
    overall_top1 = sum(r["top1"] for r in results) / max(total_count, 1) * 100.0
    overall_top5 = sum(r["top5"] for r in results) / max(total_count, 1) * 100.0
    overall_ce = float(np.mean([r["target_ce"] for r in results])) if results else 0.0
    overall_rank = float(np.mean([r["rank"] for r in results])) if results else 0.0

    ci_top1 = bootstrap_confidence_interval([1.0 if r["top1"] else 0.0 for r in results])

    cat_breakdown = {}
    for cat, s in category_stats.items():
        cnt = s["count"]
        top1_pct = (s["top1"] / cnt) * 100.0 if cnt else 0.0
        top5_pct = (s["top5"] / cnt) * 100.0 if cnt else 0.0
        avg_r = float(np.mean(s["rank"])) if cnt else 0.0
        avg_ce = float(np.mean(s["ce"])) if cnt else 0.0
        cat_breakdown[cat] = {
            "count": cnt,
            "top1_pct": round(top1_pct, 1),
            "top5_pct": round(top5_pct, 1),
            "avg_rank": round(avg_r, 1),
            "avg_ce": round(avg_ce, 2)
        }

    title = "BIDIRECTIONAL INFILLING PROBES" if probe_type == "infill" else "CAUSAL CONTINUATION PROBES"
    print("\n" + "=" * 88)
    print(f"  TÉLOS ENGLISH LINGUISTIC REPORT: {title} ({total_count} PROBES)")
    print("=" * 88)
    print(f"  {'Category':<38} | {'Count':<5} | {'Top-1 (%)':<9} | {'Top-5 (%)':<9} | {'Avg Rank':<8} | {'Avg CE':<6}")
    print("-" * 88)

    for cat, s in cat_breakdown.items():
        print(f"  {cat:<38} | {s['count']:<5d} | {s['top1_pct']:>8.1f}% | {s['top5_pct']:>8.1f}% | {s['avg_rank']:>8.1f} | {s['avg_ce']:>6.2f}")

    print("-" * 88)
    print(f"  {'OVERALL LINGUISTIC SUMMARY':<38} | {total_count:<5d} | {overall_top1:>8.1f}% | {overall_top5:>8.1f}% | {overall_rank:>8.1f} | {overall_ce:>6.2f}")
    print(f"  95% Bootstrap CI for Top-1 Accuracy: [{ci_top1[0]}%, {ci_top1[1]}%]")
    print("=" * 88 + "\n")

    return {
        "probe_type": probe_type,
        "overall": {
            "top1_acc_pct": round(overall_top1, 2),
            "top1_95ci": ci_top1,
            "top5_acc_pct": round(overall_top5, 2),
            "mean_rank": round(overall_rank, 1),
            "mean_ce": round(overall_ce, 2),
            "total_probes": total_count,
        },
        "categories": cat_breakdown,
        "probes": results,
    }


def evaluate_english_perplexity(
    model,
    tokenizer,
    backend: str,
    passages: Optional[List[str]] = None,
    max_seq_len: int = 512
) -> Dict[str, Any]:
    """
    Evaluates token-level cross-entropy loss and perplexity on natural English text passages.
    """
    passages = passages or ENGLISH_PERPLEXITY_PASSAGES
    total_log_likelihood = 0.0
    total_tokens = 0
    passage_results = []

    for idx, text in enumerate(passages, 1):
        token_ids = tokenizer.encode(text).ids
        if len(token_ids) <= 1:
            continue
        token_ids = token_ids[:max_seq_len]
        seq_len = len(token_ids)

        if backend == "mlx":
            import mlx.core as mx
            x = mx.array([token_ids], dtype=mx.int32)
            logits = model(x, mask_override=True)
            logits_np = np.array(logits[0].astype(mx.float32))
        else:
            import torch
            device = next(model.parameters()).device if hasattr(model, "parameters") else "cpu"
            x = torch.tensor([token_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                logits = model(x, mask_override=True)
            logits_np = logits[0].detach().cpu().numpy()

        # Shift logits and targets for causal next-token prediction
        # Target for token t is token_ids[t+1]
        shift_logits = logits_np[:-1, :]
        shift_targets = np.array(token_ids[1:], dtype=np.int64)

        # Compute cross-entropy per position using numerically stable log-softmax
        max_logits = np.max(shift_logits, axis=-1, keepdims=True)
        exp_logits = np.exp(shift_logits - max_logits)
        log_probs = shift_logits - max_logits - np.log(np.sum(exp_logits, axis=-1, keepdims=True))

        target_log_probs = log_probs[np.arange(len(shift_targets)), shift_targets]
        passage_loss = -float(np.mean(target_log_probs))
        passage_ppl = math.exp(min(passage_loss, 20.0))

        total_log_likelihood += float(np.sum(target_log_probs))
        total_tokens += len(shift_targets)

        passage_results.append({
            "passage_id": idx,
            "token_count": len(shift_targets),
            "loss_ce": round(passage_loss, 3),
            "perplexity": round(passage_ppl, 2)
        })

    avg_loss = (-total_log_likelihood / max(total_tokens, 1)) if total_tokens > 0 else 0.0
    overall_ppl = math.exp(min(avg_loss, 20.0)) if total_tokens > 0 else 0.0

    print("\n" + "=" * 80)
    print("  TÉLOS ENGLISH LINGUISTIC BENCHMARK: PERPLEXITY (PPL)")
    print("=" * 80)
    print(f"  Total Passages Evaluated: {len(passages)}")
    print(f"  Total Evaluated Tokens:   {total_tokens}")
    print(f"  Mean Cross-Entropy Loss:  {avg_loss:.3f} nats/token")
    print(f"  Overall Perplexity (PPL): {overall_ppl:.2f}")
    print("=" * 80 + "\n")

    return {
        "overall_ppl": round(overall_ppl, 2),
        "mean_loss_ce": round(avg_loss, 3),
        "total_tokens": total_tokens,
        "passages": passage_results,
    }


def evaluate_english_sample(
    model,
    tokenizer,
    backend: str,
    prompts: Optional[List[str]] = None,
    max_new_tokens: int = 48
) -> Dict[str, Any]:
    """Generates qualitative text completions for representative English prompts."""
    default_prompts = [
        "In modern physics, the theory of general relativity describes gravity as",
        "The primary advantage of solar and wind renewable energy is that",
        "When traveling to a foreign country, it is polite to learn basic phrases like",
    ]
    prompts = prompts or default_prompts
    samples = []

    print("\n" + "=" * 80)
    print("  TÉLOS QUALITATIVE ENGLISH GENERATION SAMPLING")
    print("=" * 80)

    for i, p in enumerate(prompts, 1):
        p_ids = tokenizer.encode(p).ids
        curr_ids = list(p_ids)
        stop_set = {0, 3}  # EOS / PAD

        for _ in range(max_new_tokens):
            if backend == "mlx":
                import mlx.core as mx
                x = mx.array([curr_ids], dtype=mx.int32)
                logits = model(x)
                next_tok = int(np.argmax(np.array(logits[0, -1].astype(mx.float32))))
            else:
                import torch
                device = next(model.parameters()).device if hasattr(model, "parameters") else "cpu"
                x = torch.tensor([curr_ids], dtype=torch.long, device=device)
                with torch.no_grad():
                    logits = model(x)
                next_tok = int(torch.argmax(logits[0, -1]).item())

            if next_tok in stop_set:
                break
            curr_ids.append(next_tok)

        continuation = tokenizer.decode(curr_ids[len(p_ids):])
        full_text = p + continuation
        print(f"\n--- [English Prompt {i}] ---\n{p}")
        print(f"--- [Continuation] ---\n{full_text}\n")
        samples.append({"prompt": p, "continuation": continuation, "full_text": full_text})

    print("=" * 80 + "\n")
    return {"samples": samples}


def evaluate_linguistic(
    model,
    tokenizer,
    backend: str,
    paradigm: str = "corosred",
    language: str = "english",
    mode: str = "probes",
    num_probes: int = 100,
    probe_type: str = "both",
    mask_token_id: int = 1,
    **kwargs
) -> Dict[str, Any]:
    """
    Master evaluator entrypoint for linguistic benchmarks.
    
    Args:
        model: Loaded model instance.
        tokenizer: Tokenizer instance.
        backend: 'mlx' or 'pytorch'.
        paradigm: 'corosred', 'ar', 'mdlm', 'undlm'.
        language: Language target (default: 'english').
        mode: 'probes', 'perplexity', 'sample', or 'full'.
        num_probes: Number of contextual probes to evaluate.
        probe_type: 'infill', 'causal', or 'both'.
        mask_token_id: Token ID for [MASK].
        
    Returns:
        Comprehensive evaluation report dictionary.
    """
    report: Dict[str, Any] = {
        "benchmark_type": "linguistic",
        "language": language,
        "mode": mode,
        "timestamp": int(time.time()),
    }

    if language.lower() != "english":
        raise NotImplementedError(f"Linguistic benchmark for language '{language}' is not yet supported. Available: 'english'.")

    probes_list = load_english_probes(num_probes)

    if mode in ["probes", "full"]:
        probes_report: Dict[str, Any] = {}

        # 1. Causal next-token continuation pass across all models
        if probe_type in ["causal", "both"]:
            causal_probes = [p for p in probes_list if p.get("mode", "both") in ["both", "causal"]]
            res_causal = _evaluate_single_linguistic_probe_type(
                model=model,
                tokenizer=tokenizer,
                backend=backend,
                probes_list=causal_probes,
                probe_type="causal",
                mask_token_id=mask_token_id
            )
            probes_report["causal"] = res_causal

        # 2. Bidirectional infilling pass
        # Causal AR models cannot condition on trailing English suffixes; marked N/A.
        if probe_type in ["infill", "both"]:
            if str(paradigm).lower() == "ar":
                probes_report["infill"] = {
                    "status": "not_applicable",
                    "reason": "AR models are causal-only and cannot condition on bidirectional suffixes",
                    "overall": {
                        "top1_acc_pct": None,
                        "top5_acc_pct": None,
                        "mean_ce": None,
                        "mean_rank": None,
                        "total_probes": 0,
                    },
                    "categories": {}
                }
            else:
                res_infill = _evaluate_single_linguistic_probe_type(
                    model=model,
                    tokenizer=tokenizer,
                    backend=backend,
                    probes_list=probes_list,
                    probe_type="infill",
                    mask_token_id=mask_token_id
                )
                probes_report["infill"] = res_infill

        report["probes"] = probes_report

    if mode in ["perplexity", "full"]:
        report["perplexity"] = evaluate_english_perplexity(model, tokenizer, backend)

    if mode in ["sample", "full"]:
        report["sample"] = evaluate_english_sample(model, tokenizer, backend)

    return report
