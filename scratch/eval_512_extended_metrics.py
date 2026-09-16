"""
Comprehensive 512-Task Extended Metrics Suite for Télos Models:
1. Repetition Rates (2-gram, 3-gram, 4-gram, line-level, degenerate loops).
2. Generation Length & Early-Exit Profiles (token counts, trivial exits vs max token hits).
3. Numerical & Mathematical Literal Accuracy (specifically for Algorithms & Numerical):
   - Recall of prompt/docstring constants
   - Precision of generated numbers
   - Numeric hallucination rates
   - Zero-number emission rates
4. Category-level breakdowns across all 8 categories for 50M, 75M, and 100M.
"""

import json
import re
import time
from pathlib import Path
from collections import Counter
import numpy as np
import torch

from telos.eval.runner import load_model_from_checkpoint, _generate_greedy_completion, clean_functional_completion
from telos.data.tokenizer import load_tokenizer

MODELS = [
    ("50M COROSred (CUDA, Lightning)", "checkpoints/corosred/50m_lightning/checkpoint_final.pt"),
    ("75M COROSred (TPU)", "checkpoints/corosred/unified/75m_python/checkpoint_final.pt"),
    ("100M COROSred (TPU, alpha=0.5)", "checkpoints/corosred/unified/100m_5b/checkpoint_final.pt"),
]

def compute_ngram_repetition(tokens: list[int], n: int) -> float:
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    unique = len(set(ngrams))
    return 1.0 - (unique / len(ngrams))

def detect_degenerate_loop(tokens: list[int], n: int = 3, min_repeats: int = 3) -> bool:
    if len(tokens) < n * min_repeats:
        return False
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    streak = 1
    for i in range(1, len(ngrams)):
        if ngrams[i] == ngrams[i-1]:
            streak += 1
            if streak >= min_repeats:
                return True
        else:
            streak = 1
    return False

def compute_line_repetition(text: str) -> tuple[float, bool]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) <= 1:
        return 0.0, False
    has_consec = any(lines[i] == lines[i-1] for i in range(1, len(lines)))
    unique = len(set(lines))
    rep_rate = 1.0 - (unique / len(lines))
    return rep_rate, has_consec

def extract_numbers(text: str) -> list[str]:
    # Match integer literals and float literals
    return re.findall(r'\b\d+(?:\.\d+)?\b', text)

print("=" * 80)
print("  TÉLOS 512-TASK EXTENDED EVALUATION: REPETITION & NUMERICAL ACCURACY")
print("=" * 80)

tok = load_tokenizer("configs/tokenizer_mac.json")

bench_file = Path("evals/benchmarks/private_unseen_suite.json")
with open(bench_file) as f:
    tasks = json.load(f)

print(f"Loaded {len(tasks)} benchmark tasks across {len(set(t['category'] for t in tasks))} categories.")

master_report = {}

for model_name, ckpt_path in MODELS:
    print(f"\n" + "=" * 80)
    print(f"  EVALUATING MODEL: {model_name}")
    print(f"  Checkpoint: {ckpt_path}")
    print("=" * 80)
    
    t0 = time.time()
    model, backend, _ = load_model_from_checkpoint(ckpt_path)
    model.eval()
    
    task_results = []
    
    # Aggregators
    cat_metrics = {}
    
    for idx, t in enumerate(tasks, 1):
        cat = t.get("category", "General")
        prompt = t["prompt"]
        prompt_nums = extract_numbers(prompt)
        
        # Greedy completion
        raw_comp = _generate_greedy_completion(model, tok, backend, prompt, max_new_tokens=64)
        clean_comp = clean_functional_completion(prompt, raw_comp)
        
        comp_token_ids = tok.encode(raw_comp).ids
        comp_nums = extract_numbers(clean_comp)
        
        # Repetition metrics
        n_tokens = len(comp_token_ids)
        rep2 = compute_ngram_repetition(comp_token_ids, 2)
        rep3 = compute_ngram_repetition(comp_token_ids, 3)
        rep4 = compute_ngram_repetition(comp_token_ids, 4)
        line_rep, has_consec_dup = compute_line_repetition(clean_comp)
        is_loop = detect_degenerate_loop(comp_token_ids, 3, min_repeats=3)
        
        is_early_exit = (n_tokens <= 4)
        hit_max = (n_tokens >= 64)
        
        # Numerical metrics
        p_set = set(prompt_nums)
        c_set = set(comp_nums)
        has_p_nums = len(p_set) > 0
        
        if has_p_nums:
            overlap = p_set & c_set
            hallucinated = c_set - p_set
            num_recall = len(overlap) / len(p_set)
            num_prec = len(overlap) / len(c_set) if c_set else 0.0
            exact_num = p_set.issubset(c_set)
            zero_nums = (len(c_set) == 0)
        else:
            num_recall = None
            num_prec = None
            exact_num = None
            zero_nums = None
            hallucinated = c_set
        
        res = {
            "id": t["id"],
            "category": cat,
            "tokens": n_tokens,
            "early_exit": is_early_exit,
            "hit_max": hit_max,
            "rep2": rep2,
            "rep3": rep3,
            "rep4": rep4,
            "line_rep": line_rep,
            "consec_dup": has_consec_dup,
            "is_loop": is_loop,
            "has_p_nums": has_p_nums,
            "num_recall": num_recall,
            "num_prec": num_prec,
            "exact_num": exact_num,
            "zero_nums": zero_nums,
            "p_nums": list(p_set),
            "c_nums": list(c_set),
            "hallucinated": list(hallucinated),
        }
        task_results.append(res)
        
        if idx % 100 == 0 or idx == len(tasks):
            print(f"  [{idx}/{len(tasks)}] tasks processed... ({time.time()-t0:.1f}s)")
            
    # Compute aggregates overall and by category
    def summarize_list(items):
        n = len(items)
        if n == 0:
            return {}
        tok_lens = [x["tokens"] for x in items]
        early_exits = sum(1 for x in items if x["early_exit"])
        hit_maxes = sum(1 for x in items if x["hit_max"])
        rep2s = [x["rep2"] for x in items]
        rep3s = [x["rep3"] for x in items]
        rep4s = [x["rep4"] for x in items]
        line_reps = [x["line_rep"] for x in items]
        loops = sum(1 for x in items if x["is_loop"])
        consec_dups = sum(1 for x in items if x["consec_dup"])
        
        num_items = [x for x in items if x["has_p_nums"]]
        n_num = len(num_items)
        if n_num > 0:
            num_rec = float(np.mean([x["num_recall"] for x in num_items]))
            num_prc = float(np.mean([x["num_prec"] for x in num_items]))
            exact_m = sum(1 for x in num_items if x["exact_num"]) / n_num
            zero_n = sum(1 for x in num_items if x["zero_nums"]) / n_num
            halluc_rate = sum(1 for x in num_items if len(x["hallucinated"]) > 0) / n_num
        else:
            num_rec, num_prc, exact_m, zero_n, halluc_rate = 0.0, 0.0, 0.0, 0.0, 0.0
            
        return {
            "count": n,
            "avg_tokens": float(np.mean(tok_lens)),
            "early_exit_pct": round(early_exits / n * 100.0, 1),
            "hit_max_pct": round(hit_maxes / n * 100.0, 1),
            "rep_2gram_pct": round(float(np.mean(rep2s)) * 100.0, 1),
            "rep_3gram_pct": round(float(np.mean(rep3s)) * 100.0, 1),
            "rep_4gram_pct": round(float(np.mean(rep4s)) * 100.0, 1),
            "line_rep_pct": round(float(np.mean(line_reps)) * 100.0, 1),
            "degenerate_loop_pct": round(loops / n * 100.0, 1),
            "consec_line_dup_pct": round(consec_dups / n * 100.0, 1),
            "numerical": {
                "tasks_with_numbers": n_num,
                "numeric_recall_pct": round(num_rec * 100.0, 1),
                "numeric_precision_pct": round(num_prc * 100.0, 1),
                "exact_number_match_pct": round(exact_m * 100.0, 1),
                "zero_number_emitted_pct": round(zero_n * 100.0, 1),
                "hallucinated_number_rate_pct": round(halluc_rate * 100.0, 1),
            }
        }
        
    overall_summary = summarize_list(task_results)
    
    by_category = {}
    for cat in sorted(list(set(t["category"] for t in tasks))):
        cat_items = [x for x in task_results if x["category"] == cat]
        by_category[cat] = summarize_list(cat_items)
        
    master_report[model_name] = {
        "overall": overall_summary,
        "by_category": by_category,
    }

# Save consolidated JSON report
out_path = Path("logs/eval_512_extended_metrics.json")
with open(out_path, "w") as f:
    json.dump(master_report, f, indent=2)
print(f"\n✓ Saved extended metrics report to {out_path}")

# Print Scorecards
print("\n" + "=" * 105)
print("  TABLE 1: OVERALL 512-TASK GENERATION DYNAMICS & REPETITION METRICS")
print("=" * 105)
print(f"{'Model Name':<32} | {'Avg Len':<8} | {'EarlyExit':<10} | {'HitMax':<8} | {'2-GramRep':<10} | {'3-GramRep':<10} | {'LineRep':<8} | {'LoopRate':<8}")
print("-" * 105)
for name, rep in master_report.items():
    ov = rep["overall"]
    print(f"{name:<32} | {ov['avg_tokens']:>6.1f} t | {ov['early_exit_pct']:>8.1f}% | {ov['hit_max_pct']:>6.1f}% | {ov['rep_2gram_pct']:>8.1f}%  | {ov['rep_3gram_pct']:>8.1f}%  | {ov['line_rep_pct']:>6.1f}% | {ov['degenerate_loop_pct']:>6.1f}%")
print("=" * 105)

print("\n" + "=" * 105)
print("  TABLE 2: ALGORITHMS & NUMERICAL (64 TASKS) - NUMBER RETENTION & ACCURACY")
print("=" * 105)
print(f"{'Model Name':<32} | {'Recall':<8} | {'Precision':<10} | {'ExactMatch':<11} | {'ZeroNumEmit':<12} | {'HallucRate':<11}")
print("-" * 105)
for name, rep in master_report.items():
    alg = rep["by_category"].get("Algorithms & Numerical", {}).get("numerical", {})
    print(f"{name:<32} | {alg['numeric_recall_pct']:>6.1f}% | {alg['numeric_precision_pct']:>8.1f}%  | {alg['exact_number_match_pct']:>9.1f}%  | {alg['zero_number_emitted_pct']:>10.1f}% | {alg['hallucinated_number_rate_pct']:>9.1f}%")
print("=" * 105)
