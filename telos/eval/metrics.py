"""
Extended Generation & Evaluation Metrics for Télos Evaluation Engine.

Provides modular analytical tools for:
1. Token & N-Gram Repetition Dynamics (2-gram, 3-gram, 4-gram repetition rates, degenerate loops).
2. Line-Level Code Duplication (consecutive repeated lines, body redundancy).
3. Numerical & Mathematical Constant Precision (extracting expected literals from docstring/prompt,
   measuring recall, precision, zero-number emission rates, and hallucinated constants).
"""

import re
from typing import Dict, Any, List, Tuple, Set, Optional
import numpy as np


def compute_ngram_repetition(token_ids: List[int], n: int = 3) -> float:
    """
    Computes the repetition rate of n-grams in a sequence of token IDs.
    Repetition rate = 1.0 - (unique_ngrams / total_ngrams).
    Returns 0.0 if sequence length is less than n.
    """
    if len(token_ids) < n:
        return 0.0
    ngrams = [tuple(token_ids[i : i + n]) for i in range(len(token_ids) - n + 1)]
    if not ngrams:
        return 0.0
    unique_count = len(set(ngrams))
    return float(1.0 - (unique_count / len(ngrams)))


def detect_degenerate_loop(token_ids: List[int], n: int = 3, min_repeats: int = 3, max_period: int = 32) -> bool:
    """
    Detects whether an n-gram or periodic token cycle repeats consecutively min_repeats or more times.
    Checks strides k from 1 up to max_period to capture runaway single-token, phrase, and line-level cycles.
    """
    total = len(token_ids)
    if total < min_repeats:
        return False
    limit_k = min(max_period, total // min_repeats)
    for k in range(1, limit_k + 1):
        streak = 0
        target = k * (min_repeats - 1)
        for i in range(k, total):
            if token_ids[i] == token_ids[i - k]:
                streak += 1
                if streak >= target:
                    return True
            else:
                streak = 0
    return False


def compute_line_repetition(code_text: str) -> Tuple[float, bool]:
    """
    Analyzes line-level repetition in generated code blocks.
    Returns:
        rep_rate: Ratio of duplicate non-empty lines (1.0 - unique/total).
        has_consecutive: True if any identical non-empty line repeats consecutively.
    """
    lines = [line.strip() for line in code_text.splitlines() if line.strip()]
    if len(lines) <= 1:
        return 0.0, False
    has_consecutive = any(lines[i] == lines[i - 1] for i in range(1, len(lines)))
    unique_lines = len(set(lines))
    rep_rate = float(1.0 - (unique_lines / len(lines)))
    return rep_rate, has_consecutive


def extract_numeric_literals(text: str) -> List[str]:
    """
    Extracts all integer and floating-point numeric literals from text using regex.
    Ignores variable names containing digits (e.g. 'task_19' -> matches '19' if word-bounded).
    """
    # Matches integers and decimals bounded by word boundaries
    return re.findall(r"\b\d+(?:\.\d+)?\b", text)


def evaluate_numerical_accuracy(prompt: str, completion: str) -> Dict[str, Any]:
    """
    Evaluates whether numbers specified in the prompt/docstring were properly preserved,
    missed, or hallucinated in the model's generated code completion.
    """
    prompt_nums = set(extract_numeric_literals(prompt))
    comp_nums = set(extract_numeric_literals(completion))

    has_prompt_numbers = len(prompt_nums) > 0
    if not has_prompt_numbers:
        return {
            "has_prompt_numbers": False,
            "prompt_numbers": [],
            "completion_numbers": list(comp_nums),
            "overlap_numbers": [],
            "hallucinated_numbers": list(comp_nums),
            "numeric_recall": None,
            "numeric_precision": None,
            "exact_number_match": None,
            "zero_numbers_emitted": None,
        }

    overlap = prompt_nums & comp_nums
    hallucinated = comp_nums - prompt_nums

    # Metric calculations
    recall = len(overlap) / len(prompt_nums)
    precision = len(overlap) / len(comp_nums) if comp_nums else 0.0
    exact_match = prompt_nums.issubset(comp_nums)
    zero_numbers = (len(comp_nums) == 0)

    return {
        "has_prompt_numbers": True,
        "prompt_numbers": list(prompt_nums),
        "completion_numbers": list(comp_nums),
        "overlap_numbers": list(overlap),
        "hallucinated_numbers": list(hallucinated),
        "numeric_recall": float(recall),
        "numeric_precision": float(precision),
        "exact_number_match": bool(exact_match),
        "zero_numbers_emitted": bool(zero_numbers),
    }


def analyze_task_completion(
    prompt: str,
    raw_completion: str,
    clean_completion: str,
    tokenizer: Any
) -> Dict[str, Any]:
    """
    Analyzes a single generated code completion across repetition dynamics and numerical accuracy.
    """
    # Tokenize raw generated continuation tokens
    token_ids = tokenizer.encode(raw_completion).ids if hasattr(tokenizer, "encode") else []
    token_count = len(token_ids)

    rep_2gram = compute_ngram_repetition(token_ids, n=2)
    rep_3gram = compute_ngram_repetition(token_ids, n=3)
    rep_4gram = compute_ngram_repetition(token_ids, n=4)
    line_rep, has_consec_line = compute_line_repetition(clean_completion)
    is_loop = detect_degenerate_loop(token_ids, n=3, min_repeats=3)

    is_early_exit = (token_count <= 4)
    num_eval = evaluate_numerical_accuracy(prompt, clean_completion)

    return {
        "token_count": token_count,
        "is_early_exit": is_early_exit,
        "rep_2gram": rep_2gram,
        "rep_3gram": rep_3gram,
        "rep_4gram": rep_4gram,
        "line_rep": line_rep,
        "has_consecutive_dup_lines": has_consec_line,
        "is_degenerate_loop": is_loop,
        "numerical": num_eval,
    }


def aggregate_extended_metrics(task_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregates per-task extended metrics into a comprehensive statistical summary.
    """
    total = len(task_metrics)
    if total == 0:
        return {}

    tok_counts = [m["token_count"] for m in task_metrics]
    early_exits = sum(1 for m in task_metrics if m["is_early_exit"])
    rep2_vals = [m["rep_2gram"] for m in task_metrics]
    rep3_vals = [m["rep_3gram"] for m in task_metrics]
    rep4_vals = [m["rep_4gram"] for m in task_metrics]
    line_rep_vals = [m["line_rep"] for m in task_metrics]
    consec_dups = sum(1 for m in task_metrics if m["has_consecutive_dup_lines"])
    loops = sum(1 for m in task_metrics if m["is_degenerate_loop"])

    # Numerical metrics filtered to tasks that actually had numbers in the prompt
    num_tasks = [m["numerical"] for m in task_metrics if m["numerical"]["has_prompt_numbers"]]
    n_num = len(num_tasks)

    if n_num > 0:
        mean_recall = float(np.mean([x["numeric_recall"] for x in num_tasks]))
        mean_precision = float(np.mean([x["numeric_precision"] for x in num_tasks]))
        exact_matches = sum(1 for x in num_tasks if x["exact_number_match"])
        zero_emitted = sum(1 for x in num_tasks if x["zero_numbers_emitted"])
        hallucinated_count = sum(1 for x in num_tasks if len(x["hallucinated_numbers"]) > 0)
        exact_pct = round((exact_matches / n_num) * 100.0, 1)
        zero_pct = round((zero_emitted / n_num) * 100.0, 1)
        halluc_pct = round((hallucinated_count / n_num) * 100.0, 1)
    else:
        mean_recall, mean_precision, exact_pct, zero_pct, halluc_pct = 0.0, 0.0, 0.0, 0.0, 0.0

    return {
        "count": total,
        "avg_token_length": round(float(np.mean(tok_counts)), 1),
        "early_exit_pct": round((early_exits / total) * 100.0, 1),
        "rep_2gram_pct": round(float(np.mean(rep2_vals)) * 100.0, 1),
        "rep_3gram_pct": round(float(np.mean(rep3_vals)) * 100.0, 1),
        "rep_4gram_pct": round(float(np.mean(rep4_vals)) * 100.0, 1),
        "line_repetition_pct": round(float(np.mean(line_rep_vals)) * 100.0, 1),
        "consecutive_dup_line_pct": round((consec_dups / total) * 100.0, 1),
        "degenerate_loop_pct": round((loops / total) * 100.0, 1),
        "numerical": {
            "tasks_with_prompt_numbers": n_num,
            "numeric_recall_pct": round(mean_recall * 100.0, 1),
            "numeric_precision_pct": round(mean_precision * 100.0, 1),
            "exact_number_match_pct": exact_pct,
            "zero_numbers_emitted_pct": zero_pct,
            "hallucinated_numbers_pct": halluc_pct,
        },
    }
