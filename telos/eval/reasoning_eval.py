"""
Reasoning, Mathematical, and Scientific Evaluation Engine for Télos.

Evaluates:
1. GSM8K: Grade School Math with exact numeric equivalence.
2. ARC-Challenge: Scientific multiple choice reasoning (A/B/C/D).
3. Competition MATH: High-school & Olympiad LaTeX \\boxed{...} problems.
"""

import re
import string
from typing import Dict, Any, List, Optional, Tuple


def normalize_numeric_string(s: str) -> Optional[float]:
    """Parses numeric string, removing commas, currency symbols, and whitespace."""
    s = s.strip().replace(",", "").replace("$", "").replace("%", "")
    try:
        return float(s)
    except ValueError:
        # Check fraction like '3/4'
        if "/" in s:
            parts = s.split("/")
            if len(parts) == 2:
                try:
                    return float(parts[0]) / float(parts[1])
                except (ValueError, ZeroDivisionError):
                    pass
        return None


def extract_gsm8k_numeric_answer(completion_text: str) -> Optional[str]:
    """
    Extracts final numeric answer from model generation.
    Supports:
    1. Standard '#### 42'
    2. 'The answer is: 42' or 'Answer: 42'
    3. Final standalone number on the last line
    """
    text = completion_text.strip()

    # Pattern 1: #### <number>
    match_hash = re.search(r"####\s*([+-]?\d+(?:\.\d+)?)", text)
    if match_hash:
        return match_hash.group(1).strip()

    # Pattern 2: (?:the answer is|answer is|final answer:?)\s*([+-]?\d+(?:\.\d+)?)
    match_ans = re.search(r"(?:answer is|final answer:?|answer:)\s*([+-]?\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if match_ans:
        return match_ans.group(1).strip()

    # Pattern 3: Search backwards for any number
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].strip()

    return None


def evaluate_gsm8k_sample(candidate_completion: str, target_answer: str) -> Tuple[bool, str]:
    """Evaluates whether model completion matches GSM8K ground truth numeric answer."""
    extracted = extract_gsm8k_numeric_answer(candidate_completion)
    if not extracted:
        return False, "No numeric answer extracted"

    cand_val = normalize_numeric_string(extracted)
    gold_val = normalize_numeric_string(target_answer)

    if cand_val is not None and gold_val is not None:
        is_match = abs(cand_val - gold_val) < 1e-4
        return is_match, f"Cand={cand_val}, Gold={gold_val}"

    # Fallback to string equality
    return extracted.strip() == target_answer.strip(), f"Cand='{extracted}', Gold='{target_answer}'"


def extract_arc_choice(completion_text: str) -> Optional[str]:
    """
    Extracts multiple-choice letter (A, B, C, D, etc.) from ARC completion.
    """
    text = completion_text.strip()

    # 1. Look for 'Answer: [A-D]' or 'Choice: [A-D]'
    match = re.search(r"(?:answer|choice|option):\s*\(?([A-E1-4])\)?", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # 2. Look for '\b([A-D])\b' on the first or last line
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines:
        for target_line in [lines[-1], lines[0]]:
            match_letter = re.search(r"\b([A-D])\b", target_line)
            if match_letter:
                return match_letter.group(1)

    return None


def evaluate_arc_sample(candidate_completion: str, target_answer_key: str) -> Tuple[bool, str]:
    """Evaluates multiple choice letter accuracy for ARC-Challenge."""
    extracted = extract_arc_choice(candidate_completion)
    if not extracted:
        return False, "No multiple choice option extracted"

    gold = target_answer_key.strip().upper()
    is_correct = (extracted == gold)
    return is_correct, f"Extracted='{extracted}', Gold='{gold}'"


def extract_boxed_latex(text: str) -> Optional[str]:
    """Extracts content inside LaTeX \\boxed{...}."""
    idx = text.rfind(r"\boxed{")
    if idx == -1:
        return None
    start = idx + len(r"\boxed{")
    depth = 1
    end = start
    while end < len(text) and depth > 0:
        if text[end] == "{":
            depth += 1
        elif text[end] == "}":
            depth -= 1
        end += 1
    if depth == 0:
        return text[start:end - 1].strip()
    return None


def evaluate_competition_math_sample(candidate_completion: str, target_boxed: str) -> Tuple[bool, str]:
    """Evaluates Competition MATH answer against gold target."""
    cand_boxed = extract_boxed_latex(candidate_completion)
    if not cand_boxed:
        # Check if the exact target is anywhere in the final lines
        if target_boxed and target_boxed in candidate_completion.splitlines()[-1]:
            return True, "Found in final line"
        return False, "No \\boxed{...} answer found"

    # Clean whitespace and standard LaTeX markers
    clean_cand = cand_boxed.strip().replace(" ", "").replace("$", "")
    clean_gold = target_boxed.strip().replace(" ", "").replace("$", "")

    if clean_cand == clean_gold:
        return True, f"Exact match: {cand_boxed}"

    # Try numeric equality
    num_cand = normalize_numeric_string(clean_cand)
    num_gold = normalize_numeric_string(clean_gold)
    if num_cand is not None and num_gold is not None:
        if abs(num_cand - num_gold) < 1e-4:
            return True, f"Numeric equivalence: {num_cand} == {num_gold}"

    return False, f"Mismatch: Cand='{clean_cand}' vs Gold='{clean_gold}'"
