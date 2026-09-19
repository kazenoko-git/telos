"""
Math & Science Benchmark Suites Fetcher.

Fetches and standardizes datasets for:
1. GSM8K (Grade School Math 8K) -> evals/benchmarks/gsm8k_suite.json
2. ARC-Challenge (AI2 Reasoning Challenge) -> evals/benchmarks/arc_challenge_suite.json
3. Competition MATH (Hendrycks MATH) -> evals/benchmarks/competition_math_suite.json
"""

import json
import re
import urllib.request
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd


def extract_gsm8k_answer(solution_str: str) -> str:
    """
    Extracts the final numeric ground truth from GSM8K solution string.
    GSM8K solutions follow the standard convention: '... #### 42' or '... #### -15.5'
    """
    if "####" in solution_str:
        ans = solution_str.split("####")[-1].strip()
        # Remove commas from formatted numbers like 1,000 -> 1000
        return ans.replace(",", "")
    return solution_str.strip()


def extract_boxed_math_answer(solution_str: str) -> str:
    """
    Extracts content inside LaTeX \\boxed{...} from Hendrycks MATH solution.
    Handles nested braces properly using regex/stack parsing.
    """
    # Look for \boxed{
    idx = solution_str.rfind(r"\boxed{")
    if idx == -1:
        return ""
    start = idx + len(r"\boxed{")
    depth = 1
    end = start
    while end < len(solution_str) and depth > 0:
        if solution_str[end] == "{":
            depth += 1
        elif solution_str[end] == "}":
            depth -= 1
        end += 1
    if depth == 0:
        return solution_str[start:end - 1].strip()
    return ""


def fetch_gsm8k(output_path: Path) -> List[Dict[str, Any]]:
    """Fetches OpenAI GSM8K test set directly from official repository."""
    print("Fetching GSM8K test benchmark (1,319 problems)...")
    url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    req = urllib.request.Request(url, headers={"User-Agent": "Télos-Math-Fetcher/2.0"})

    tasks = []
    with urllib.request.urlopen(req, timeout=30) as resp:
        for line_idx, line in enumerate(resp):
            text = line.decode("utf-8").strip()
            if not text:
                continue
            item = json.loads(text)
            q = item.get("question", "")
            sol = item.get("answer", "")
            target_val = extract_gsm8k_answer(sol)

            tasks.append({
                "id": f"gsm8k/{line_idx + 1:04d}",
                "domain": "math",
                "subdomain": "grade_school_math",
                "prompt": (
                    f"Solve the following grade-school math problem step-by-step. "
                    f"End your final answer clearly on a new line as: '#### <number>'.\n\n"
                    f"Question: {q}"
                ),
                "question": q,
                "gold_solution": sol,
                "target_answer": target_val,
                "evaluation_type": "exact_numeric",
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(tasks, f, indent=2)

    print(f"  ✓ Saved {len(tasks)} GSM8K problems -> {output_path}")
    return tasks


def fetch_arc_challenge(output_path: Path) -> List[Dict[str, Any]]:
    """Fetches AI2 Reasoning Challenge (ARC-Challenge) test set via direct Parquet endpoint."""
    print("Fetching ARC-Challenge science benchmark (1,172 questions)...")
    url = "https://huggingface.co/api/datasets/allenai/ai2_arc/parquet/ARC-Challenge/test/0.parquet"
    df = pd.read_parquet(url)

    tasks = []
    for idx, row in df.iterrows():
        task_id = str(row.get("id", f"arc_{idx}"))
        q = str(row.get("question", ""))
        choices_dict = row.get("choices", {})
        labels = [str(x) for x in choices_dict.get("label", [])]
        texts = [str(x) for x in choices_dict.get("text", [])]
        answer_key = str(row.get("answerKey", "")).strip()

        # Format multiple choice prompt
        choices_formatted = "\n".join([f"({lbl}) {txt}" for lbl, txt in zip(labels, texts)])
        prompt = (
            f"The following is a multiple-choice question on scientific reasoning. "
            f"Think step-by-step, then state the single correct letter choice (e.g. 'Answer: A').\n\n"
            f"Question: {q}\n\n"
            f"Choices:\n{choices_formatted}\n"
        )

        tasks.append({
            "id": f"arc_challenge/{task_id}",
            "domain": "science",
            "subdomain": "reasoning",
            "prompt": prompt,
            "question": q,
            "choices": [{"label": lbl, "text": txt} for lbl, txt in zip(labels, texts)],
            "answer_key": answer_key,
            "evaluation_type": "multiple_choice_letter",
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(tasks, f, indent=2)

    print(f"  ✓ Saved {len(tasks)} ARC-Challenge questions -> {output_path}")
    return tasks


def fetch_competition_math(output_path: Path, max_per_subject: int = 100) -> List[Dict[str, Any]]:
    """
    Fetches Hendrycks Competition MATH test set across subjects:
    algebra, counting_and_probability, geometry, intermediate_algebra,
    number_theory, prealgebra, precalculus.
    Samples high-quality olympiad & competition problems.
    """
    subjects = [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]
    print(f"Fetching Competition MATH benchmark across {len(subjects)} subjects...")

    tasks = []
    task_counter = 1

    for subj in subjects:
        try:
            url = f"https://huggingface.co/api/datasets/EleutherAI/hendrycks_math/parquet/{subj}/test/0.parquet"
            df = pd.read_parquet(url)
            sample_df = df.head(max_per_subject) if max_per_subject > 0 else df

            for _, row in sample_df.iterrows():
                prob = str(row.get("problem", ""))
                level = str(row.get("level", ""))
                prob_type = str(row.get("type", subj))
                sol = str(row.get("solution", ""))
                boxed_ans = extract_boxed_math_answer(sol)

                tasks.append({
                    "id": f"competition_math/{subj}_{task_counter:04d}",
                    "domain": "math",
                    "subdomain": "competition_math",
                    "subject": subj,
                    "level": level,
                    "type": prob_type,
                    "prompt": (
                        f"Solve the following high-school / competition mathematics problem. "
                        f"Show your work and place your final concise answer inside LaTeX \\boxed{{...}}.\n\n"
                        f"Problem:\n{prob}"
                    ),
                    "problem": prob,
                    "gold_solution": sol,
                    "target_answer": boxed_ans,
                    "evaluation_type": "boxed_latex_math",
                })
                task_counter += 1
            print(f"  ✓ Processed {len(sample_df)} problems for subject: {subj}")
        except Exception as exc:
            print(f"  Warning: failed fetching {subj} ({exc})")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(tasks, f, indent=2)

    print(f"  ✓ Saved {len(tasks)} Competition MATH problems -> {output_path}")
    return tasks


def fetch_and_build_all_math_science(benchmark_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Fetches all Math and Science benchmark suites."""
    gsm8k_path = benchmark_dir / "gsm8k_suite.json"
    arc_path = benchmark_dir / "arc_challenge_suite.json"
    math_path = benchmark_dir / "competition_math_suite.json"

    gsm8k_tasks = fetch_gsm8k(gsm8k_path)
    arc_tasks = fetch_arc_challenge(arc_path)
    math_tasks = fetch_competition_math(math_path, max_per_subject=100)

    return {
        "gsm8k": gsm8k_tasks,
        "arc_challenge": arc_tasks,
        "competition_math": math_tasks,
    }


if __name__ == "__main__":
    b_dir = Path(__file__).resolve().parents[2] / "evals" / "benchmarks"
    fetch_and_build_all_math_science(b_dir)
