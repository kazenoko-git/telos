import io
import json
import random
import urllib.request
import zipfile
from pathlib import Path
import pandas as pd
import datasets

BENCHMARK_DIR = Path("evals/benchmarks")
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

def build_gpqa_diamond():
    print("Fetching GPQA Diamond benchmark...")
    url = "https://raw.githubusercontent.com/idavidrein/gpqa/main/dataset.zip"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        z = zipfile.ZipFile(io.BytesIO(resp.read()))
        with z.open("dataset/gpqa_diamond.csv", pwd=b"deserted-untie-orchid") as f:
            df = pd.read_csv(f)

    tasks = []
    letters = ["A", "B", "C", "D"]
    for idx, row in df.iterrows():
        rec_id = str(row.get("Record ID", f"gpqa_{idx}"))
        q = str(row.get("Question", "")).strip()
        correct_ans = str(row.get("Correct Answer", "")).strip()
        incorrect_1 = str(row.get("Incorrect Answer 1", "")).strip()
        incorrect_2 = str(row.get("Incorrect Answer 2", "")).strip()
        incorrect_3 = str(row.get("Incorrect Answer 3", "")).strip()
        subdomain = str(row.get("Subdomain", "Science")).strip()

        # Deterministically shuffle options using record ID as seed
        rng = random.Random(rec_id)
        choices_raw = [
            {"text": correct_ans, "is_correct": True},
            {"text": incorrect_1, "is_correct": False},
            {"text": incorrect_2, "is_correct": False},
            {"text": incorrect_3, "is_correct": False},
        ]
        rng.shuffle(choices_raw)

        choices_formatted = []
        answer_key = "A"
        for l, c in zip(letters, choices_raw):
            choices_formatted.append(f"({l}) {c['text']}")
            if c["is_correct"]:
                answer_key = l

        prompt = (
            f"The following is a graduate-level scientific question on {subdomain}. "
            f"Think step-by-step, then state the single correct letter choice (e.g. 'Answer: A').\n\n"
            f"Question: {q}\n\n"
            f"Choices:\n" + "\n".join(choices_formatted) + "\n"
        )

        tasks.append({
            "id": f"gpqa_diamond/{rec_id}",
            "domain": "science",
            "subdomain": subdomain,
            "prompt": prompt,
            "question": q,
            "choices": [{"label": l, "text": c["text"]} for l, c in zip(letters, choices_raw)],
            "answer_key": answer_key,
            "evaluation_type": "multiple_choice_letter",
        })

    out_path = BENCHMARK_DIR / "gpqa_diamond_suite.json"
    with open(out_path, "w") as f:
        json.dump(tasks, f, indent=2)
    print(f"  ✓ Saved {len(tasks)} GPQA Diamond tasks -> {out_path}")
    return tasks

def build_mmlu_science():
    print("Fetching MMLU Science & STEM benchmark...")
    subjects = [
        "college_biology",
        "college_chemistry",
        "college_physics",
        "college_computer_science",
        "astronomy",
        "conceptual_physics",
    ]
    tasks = []
    letters = ["A", "B", "C", "D"]
    for subj in subjects:
        try:
            ds = datasets.load_dataset("cais/mmlu", subj, split="test")
            for idx, row in enumerate(ds):
                q = str(row.get("question", "")).strip()
                choices = [str(c).strip() for c in row.get("choices", [])]
                ans_idx = int(row.get("answer", 0))
                ans_letter = letters[ans_idx] if ans_idx < len(letters) else "A"

                choices_formatted = "\n".join([f"({l}) {c}" for l, c in zip(letters, choices)])
                prompt = (
                    f"The following is a multiple-choice question on {subj.replace('_', ' ')}. "
                    f"Think step-by-step, then state the single correct letter choice (e.g. 'Answer: A').\n\n"
                    f"Question: {q}\n\n"
                    f"Choices:\n{choices_formatted}\n"
                )

                tasks.append({
                    "id": f"mmlu/{subj}_{idx}",
                    "domain": "science",
                    "subdomain": subj,
                    "prompt": prompt,
                    "question": q,
                    "choices": [{"label": l, "text": c} for l, c in zip(letters, choices)],
                    "answer_key": ans_letter,
                    "evaluation_type": "multiple_choice_letter",
                })
        except Exception as e:
            print(f"  Warning: failed {subj}: {e}")

    out_path = BENCHMARK_DIR / "mmlu_science_suite.json"
    with open(out_path, "w") as f:
        json.dump(tasks, f, indent=2)
    print(f"  ✓ Saved {len(tasks)} MMLU Science tasks -> {out_path}")
    return tasks

build_gpqa_diamond()
build_mmlu_science()
