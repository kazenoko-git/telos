"""
Consolidated Scorecard Compiler for Apple AFM 3 Core Advanced vs IBM Granite 4.2 3B MLX.
Reads all 7 benchmark reports from disk, extracts official metrics, and writes:
1. logs/eval_report_granite_mlx_master_summary.json
2. logs/eval_report_afm3_master_summary.json
3. logs/comparative_report_afm3_vs_granite_mlx.json
"""

import json
import time
from pathlib import Path

LOGS_DIR = Path("logs")

SUITES = [
    ("humaneval", "OpenAI HumanEval (Python Code)", "code", "humaneval", 164),
    ("tooluse", "BFCL Tool-Use & Function Calling", "tooluse", "tooluse", 30),
    ("cyber", "Cybersecurity OWASP/CWE Auditing", "cyber", "cyber", 50),
    ("gpqa_diamond", "GPQA Diamond (PhD Science — 1,024 Tokens)", "science", "gpqa_diamond", 198),
    ("mmlu_science", "MMLU Science & STEM", "science", "mmlu_science", 833),
    ("competition_math", "Competition MATH (Hendrycks — 1,024 Tokens)", "math", "competition_math", 700),
    ("arc", "ARC-Challenge Science Reasoning", "science", "arc", 1172),
]


def extract_metrics(filepath: Path, b_type: str):
    if not filepath.exists():
        return None
    with open(filepath) as f:
        data = json.load(f)
    obj = data.get(b_type, {})
    if "functional" in obj:
        obj = obj["functional"]

    pass_pct = obj.get("pass_at_1_pct")
    if pass_pct is None:
        pass_pct = obj.get("pass_rate_pct", 0.0)

    ci = obj.get("pass_at_1_95ci") or obj.get("pass_rate_95ci")
    ast_val = obj.get("ast_validity_pct") or obj.get("syntax_validity_pct")

    return {
        "total_tasks": obj.get("total_tasks"),
        "pass_at_1_pct": round(pass_pct, 2),
        "pass_at_1_95ci": ci,
        "ast_validity_pct": ast_val,
        "execution_outcomes": obj.get("execution_outcomes"),
        "report_file": str(filepath)
    }


def main():
    granite_master = {}
    afm3_master = {}
    comparative = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "model_a": {
                "name": "Apple AFM 3 Core Advanced",
                "runtime": "Native Swift FoundationModels.framework IPC (Apple Silicon)"
            },
            "model_b": {
                "name": "IBM Granite 4.2 3B MLX",
                "runtime": "LM Studio Metal Acceleration (Apple Silicon)"
            }
        },
        "benchmarks": {}
    }

    for key, label, b_type, s_name, expected_tasks in SUITES:
        g_file = LOGS_DIR / f"eval_report_granite_mlx_{key}_full.json"
        a_file = LOGS_DIR / f"eval_report_afm3_{key}_full.json"

        g_met = extract_metrics(g_file, b_type)
        a_met = extract_metrics(a_file, b_type)

        if g_met:
            g_met["label"] = label
            granite_master[key] = g_met

        if a_met:
            a_met["label"] = label
            afm3_master[key] = a_met

        a_pass = a_met["pass_at_1_pct"] if a_met else 0.0
        g_pass = g_met["pass_at_1_pct"] if g_met else 0.0
        delta = round(g_pass - a_pass, 2)

        comparative["benchmarks"][key] = {
            "label": label,
            "tasks": expected_tasks,
            "afm3_core_advanced": {
                "pass_at_1_pct": a_pass,
                "pass_at_1_95ci": a_met.get("pass_at_1_95ci") if a_met else None,
                "ast_validity_pct": a_met.get("ast_validity_pct") if a_met else None
            },
            "granite_4_2_3b_mlx": {
                "pass_at_1_pct": g_pass,
                "pass_at_1_95ci": g_met.get("pass_at_1_95ci") if g_met else None,
                "ast_validity_pct": g_met.get("ast_validity_pct") if g_met else None
            },
            "delta_granite_vs_afm3_pct": delta,
            "winner": "IBM Granite 4.2 3B" if delta > 0 else ("Apple AFM 3 Core Advanced" if delta < 0 else "Tie")
        }

    with open(LOGS_DIR / "eval_report_granite_mlx_master_summary.json", "w") as f:
        json.dump(granite_master, f, indent=2)

    with open(LOGS_DIR / "eval_report_afm3_master_summary.json", "w") as f:
        json.dump(afm3_master, f, indent=2)

    with open(LOGS_DIR / "comparative_report_afm3_vs_granite_mlx.json", "w") as f:
        json.dump(comparative, f, indent=2)

    print("[OK] Successfully compiled master summaries and comparative report.")


if __name__ == "__main__":
    main()
