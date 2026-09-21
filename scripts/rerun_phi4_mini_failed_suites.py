#!/usr/bin/env python3
"""
Targeted evaluation runner for Microsoft Phi-4 Mini Reasoning.
Reruns OpenAI HumanEval, BFCL Tool-Use, and Cybersecurity with expanded 2,048 token runway.
"""

import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = PROJECT_ROOT / "logs"
LMS_BIN = Path.home() / ".lmstudio" / "bin" / "lms"

sys.path.insert(0, str(PROJECT_ROOT))
from telos.eval.adapters.openai_api import OpenAIAPIAdapter
from telos.eval.runner import evaluate


def ensure_phi4_loaded() -> None:
    """Ensure Microsoft Phi-4 Mini Reasoning is active in LM Studio with 16k context."""
    print(">>> Verifying Phi-4 Mini Reasoning status in LM Studio...")
    res = subprocess.run([str(LMS_BIN), "ps"], capture_output=True, text=True)
    if "microsoft/phi-4-mini-reasoning" not in res.stdout:
        print("  Loading microsoft/phi-4-mini-reasoning with 16,384 context runway...")
        subprocess.run(
            [str(LMS_BIN), "load", "microsoft/phi-4-mini-reasoning", "-y", "-c", "16384", "--gpu", "max"],
            check=True,
        )
        time.sleep(5)
    else:
        print("  ✓ Microsoft Phi-4 Mini Reasoning is active in LM Studio.")


def rerun_failed_suites() -> None:
    """Rerun HumanEval, Tool-Use, and Cyber with 2,048 tokens and reasoning extraction."""
    ensure_phi4_loaded()

    suites = [
        {
            "name": "humaneval",
            "type": "code",
            "label": "OpenAI HumanEval (Python Code — 2,048 Tokens)",
            "output": "eval_report_phi4_mini_humaneval_full.json",
            "tokens": 2048,
            "prompt": "You are an expert Python programmer. Complete the code strictly without conversational text.",
        },
        {
            "name": "tooluse",
            "type": "tooluse",
            "label": "BFCL Tool-Use & Function Calling (2,048 Tokens)",
            "output": "eval_report_phi4_mini_tooluse_full.json",
            "tokens": 2048,
            "prompt": "You are an expert tool-calling assistant. Emit tool calls as a valid JSON list.",
        },
        {
            "name": "cyber",
            "type": "cyber",
            "label": "Cybersecurity OWASP/CWE Auditing (2,048 Tokens)",
            "output": "eval_report_phi4_mini_cyber_full.json",
            "tokens": 2048,
            "prompt": "You are a senior security researcher. Fix vulnerabilities and return clean code.",
        },
    ]

    for s in suites:
        out_file = LOGS_DIR / s["output"]
        print(f"\n" + "=" * 85)
        print(f"  RERUNNING {s['label'].upper()} FOR PHI-4 MINI REASONING")
        print("=" * 85)

        # Initialize adapter with reasoning content fallback and stripped blank line stop tokens
        adapter = OpenAIAPIAdapter(
            model_name="microsoft/phi-4-mini-reasoning",
            api_base="http://127.0.0.1:1234/v1",
            system_prompt=s["prompt"],
            timeout=180.0,
        )

        t0 = time.time()
        evaluate(
            checkpoint=adapter,
            mode="functional",
            suite=s["name"],
            benchmark_type=s["type"],
            max_tasks=None,
            max_new_tokens=s["tokens"],
            output_path=str(out_file),
        )
        elapsed = time.time() - t0

        with open(out_file) as f:
            rep_d = json.load(f)
        b_data = (
            rep_d.get(s["type"], {})
            or rep_d.get("functional", {})
            or rep_d.get("math", {})
            or rep_d.get("science", {})
        )
        p_rate = b_data.get("pass_at_1_pct") or b_data.get("pass_rate_pct", 0.0)
        print(f"✓ Completed {s['label']} in {elapsed/60.0:.2f} mins (Pass@1: {p_rate:.2f}%)")


def update_master_summary_and_scorecard() -> None:
    """Update master summary and regenerate final 5-model institutional scorecard."""
    summary_path = LOGS_DIR / "eval_report_phi4_mini_master_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            master_summary = json.load(f)
    else:
        master_summary = {}

    suite_mapping = [
        ("humaneval", "code", "eval_report_phi4_mini_humaneval_full.json", "OpenAI HumanEval (Python Code)"),
        ("tooluse", "tooluse", "eval_report_phi4_mini_tooluse_full.json", "BFCL Tool-Use & Function Calling"),
        ("cyber", "cyber", "eval_report_phi4_mini_cyber_full.json", "Cybersecurity OWASP/CWE Auditing"),
        ("gpqa_diamond", "science", "eval_report_phi4_mini_gpqa_diamond_full.json", "GPQA Diamond (PhD Science)"),
        ("mmlu_science", "science", "eval_report_phi4_mini_mmlu_science_full.json", "MMLU Science & STEM"),
        ("competition_math", "math", "eval_report_phi4_mini_competition_math_full.json", "Competition MATH (Hendrycks)"),
        ("arc", "science", "eval_report_phi4_mini_arc_full.json", "ARC-Challenge Science Reasoning"),
    ]

    for s_name, s_type, s_file, s_label in suite_mapping:
        p = LOGS_DIR / s_file
        if p.exists():
            with open(p) as f:
                rep_d = json.load(f)
            b_data = (
                rep_d.get(s_type, {})
                or rep_d.get("functional", {})
                or rep_d.get("math", {})
                or rep_d.get("science", {})
            )
            p_rate = b_data.get("pass_at_1_pct") or b_data.get("pass_rate_pct", 0.0)
            master_summary[s_name] = {
                "total_tasks": b_data.get("total_tasks"),
                "pass_at_1_pct": p_rate,
                "pass_at_1_95ci": b_data.get("pass_at_1_95ci") or b_data.get("pass_rate_95ci"),
                "report_file": str(p),
                "label": s_label,
            }

    with open(summary_path, "w") as f:
        json.dump(master_summary, f, indent=2)
    print(f"✓ Updated Phi-4 Mini Master Summary at {summary_path}")

    # Regenerate publication markdown and json matrices
    from scripts.run_post_school_master_pipeline import compile_5model_publication_scorecard
    compile_5model_publication_scorecard()


if __name__ == "__main__":
    rerun_failed_suites()
    update_master_summary_and_scorecard()
