"""
Comparative Cloud Benchmark Orchestrator: Google Gemma 4 26B & Qwen 27B.

Runs fully unconstrained institutional benchmarks in parallel with on-device AFM 3:
- Google Gemma 4 26B A4B MoE (via Google AI Studio)
- Qwen 27B (via Hack Club AI high-throughput proxy)

Zero local compute load. Isolated output filenames per model.
"""

import os
import sys
import time
from pathlib import Path
from telos.eval.runner import evaluate

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# API Keys (loaded securely from environment)
GOOGLE_KEY = os.environ.get("GEMINI_API_KEY", "")
HACKCLUB_KEY = os.environ.get("HACKCLUB_API_KEY", "")

SUITES = [
    {"name": "humaneval", "label": "OpenAI HumanEval (Python)", "type": "code"},
    {"name": "tooluse", "label": "BFCL Tool-Use & Function Calling", "type": "tooluse"},
    {"name": "cyber", "label": "Cybersecurity OWASP/CWE Auditing", "type": "cyber"},
    {"name": "gpqa_diamond", "label": "GPQA Diamond (PhD Science)", "type": "science"},
    {"name": "mmlu_science", "label": "MMLU Science & STEM", "type": "science"},
    {"name": "competition_math", "label": "Competition MATH (Hendrycks)", "type": "math"},
    {"name": "arc", "label": "ARC-Challenge Science Reasoning", "type": "science"},
]


def run_gemma_pipeline():
    """Evaluates Google Gemma 4 26B A4B MoE via Google AI Studio."""
    print("\n" + "=" * 80)
    print("  LAUNCHING BENCHMARKS: GOOGLE GEMMA 4 26B A4B MoE")
    print("  Provider: Google AI Studio | Model: gemma-4-26b-a4b-it")
    print("=" * 80)

    os.environ["GEMINI_API_KEY"] = GOOGLE_KEY

    for s in SUITES:
        suite_name = s["name"]
        label = s["label"]
        b_type = s["type"]
        out_file = LOGS_DIR / f"eval_report_gemma4_26b_{suite_name}_full.json"

        print(f"\n>>> [Gemma 4 26B] Starting: {label.upper()} -> {out_file}")
        t0 = time.time()
        try:
            evaluate(
                checkpoint="gemma-4-26b-a4b-it",
                mode="functional",
                suite=suite_name,
                benchmark_type=b_type,
                max_tasks=None,
                backend="gemini_api",
                api_key=GOOGLE_KEY,
                output_path=str(out_file),
            )
            print(f"✓ [Gemma 4 26B] Finished {label} in {(time.time()-t0)/60.0:.2f} min")
        except Exception as e:
            print(f"✗ [Gemma 4 26B] Error on {label}: {e}")


def run_qwen_pipeline():
    """Evaluates Qwen 3.8 27B via Hack Club AI high-throughput proxy."""
    print("\n" + "=" * 80)
    print("  LAUNCHING BENCHMARKS: ALIBABA QWEN 3.8 27B")
    print("  Provider: Hack Club AI Proxy | Model: qwen/qwen3.8-27b")
    print("=" * 80)

    for s in SUITES:
        suite_name = s["name"]
        label = s["label"]
        b_type = s["type"]
        out_file = LOGS_DIR / f"eval_report_qwen38_27b_{suite_name}_full.json"

        print(f"\n>>> [Qwen 3.8 27B] Starting: {label.upper()} -> {out_file}")
        t0 = time.time()
        try:
            evaluate(
                checkpoint="qwen/qwen3.8-27b",
                mode="functional",
                suite=suite_name,
                benchmark_type=b_type,
                max_tasks=None,
                backend="openai_api",
                api_base="https://ai.hackclub.com/proxy/v1",
                api_key=HACKCLUB_KEY,
                output_path=str(out_file),
            )
            print(f"✓ [Qwen 3.8 27B] Finished {label} in {(time.time()-t0)/60.0:.2f} min")
        except Exception as e:
            print(f"✗ [Qwen 3.8 27B] Error on {label}: {e}")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    if target == "gemma":
        run_gemma_pipeline()
    elif target == "qwen":
        run_qwen_pipeline()
    else:
        run_gemma_pipeline()
        run_qwen_pipeline()
