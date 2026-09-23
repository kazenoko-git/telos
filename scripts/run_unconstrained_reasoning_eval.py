"""
Unconstrained Reasoning Benchmark Orchestrator.

Evaluates deep reasoning suites with extended 4,096-token budgets to eliminate
mid-derivation truncation:
1. IBM Granite 4.2 3B MLX (LM Studio at localhost:1234)
   - Competition MATH (700 tasks)
   - GPQA Diamond PhD Science (198 tasks)
2. Apple AFM 3 Core Advanced (Native Swift FoundationModels IPC)
   - Competition MATH (700 tasks)
   - GPQA Diamond PhD Science (198 tasks)

Generates comparative scorecards measuring 1,024-token vs. 4,096-token scaling.
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

from telos.eval.adapters.openai_api import OpenAIAPIAdapter
from telos.eval.adapters.swift_afm import SwiftAFMAdapter
from telos.eval.runner import evaluate

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

LMS_BIN = Path.home() / ".lmstudio" / "bin" / "lms"


class GraniteLMStudioAdapter(OpenAIAPIAdapter):
    """LM Studio adapter for IBM Granite 4.2 3B MLX with token hygiene."""

    def __init__(
        self,
        model_name: str = "granite-4.2-3b-mlx",
        api_base: str = "http://127.0.0.1:1234/v1",
        system_prompt: str = (
            "You are an expert mathematician and scientist. "
            "Solve the problem step by step and provide the final answer clearly in \\boxed{...}."
        ),
        timeout: float = 180.0,
    ):
        super().__init__(
            model_name=model_name,
            api_base=api_base,
            system_prompt=system_prompt,
            timeout=timeout,
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        stop: Any = None,
        **kwargs,
    ) -> str:
        # Exclude newline stop tokens to allow full reasoning chains
        stop_list = stop or kwargs.pop("stop_words", None) or kwargs.pop("stop", None)
        clean_stop = [s for s in stop_list if s != "\n\n"] if stop_list else None

        for attempt in range(3):
            try:
                out = super().generate(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    stop=clean_stop,
                    **kwargs,
                )
                return out
            except Exception as e:
                print(f"  [Granite Retry attempt {attempt}]: {e}", file=sys.stderr)
                time.sleep(1.0 + attempt * 2.0)
                if attempt == 2:
                    return ""
        return ""


def run_granite_evaluations() -> Dict[str, Any]:
    """Runs unconstrained reasoning evaluations for IBM Granite 4.2 3B MLX."""
    print("\n" + "=" * 90)
    print("  PHASE 1: IBM GRANITE 4.2 3B MLX (4,096-TOKEN UNCONSTRAINED REASONING)")
    print("=" * 90 + "\n")

    adapter = GraniteLMStudioAdapter()
    results = {}

    suites = [
        {
            "name": "competition_math",
            "label": "Competition MATH (Hendrycks — 4,096 Tokens)",
            "type": "math",
            "output": "eval_report_granite_mlx_competition_math_unconstrained.json",
        },
        {
            "name": "gpqa_diamond",
            "label": "GPQA Diamond (PhD Science — 4,096 Tokens)",
            "type": "science",
            "output": "eval_report_granite_mlx_gpqa_diamond_unconstrained.json",
        },
    ]

    for s in suites:
        out_path = LOGS_DIR / s["output"]
        print(f"\n>>> Starting {s['label']} -> {out_path}")
        t0 = time.time()
        rep = evaluate(
            checkpoint=adapter,
            mode="functional",
            suite=s["name"],
            benchmark_type=s["type"],
            max_tasks=None,
            max_new_tokens=4096,
            output_path=str(out_path),
        )
        elapsed = time.time() - t0
        score = rep.get(s["name"], {}).get("pass_at_1_pct", 0.0)
        results[s["name"]] = {
            "label": s["label"],
            "score": score,
            "elapsed_minutes": round(elapsed / 60.0, 2),
            "file": str(out_path),
        }
        print(f"[OK] Finished {s['label']} in {elapsed/60.0:.2f} mins (Pass@1: {score:.2f}%)")

    return results


def unload_granite():
    """Unloads Granite from LM Studio to free unified memory for Apple AFM 3."""
    print("\n>>> Unloading Granite 4.2 3B MLX from LM Studio...")
    try:
        subprocess.run(
            [str(LMS_BIN), "unload", "granite-4.2-3b-mlx"],
            check=True,
            capture_output=True,
            text=True,
        )
        print("[OK] Granite unloaded successfully.")
    except Exception as e:
        print(f"Warning: Failed to unload Granite via LMS CLI: {e}")


def run_afm3_evaluations() -> Dict[str, Any]:
    """Runs unconstrained reasoning evaluations for Apple AFM 3 Core Advanced."""
    print("\n" + "=" * 90)
    print("  PHASE 2: APPLE AFM 3 CORE ADVANCED (4,096-TOKEN UNCONSTRAINED REASONING)")
    print("  Runtime: Native FoundationModels.framework Swift IPC on Apple Silicon")
    print("=" * 90 + "\n")

    adapter = SwiftAFMAdapter("afm-3-core-advanced")
    results = {}

    suites = [
        {
            "name": "competition_math",
            "label": "Competition MATH (Hendrycks — 4,096 Tokens)",
            "type": "math",
            "output": "eval_report_afm3_competition_math_unconstrained.json",
        },
        {
            "name": "gpqa_diamond",
            "label": "GPQA Diamond (PhD Science — 4,096 Tokens)",
            "type": "science",
            "output": "eval_report_afm3_gpqa_diamond_unconstrained.json",
        },
    ]

    for s in suites:
        out_path = LOGS_DIR / s["output"]
        print(f"\n>>> Starting {s['label']} -> {out_path}")
        t0 = time.time()
        rep = evaluate(
            checkpoint=adapter,
            mode="functional",
            suite=s["name"],
            benchmark_type=s["type"],
            max_tasks=None,
            max_new_tokens=4096,
            output_path=str(out_path),
        )
        elapsed = time.time() - t0
        score = rep.get(s["name"], {}).get("pass_at_1_pct", 0.0)
        results[s["name"]] = {
            "label": s["label"],
            "score": score,
            "elapsed_minutes": round(elapsed / 60.0, 2),
            "file": str(out_path),
        }
        print(f"[OK] Finished {s['label']} in {elapsed/60.0:.2f} mins (Pass@1: {score:.2f}%)")

    adapter.close()
    return results


def compile_comparison(granite_res: Dict[str, Any], afm3_res: Dict[str, Any]):
    """Compiles consolidated 1,024-token vs 4,096-token token scaling comparison."""
    summary_path = LOGS_DIR / "eval_report_unconstrained_reasoning_comparison.json"
    data = {
        "timestamp": time.time(),
        "granite_4_2_3b_mlx": granite_res,
        "apple_afm_3_core_advanced": afm3_res,
    }
    with open(summary_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n[OK] Unconstrained reasoning comparison saved to: {summary_path}")


def main():
    t_start = time.time()
    granite_res = run_granite_evaluations()
    unload_granite()
    afm3_res = run_afm3_evaluations()
    compile_comparison(granite_res, afm3_res)
    total_min = (time.time() - t_start) / 60.0
    print(f"\n{'='*90}\nALL UNCONSTRAINED EVALUATIONS COMPLETED IN {total_min:.1f} MINUTES\n{'='*90}\n")


if __name__ == "__main__":
    main()
