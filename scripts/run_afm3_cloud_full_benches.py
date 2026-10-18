"""
Master Benchmark Orchestrator for Apple AFM 3 Cloud (Private Cloud Compute).

Runs full 7-suite evaluation across:
1. OpenAI HumanEval (164 tasks)
2. BFCL Tool-Use (30 tasks)
3. Cybersecurity OWASP/CWE Auditing (50 tasks)
4. GPQA Diamond (198 tasks)
5. Competition MATH (140 tasks)
6. ARC-Challenge (150 tasks)
7. MMLU Science (119 tasks)

Saves standalone JSON reports per suite and outputs a master summary.
"""

import os
import json
import time
import subprocess
from pathlib import Path
from telos.eval.adapters.base import BaseModelAdapter
from telos.eval.runner import evaluate

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)
SCRATCH_DIR = Path("scratch")
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)


class AFMCloudAdapter(BaseModelAdapter):
    """Adapter for Apple AFM 3 Cloud via Private Cloud Compute macOS shortcut."""

    def __init__(self, shortcut_name: str = "AFM3Cloud"):
        super().__init__(model_name="afm-3-cloud", backend_name="shortcuts_afm_cloud")
        self.shortcut_name = shortcut_name
        self._counter = 0

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        stop=None,
        **kwargs
    ) -> str:
        self._counter += 1
        pid = os.getpid()
        t_ms = int(time.time() * 1000)
        in_file = SCRATCH_DIR / f"_afm_c_in_{pid}_{self._counter}_{t_ms}.txt"
        out_file = SCRATCH_DIR / f"_afm_c_out_{pid}_{self._counter}_{t_ms}.txt"

        in_file.write_text(prompt, encoding="utf-8")
        if out_file.exists():
            out_file.unlink()

        try:
            for attempt in range(2):
                res = subprocess.run(
                    ["shortcuts", "run", self.shortcut_name, "-i", str(in_file), "-o", str(out_file)],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if out_file.exists():
                    content = out_file.read_text(encoding="utf-8").strip()
                    if content:
                        return content
                time.sleep(0.5)
            return ""
        except Exception as exc:
            print(f"  [AFM 3 Cloud Warning] Request failed: {exc}")
            return ""
        finally:
            if in_file.exists():
                in_file.unlink()
            if out_file.exists():
                out_file.unlink()


SUITES = [
    {
        "key": "humaneval",
        "name": "humaneval",
        "label": "OpenAI HumanEval (Python Code — 164 Tasks)",
        "type": "code",
        "output": "eval_report_afm3_cloud_humaneval_full.json",
    },
    {
        "key": "tooluse",
        "name": "tooluse",
        "label": "BFCL Tool-Use & Function Calling (30 Tasks)",
        "type": "tooluse",
        "output": "eval_report_afm3_cloud_tooluse_full.json",
    },
    {
        "key": "cyber",
        "name": "cybersecurity",
        "label": "Cybersecurity OWASP/CWE Auditing (50 Tasks)",
        "type": "cyber",
        "output": "eval_report_afm3_cloud_cyber_full.json",
    },
    {
        "key": "gpqa_diamond",
        "name": "gpqa_diamond",
        "label": "GPQA Diamond (PhD Science — 198 Tasks)",
        "type": "science",
        "output": "eval_report_afm3_cloud_gpqa_diamond_full.json",
    },
    {
        "key": "competition_math",
        "name": "competition_math",
        "label": "Competition MATH (Hendrycks — 140 Tasks)",
        "type": "math",
        "output": "eval_report_afm3_cloud_competition_math_full.json",
    },
    {
        "key": "arc",
        "name": "arc_challenge",
        "label": "ARC-Challenge Science Reasoning (150 Tasks)",
        "type": "science",
        "output": "eval_report_afm3_cloud_arc_full.json",
    },
    {
        "key": "mmlu_science",
        "name": "mmlu_science",
        "label": "MMLU Science & STEM (119 Tasks)",
        "type": "science",
        "output": "eval_report_afm3_cloud_mmlu_science_full.json",
    },
]


def extract_score(report: dict, suite_key: str, b_type: str) -> float:
    """Extracts pass_at_1_pct score from report dict."""
    if suite_key == "humaneval":
        return report.get("code", {}).get("functional", {}).get("pass_at_1_pct", 0.0)
    elif suite_key == "tooluse":
        return report.get("tooluse", {}).get("pass_at_1_pct", 0.0)
    elif suite_key == "cyber":
        return report.get("cyber", {}).get("pass_at_1_pct", 0.0)
    elif suite_key in ["gpqa_diamond", "arc", "mmlu_science"]:
        return report.get("science", {}).get("pass_at_1_pct", 0.0)
    elif suite_key == "competition_math":
        return report.get("math", {}).get("pass_at_1_pct", 0.0)
    return 0.0


def main():
    print("=" * 80)
    print("  AFM 3 CLOUD (PRIVATE CLOUD COMPUTE) MASTER BENCHMARK EVALUATOR")
    print("  Comparing Against Published Frontier Leaderboard")
    print("=" * 80)

    adapter = AFMCloudAdapter()
    master_summary = {}

    for s in SUITES:
        suite_key = s["key"]
        suite_name = s["name"]
        label = s["label"]
        b_type = s["type"]
        out_file = LOGS_DIR / s["output"]

        print("\n" + "-" * 80)
        print(f"  LAUNCHING: {label.upper()}")
        print(f"  Destination: {out_file}")
        print("-" * 80)

        t0 = time.time()
        try:
            report = evaluate(
                checkpoint=adapter,
                mode="functional",
                suite=suite_name,
                benchmark_type=b_type,
                max_tasks=None,
                output_path=str(out_file),
            )
            elapsed = time.time() - t0
            score = extract_score(report, suite_key, b_type)

            master_summary[suite_key] = {
                "label": label,
                "pass_at_1_pct": score,
                "elapsed_seconds": round(elapsed, 1),
                "report_file": str(out_file),
                "status": "completed",
            }
            print(f"\n[OK] Completed {label} in {elapsed/60.0:.2f} min | Pass@1: {score:.2f}%")
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"\n[FAILED] Error in {label}: {exc}")
            master_summary[suite_key] = {
                "label": label,
                "status": f"failed: {exc}",
                "elapsed_seconds": round(elapsed, 1),
            }

        # Save checkpointed summary after each suite
        summary_path = LOGS_DIR / "eval_report_afm3_cloud_master_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(master_summary, f, indent=2)

    # Print final scorecard
    print("\n" + "=" * 80)
    print("  AFM 3 CLOUD FINAL SCORECARD")
    print("=" * 80)
    scores = []
    for s in SUITES:
        k = s["key"]
        info = master_summary.get(k, {})
        sc = info.get("pass_at_1_pct", 0.0)
        scores.append(sc)
        print(f"  · {s['label']:<48}: {sc:.2f}%")

    if scores:
        macro = sum(scores) / len(scores)
        print("-" * 80)
        print(f"  MACRO AVERAGE (7 SUITES): {macro:.2f}%")
        print("=" * 80)


if __name__ == "__main__":
    main()
