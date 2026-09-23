"""
Polyglot & Extended Code Benchmark Orchestrator for Apple AFM 3 Core Advanced.

Executes unconstrained full-suite evaluations across:
1. C# (.NET) HumanEval (158 tasks)
2. Java HumanEval (158 tasks)
3. JavaScript HumanEval (161 tasks)
4. TypeScript HumanEval (159 tasks)
5. Rust HumanEval (156 tasks)
6. React Frontend JSX (50 tasks)
7. MBPP Sanitized Python (427 tasks)
8. Private Unseen Master Suite (512 tasks)

Saves standalone JSON reports per suite and prints a final scorecard.
"""

import json
import time
from pathlib import Path
from telos.eval.runner import evaluate

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

POLYGLOT_SUITES = [
    {"name": "humaneval_cs", "label": "MultiPL-E C# (.NET)", "type": "code", "output": "eval_report_afm3_humaneval_cs_full.json"},
    {"name": "humaneval_java", "label": "MultiPL-E Java", "type": "code", "output": "eval_report_afm3_humaneval_java_full.json"},
    {"name": "humaneval_js", "label": "MultiPL-E JavaScript", "type": "code", "output": "eval_report_afm3_humaneval_js_full.json"},
    {"name": "humaneval_ts", "label": "MultiPL-E TypeScript", "type": "code", "output": "eval_report_afm3_humaneval_ts_full.json"},
    {"name": "humaneval_rust", "label": "MultiPL-E Rust", "type": "code", "output": "eval_report_afm3_humaneval_rust_full.json"},
    {"name": "react", "label": "React Frontend Component Suite", "type": "code", "output": "eval_report_afm3_react_full.json"},
    {"name": "mbpp", "label": "MBPP Python Sanitized", "type": "code", "output": "eval_report_afm3_mbpp_full.json"},
    {"name": "private_unseen", "label": "Private Unseen Anti-Cheat Suite", "type": "code", "output": "eval_report_afm3_private_unseen_full.json"},
]

def main():
    print("\n" + "=" * 90)
    print("  TÉLOS POLYGLOT & EXTENDED CODE BENCHMARK SUITE — APPLE AFM 3 CORE ADVANCED")
    print("  Runtime Backend: Native Swift FoundationModels.framework IPC")
    print("  Execution Mode:  Unconstrained Full Benchmark (max_tasks = None)")
    print("=" * 90 + "\n")

    summary_results = {}

    for s in POLYGLOT_SUITES:
        suite_name = s["name"]
        label = s["label"]
        b_type = s["type"]
        out_file = LOGS_DIR / s["output"]

        print("\n" + "#" * 90)
        print(f"  STARTING FULL BENCHMARK: {label.upper()}")
        print(f"  Suite: {suite_name} | Type: {b_type} | Destination: {out_file}")
        print("#" * 90 + "\n")

        t0 = time.time()
        try:
            report = evaluate(
                checkpoint="afm-3-core-advanced",
                mode="functional",
                suite=suite_name,
                benchmark_type=b_type,
                max_tasks=None,
                backend="swift_afm",
                output_path=str(out_file),
            )
            elapsed = time.time() - t0
            summary_results[suite_name] = {
                "label": label,
                "status": "completed",
                "elapsed_seconds": round(elapsed, 1),
                "report_file": str(out_file),
            }
            print(f"\n[OK] Completed {label} in {elapsed/60.0:.2f} minutes.")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"\n[FAILED] Error in {label}: {e}")
            summary_results[suite_name] = {
                "label": label,
                "status": f"failed: {e}",
                "elapsed_seconds": round(elapsed, 1),
            }

    # Consolidated Master Summary
    master_file = LOGS_DIR / "eval_report_afm3_polyglot_summary.json"
    with open(master_file, "w") as f:
        json.dump(summary_results, f, indent=2)

    print("\n" + "=" * 90)
    print("  ALL POLYGLOT BENCHMARKS COMPLETED FOR AFM 3 CORE ADVANCED")
    print(f"  Master Summary Saved to: {master_file}")
    print("=" * 90 + "\n")

if __name__ == "__main__":
    main()
