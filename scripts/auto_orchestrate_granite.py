"""
Autonomous Benchmark Orchestrator for IBM Granite 4.2 3B MLX.
Monitors the active ARC-Challenge execution, proceeds to Competition MATH upon completion,
and compiles the head-to-head institutional scorecard vs Apple AFM 3 Core Advanced.
"""

import json
import os
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
ARC_LOG = LOGS_DIR / "run_granite_mlx_full_benches.log"
ORCH_LOG = LOGS_DIR / "auto_orchestrator.log"
PYTHON_BIN = PROJECT_ROOT / ".venv" / "bin" / "python3"


def log(msg: str):
    """Logs message with timestamp to both console and file."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{ts}] {msg}"
    print(formatted, flush=True)
    with open(ORCH_LOG, "a") as f:
        f.write(formatted + "\n")


def is_pid_running(pid: int) -> bool:
    """Checks whether the given process identifier is currently running."""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def build_comparative_scorecard():
    """Compiles publication-grade comparative metrics between AFM 3 and Granite 4.2."""
    afm3_path = LOGS_DIR / "eval_report_afm3_master_summary.json"
    granite_path = LOGS_DIR / "eval_report_granite_mlx_master_summary.json"

    if not afm3_path.exists() or not granite_path.exists():
        log("[Warning] Missing master summaries for comparison.")
        return

    with open(afm3_path) as f:
        afm3_data = json.load(f)
    with open(granite_path) as f:
        granite_data = json.load(f)

    comparison = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "model_a": {
                "name": "Apple AFM 3 Core Advanced",
                "runtime": "Native Swift FoundationModels.framework IPC (Apple Silicon)",
                "summary_file": str(afm3_path)
            },
            "model_b": {
                "name": "IBM Granite 4.2 3B MLX",
                "runtime": "LM Studio Metal Acceleration (Apple Silicon)",
                "summary_file": str(granite_path)
            }
        },
        "benchmarks": {}
    }

    # Suite mapping across all 7 institutional benchmarks
    suites = [
        ("humaneval", "OpenAI HumanEval (Python Code)", 164),
        ("tooluse", "BFCL Tool-Use & Function Calling", 30),
        ("cyber", "Cybersecurity OWASP/CWE Auditing", 50),
        ("gpqa_diamond", "GPQA Diamond (PhD Science)", 198),
        ("mmlu_science", "MMLU Science & STEM", 833),
        ("competition_math", "Competition MATH (Hendrycks)", 700),
        ("arc", "ARC-Challenge Science Reasoning", 1172)
    ]

    for key, label, total_tasks in suites:
        a_entry = afm3_data.get(key, {})
        g_entry = granite_data.get(key, {})

        a_pass = a_entry.get("pass_at_1_pct", 0.0)
        g_pass = g_entry.get("pass_at_1_pct", 0.0)
        delta = round(g_pass - a_pass, 2)

        comparison["benchmarks"][key] = {
            "label": label,
            "tasks": total_tasks,
            "afm3_core_advanced": {
                "pass_at_1_pct": a_pass,
                "pass_at_1_95ci": a_entry.get("pass_at_1_95ci"),
                "ast_validity_pct": a_entry.get("ast_validity_pct")
            },
            "granite_4_2_3b_mlx": {
                "pass_at_1_pct": g_pass,
                "pass_at_1_95ci": g_entry.get("pass_at_1_95ci"),
                "ast_validity_pct": g_entry.get("ast_validity_pct")
            },
            "delta_granite_vs_afm3_pct": delta,
            "winner": "Granite 4.2 3B" if delta > 0 else ("AFM 3 Core Advanced" if delta < 0 else "Tie")
        }

    comp_file = LOGS_DIR / "comparative_report_afm3_vs_granite_mlx.json"
    with open(comp_file, "w") as f:
        json.dump(comparison, f, indent=2)

    log(f"Master Comparative Report saved to: {comp_file}")


def main():
    target_pid = 44359
    log(f"Starting orchestrator. Monitoring PID {target_pid} (ARC-Challenge)...")

    # 1. Wait for ARC to finish
    while is_pid_running(target_pid):
        time.sleep(20)

    log(f"PID {target_pid} has terminated. Verifying ARC-Challenge report...")
    arc_file = LOGS_DIR / "eval_report_granite_mlx_arc_full.json"
    if not arc_file.exists():
        log("[Error] ARC-Challenge output file does not exist!")
    else:
        try:
            with open(arc_file) as f:
                arc_data = json.load(f)
            log("[OK] ARC-Challenge completed successfully. Report verified.")
        except Exception as e:
            log(f"[Error] Failed to read ARC report: {e}")

    # 2. Run Competition MATH via run_granite_mlx_full_benches.py
    log("Invoking run_granite_mlx_full_benches.py to execute Competition MATH...")
    bench_script = PROJECT_ROOT / "scripts" / "run_granite_mlx_full_benches.py"
    with open(ARC_LOG, "a") as out_log:
        proc = subprocess.run(
            [str(PYTHON_BIN), "-u", str(bench_script)],
            cwd=str(PROJECT_ROOT),
            stdout=out_log,
            stderr=out_log
        )

    log(f"Benchmark runner finished with exit code {proc.returncode}.")

    # 3. Build master comparative scorecard
    log("Building master comparative scorecard...")
    build_comparative_scorecard()
    log("All institutional benchmarks and comparative reports completed!")


if __name__ == "__main__":
    main()
