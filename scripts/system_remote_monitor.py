"""
Remote Telemetry & Benchmark Monitoring Daemon.

Generates an up-to-date status dashboard for Gemini Antigravity Remote Control:
- Hardware telemetry: CPU %, RAM (used/total), Apple Silicon Metal GPU %
- LM Studio model status (loaded model, context length, active state)
- Running benchmark process states and PIDs
- Live progress table for all 5 models across all 7 institutional benchmarks
- Terminal session log tails

Outputs:
- REMOTE_MONITOR.md (Repository root for direct mobile inspection)
- logs/REMOTE_MONITOR.md
- logs/system_telemetry.log (Historical append-only log)
"""

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import psutil

REPO_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = REPO_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_MD_ROOT = REPO_ROOT / "REMOTE_MONITOR.md"
TARGET_MD_LOGS = LOGS_DIR / "REMOTE_MONITOR.md"
TELEMETRY_LOG = LOGS_DIR / "system_telemetry.log"
LMS_BIN = Path.home() / ".lmstudio" / "bin" / "lms"


def get_gpu_utilization() -> str:
    """Reads Apple Silicon Metal GPU device utilization via ioreg."""
    try:
        cmd = "ioreg -r -d 1 -w 0 -c IOAccelerator"
        out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5).stdout
        matches = re.findall(r'"Device Utilization %"=(\d+)', out)
        if matches:
            return f"{matches[0]}%"
    except Exception:
        pass
    return "N/A"


def get_lms_status() -> str:
    """Fetches output from lms ps."""
    try:
        out = subprocess.run([str(LMS_BIN), "ps"], capture_output=True, text=True, timeout=5).stdout.strip()
        return out if out else "No models actively loaded in LM Studio."
    except Exception as e:
        return f"Error querying LMS: {e}"


def get_active_eval_processes() -> List[Dict[str, str]]:
    """Inspects running benchmark processes."""
    found = []
    for proc in psutil.process_iter(["pid", "name", "cmdline", "cpu_percent", "memory_percent", "create_time"]):
        try:
            cmd = " ".join(proc.info.get("cmdline") or [])
            if any(x in cmd for x in ["run_lfm_8b", "run_post_school", "run_granite", "run_afm3", "system_remote_monitor"]):
                created = datetime.fromtimestamp(proc.info["create_time"]).strftime("%H:%M:%S")
                # Basename of the script
                script_name = "unknown"
                for part in cmd.split():
                    if part.endswith(".py"):
                        script_name = Path(part).name
                        break
                found.append({
                    "pid": str(proc.info["pid"]),
                    "script": script_name,
                    "cpu": f"{proc.cpu_percent(interval=0.1):.1f}%",
                    "mem": f"{proc.memory_percent():.1f}%",
                    "started": created,
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return found


def get_benchmark_scorecard() -> Dict[str, Dict[str, str]]:
    """Compiles status across all 5 models for the 7 benchmarks."""
    models = [
        {"name": "Apple AFM 3 Core Advanced", "summary": LOGS_DIR / "eval_report_afm3_master_summary.json", "prefix": "eval_report_afm3_"},
        {"name": "IBM Granite 4.2 3B MLX", "summary": LOGS_DIR / "eval_report_granite_mlx_master_summary.json", "prefix": "eval_report_granite_mlx_"},
        {"name": "Liquid AI LFM 2.5 8B MLX", "summary": LOGS_DIR / "eval_report_lfm8b_master_summary.json", "prefix": "eval_report_lfm8b_"},
        {"name": "Microsoft Phi 4 Mini", "summary": LOGS_DIR / "eval_report_phi4_mini_master_summary.json", "prefix": "eval_report_phi4_mini_"},
        {"name": "Google Gemma 4 E4B", "summary": LOGS_DIR / "eval_report_gemma4_e4b_master_summary.json", "prefix": "eval_report_gemma4_e4b_"},
    ]

    benchmarks = [
        ("humaneval", "HumanEval"),
        ("tooluse", "Tool-Use"),
        ("cyber", "Cybersecurity"),
        ("gpqa_diamond", "GPQA Diamond"),
        ("mmlu_science", "MMLU Science"),
        ("competition_math", "MATH (Hendrycks)"),
        ("arc", "ARC-Challenge"),
    ]

    card = {}
    for m in models:
        card[m["name"]] = {}
        # Try loading master summary
        m_data = {}
        if m["summary"].exists():
            try:
                with open(m["summary"]) as f:
                    m_data = json.load(f)
            except Exception:
                pass

        for b_key, b_label in benchmarks:
            if b_key in m_data:
                val = m_data[b_key].get("pass_at_1_pct")
                card[m["name"]][b_label] = f"{val:.1f}%" if val is not None else "Done"
            else:
                # Check individual file
                ind_file = LOGS_DIR / f"{m['prefix']}{b_key}_full.json"
                if ind_file.exists():
                    try:
                        with open(ind_file) as f:
                            d = json.load(f)
                        sub = d.get("functional") or d.get("math") or d.get("science") or d.get("code") or d.get("cyber") or d.get("tooluse") or {}
                        val = sub.get("pass_at_1_pct") or sub.get("pass_rate_pct")
                        card[m["name"]][b_label] = f"{val:.1f}%" if val is not None else "Done"
                    except Exception:
                        card[m["name"]][b_label] = "Done"
                else:
                    card[m["name"]][b_label] = "Pending"

    return card


def get_file_tail(file_path: Path, lines: int = 20) -> str:
    """Returns last N lines of a file."""
    if not file_path.exists():
        return f"[Log file {file_path.name} does not exist yet]"
    try:
        res = subprocess.run(["tail", "-n", str(lines), str(file_path)], capture_output=True, text=True)
        return res.stdout.strip()
    except Exception as e:
        return f"Error reading tail: {e}"


def generate_dashboard() -> str:
    """Builds the full Markdown dashboard."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")
    timestamp = int(time.time())

    # Hardware stats
    cpu_pct = psutil.cpu_percent(interval=0.5)
    vm = psutil.virtual_memory()
    ram_total_gb = vm.total / (1024**3)
    ram_used_gb = vm.used / (1024**3)
    ram_avail_gb = vm.available / (1024**3)
    ram_pct = vm.percent
    gpu_str = get_gpu_utilization()

    # LM Studio
    lms_raw = get_lms_status()

    # Processes
    procs = get_active_eval_processes()

    # Scorecard
    scorecard = get_benchmark_scorecard()

    # Log Tails
    lfm_tail = get_file_tail(LOGS_DIR / "run_lfm_8b_full_benches.log", lines=18)
    pipeline_tail = get_file_tail(LOGS_DIR / "run_post_school_master_pipeline.log", lines=12)

    lines = [
        "# Télos Live Remote Monitor & Benchmark Dashboard",
        f"> **Last Updated**: `{now_str}` (Epoch: `{timestamp}`)",
        f"> **Control Channel**: Gemini Antigravity Remote Control",
        "",
        "## 1. System Hardware & Memory Telemetry",
        "",
        "| Metric | Current Value | Target Threshold / Constraint |",
        "| :--- | :--- | :--- |",
        f"| **CPU Utilization** | **`{cpu_pct:.1f}%`** | Multi-core Apple Silicon |",
        f"| **Unified RAM Used** | **`{ram_used_gb:.2f} GB`** / `{ram_total_gb:.2f} GB` (`{ram_pct:.1f}%`) | Safe Headroom: `{ram_avail_gb:.2f} GB` free |",
        f"| **GPU / Metal Utilization** | **`{gpu_str}`** | Apple Silicon Metal Core |",
        "| **Context Length Lock** | **`8,192 Tokens`** | Strict Zero-OOM Ceiling |",
        "",
        "## 2. LM Studio Engine State",
        "```text",
        lms_raw,
        "```",
        "",
        "## 3. Active Benchmark Processes",
    ]

    if procs:
        lines.extend([
            "| PID | Script | CPU % | RAM % | Started At |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ])
        for p in procs:
            lines.append(f"| `{p['pid']}` | **`{p['script']}`** | `{p['cpu']}` | `{p['mem']}` | `{p['started']}` |")
    else:
        lines.append("*No active evaluation processes currently running.*")

    lines.extend([
        "",
        "## 4. Multi-Model Benchmark Scorecard (7 Suites)",
        "",
        "| Model | HumanEval | Tool-Use | Cyber | GPQA | MMLU | MATH | ARC |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for model_name, b_map in scorecard.items():
        row = [f"**{model_name}**"]
        for b_col in ["HumanEval", "Tool-Use", "Cybersecurity", "GPQA Diamond", "MMLU Science", "MATH (Hendrycks)", "ARC-Challenge"]:
            val = b_map.get(b_col, "Pending")
            row.append(f"`{val}`" if val != "Pending" else "⏳ Pending")
        lines.append(f"| {' | '.join(row)} |")

    lines.extend([
        "",
        "## 5. Live Terminal Logs",
        "",
        "### `logs/run_lfm_8b_full_benches.log` (LFM 2.5 8B Execution)",
        "```text",
        lfm_tail,
        "```",
        "",
        "### `logs/run_post_school_master_pipeline.log` (Master Pipeline Standby)",
        "```text",
        pipeline_tail,
        "```",
        "",
        "---",
        "*Automated telemetry updated every 30 minutes via scheduled daemon.*",
    ])

    content = "\n".join(lines) + "\n"

    # Write Markdown files
    TARGET_MD_ROOT.write_text(content)
    TARGET_MD_LOGS.write_text(content)

    # Append to telemetry log
    with open(TELEMETRY_LOG, "a") as f:
        f.write(f"[{now_str}] CPU={cpu_pct:.1f}% | RAM={ram_used_gb:.2f}/{ram_total_gb:.2f}GB ({ram_pct:.1f}%) | GPU={gpu_str} | LMS={lms_raw.splitlines()[1] if len(lms_raw.splitlines()) > 1 else 'N/A'}\n")

    return content


def run_daemon(interval_seconds: int = 1800):
    """Runs continuous monitoring loop every interval_seconds (default 30 mins)."""
    print(f"[Remote Monitor] Daemon started. Updating every {interval_seconds//60} minutes...")
    while True:
        try:
            generate_dashboard()
            print(f"[Remote Monitor] Updated status dashboard at {datetime.now().strftime('%H:%M:%S')}")
        except Exception as e:
            print(f"[Remote Monitor Error]: {e}", file=sys.stderr)
        time.sleep(interval_seconds)


if __name__ == "__main__":
    # If run once with --once, generate immediately and exit
    if "--once" in sys.argv:
        generate_dashboard()
        print(f"[OK] Dashboard generated at {TARGET_MD_ROOT}")
    else:
        run_daemon(interval_seconds=1800)
