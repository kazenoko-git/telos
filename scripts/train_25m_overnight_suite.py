"""Master Sequential Runner for Overnight 25M Parameter Ratio Study.

Executes 4 training runs back-to-back on Apple MLX:
1. 1:3 Ratio  (75M Tokens   — ~25.5 mins)
2. 1:10 Ratio (250M Tokens  — ~1.48 hours)
3. 1:15 Ratio (375M Tokens  — ~2.22 hours)
4. 1:20 Ratio (500M Tokens  — ~2.95 hours)

Logs results, loss stats, and 100 contextual probes summaries to logs/overnight_25m_ratio_study.log.
"""

import sys
import time
import subprocess
from pathlib import Path

# Ensure logs directory exists
Path("logs").mkdir(exist_ok=True)
LOG_FILE = Path("logs/overnight_25m_ratio_study.log")

SUITE = [
    ("1:3 Ratio (75M Tokens)", "configs/phase_b_25m_1to3_mlx.yaml", "checkpoints/phase_b_25m_1to3_mlx"),
    ("1:10 Ratio (250M Tokens)", "configs/phase_b_25m_1to10_mlx.yaml", "checkpoints/phase_b_25m_1to10_mlx"),
    ("1:15 Ratio (375M Tokens)", "configs/phase_b_25m_1to15_mlx.yaml", "checkpoints/phase_b_25m_1to15_mlx"),
    ("1:20 Ratio (500M Tokens)", "configs/phase_b_25m_1to20_mlx.yaml", "checkpoints/phase_b_25m_1to20_mlx"),
]


def log(msg: str):
    timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{timestamp} {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def main():
    log("==========================================================================")
    log(" STARTING OVERNIGHT 25M PARAMETER RATIO STUDY (1:3, 1:10, 1:15, 1:20)")
    log(" Total Tokens to Train: 1.2 Billion Tokens across 4 Models")
    log("==========================================================================")

    suite_start = time.time()

    for idx, (label, config_path, ckpt_dir) in enumerate(SUITE, start=1):
        log(f"\n[{idx}/4] STARTING TRAINING: {label}")
        log(f"      Config: {config_path}")
        run_start = time.time()

        # Run training loop
        cmd_train = ["uv", "run", "python", "scripts/train_mlx.py", "--config", config_path]
        proc = subprocess.run(cmd_train, capture_output=False)

        if proc.returncode != 0:
            log(f"ERROR: Training failed for {label} with return code {proc.returncode}")
            continue

        run_elapsed = (time.time() - run_start) / 60.0
        log(f"SUCCESS: Completed training {label} in {run_elapsed:.2f} minutes.")

        # Run 100 contextual probes evaluation
        log(f"      Running 100 Contextual Probes for {label}...")
        cmd_probe = ["uv", "run", "python", "scripts/sample.py", "--checkpoint", ckpt_dir, "--mode", "probes"]
        probe_res = subprocess.run(cmd_probe, capture_output=True, text=True)

        if probe_res.returncode == 0:
            log(f"\n--- Probe Results for {label} ---\n" + probe_res.stdout)
        else:
            log(f"Warning: Probes evaluation failed for {label}: {probe_res.stderr}")

    total_elapsed_hrs = (time.time() - suite_start) / 3600.0
    log("\n==========================================================================")
    log(f" OVERNIGHT 25M RATIO STUDY COMPLETED IN {total_elapsed_hrs:.2f} HOURS!")
    log("==========================================================================")


if __name__ == "__main__":
    main()
