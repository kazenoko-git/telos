"""Master TPU v6e-1 Sequential Training & HuggingFace Upload Suite for 50M Model.

Executes:
1. 1:30 Ratio (1.5 Billion Tokens — ~26 mins on TPU)
2. 1:40 Ratio (2.0 Billion Tokens — ~35 mins on TPU)
3. 1:20 Ratio (1.0 Billion Tokens — ~17 mins on TPU, if time permits)

Ensures:
- PyTorch-XLA device auto-verification (xm.xla_device()) — ZERO fallback to CPU!
- Dynamic progress logging & evaluation.
- Auto-upload of trained PyTorch checkpoints to HuggingFace Hub using HF Write token.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

# Ensure logs directory exists
Path("logs").mkdir(exist_ok=True)
LOG_FILE = Path("logs/tpu_50m_ratio_study.log")

HF_WRITE_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO_ID = "kazenoko/telos-50m-ratio-study"

SUITE = [
    ("1:30 Ratio (1.5B Tokens)", "configs/phase_b_50m_1to30_tpu.yaml", "checkpoints/phase_b_50m_1to30_tpu"),
    ("1:40 Ratio (2.0B Tokens)", "configs/phase_b_50m_1to40_tpu.yaml", "checkpoints/phase_b_50m_1to40_tpu"),
    ("1:20 Ratio (1.0B Tokens)", "configs/phase_b_50m_1to20_tpu.yaml", "checkpoints/phase_b_50m_1to20_tpu"),
]


def log(msg: str):
    timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{timestamp} {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def verify_tpu():
    """Verifies that PyTorch-XLA detects the TPU v6e-1 device."""
    try:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        log(f">> VERIFIED TPU XLA DEVICE: {device}")
        return device
    except Exception as e:
        log(f"CRITICAL ERROR: Failed to detect TPU XLA device: {e}")
        sys.exit(1)


def upload_to_huggingface(ckpt_dir: str, label: str):
    """Uploads trained checkpoint folder to HuggingFace Hub."""
    log(f"      Uploading {label} to HuggingFace Hub ({HF_REPO_ID})...")
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        # Create repo if not exists
        api.create_repo(repo_id=HF_REPO_ID, token=HF_WRITE_TOKEN, exist_ok=True, private=False)
        
        folder_path = Path(ckpt_dir)
        if folder_path.exists():
            api.upload_folder(
                folder_path=str(folder_path),
                path_in_repo=folder_path.name,
                repo_id=HF_REPO_ID,
                token=HF_WRITE_TOKEN
            )
            log(f"      SUCCESS: Uploaded {label} to https://huggingface.co/{HF_REPO_ID}/{folder_path.name}")
        else:
            log(f"Warning: Checkpoint folder {ckpt_dir} does not exist for upload.")
    except Exception as e:
        log(f"Warning: HuggingFace upload failed for {label}: {e}")


def main():
    log("==========================================================================")
    log(" STARTING TPU v6e-1 50M PARAMETER RATIO STUDY (1:30, 1:40, 1:20)")
    log("==========================================================================")

    # 1. Verify TPU Device
    verify_tpu()

    suite_start = time.time()

    for idx, (label, config_path, ckpt_dir) in enumerate(SUITE, start=1):
        log(f"\n[{idx}/3] STARTING TRAINING ON TPU v6e-1: {label}")
        log(f"      Config: {config_path}")
        run_start = time.time()

        # Run TPU Training via scripts/train.py --device tpu
        cmd_train = [sys.executable, "scripts/train.py", "--config", config_path, "--device", "tpu", "--model-size", "50M"]
        proc = subprocess.run(cmd_train, capture_output=False)

        if proc.returncode != 0:
            log(f"ERROR: TPU Training failed for {label} with return code {proc.returncode}")
            continue

        run_elapsed = (time.time() - run_start) / 60.0
        log(f"SUCCESS: Completed TPU training {label} in {run_elapsed:.2f} minutes.")

        # Upload to HuggingFace
        upload_to_huggingface(ckpt_dir, label)

    total_elapsed_mins = (time.time() - suite_start) / 60.0
    log("\n==========================================================================")
    log(f" TPU v6e-1 50M RATIO STUDY COMPLETED IN {total_elapsed_mins:.2f} MINUTES!")
    log("==========================================================================")


if __name__ == "__main__":
    main()
