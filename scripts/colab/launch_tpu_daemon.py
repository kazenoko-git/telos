"""
Launches the standalone TPU training process detached from the kernel.
"""
import os
import sys
import subprocess
import time
from pathlib import Path

work_dir = Path("/content/telos")
os.chdir(str(work_dir))

log_file = work_dir / "training.log"

# Check if training is already running
res = subprocess.run(["ps", "-ef"], capture_output=True, text=True)
running_pid = None
for line in res.stdout.splitlines():
    if "train_25m_upscaled.py" in line:
        parts = line.split()
        running_pid = parts[1]
        print(f"Training already active (PID: {running_pid})", flush=True)
        break

if not running_pid:
    # Unpack latest code bundle
    from huggingface_hub import hf_hub_download
    token = os.environ.get("HF_TOKEN")
    bundle_path = hf_hub_download(repo_id="Kazenowoko/telos", filename="telos_code.tar.gz", token=token, local_dir=str(work_dir))
    subprocess.run(["tar", "-xzf", str(bundle_path), "-C", str(work_dir)], check=True)
    
    with open(log_file, "a") as f_log:
        f_log.write("\n=== LAUNCHING STANDALONE TPU DAEMON ===\n")
        
    proc = subprocess.Popen(
        [sys.executable, "scripts/colab/train_25m_upscaled.py", "--ratios", "r1", "r10", "--hf-repo", "Kazenowoko/telos", "--device", "tpu"],
        stdout=open(log_file, "a"),
        stderr=subprocess.STDOUT,
        cwd=str(work_dir),
        env=os.environ.copy()
    )
    print(f"Spawned Standalone TPU Daemon Process! (PID: {proc.pid})", flush=True)
    time.sleep(2)

# Print last lines of training.log
if log_file.exists():
    lines = log_file.read_text().splitlines()
    print("\n--- LATEST TRAINING LOG ---", flush=True)
    print("\n".join(lines[-20:]), flush=True)
