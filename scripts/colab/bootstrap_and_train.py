"""
Self-contained Google Colab Bootstrap & Background Trainer on TPU v5e-1.

Downloads code bundle & dataset from HuggingFace (Kazenowoko/telos),
unpacks codebase, and launches PyTorch-XLA TPU 25M Retraining Suite.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

work_dir = Path("/content/telos")
work_dir.mkdir(parents=True, exist_ok=True)
os.chdir(str(work_dir))
sys.path.insert(0, str(work_dir))

HF_TOKEN = os.environ.get("HF_TOKEN")

print("=" * 80, flush=True)
print(f"BOOTSTRAPPING COLAB TPU ENVIRONMENT AT {work_dir}...", flush=True)
print("=" * 80, flush=True)

# 1. Install Dependencies
print("\n[1/4] Installing Required Packages...", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub", "safetensors", "pyyaml", "tokenizers", "numpy", "matplotlib"], check=True)

# 2. Download Codebase Bundle, Weights & Dataset from Hugging Face
print("\n[2/4] Downloading Code Bundle, Weights & Dataset from HuggingFace...", flush=True)
from huggingface_hub import hf_hub_download, snapshot_download

# Download and unpack code bundle
bundle_path = hf_hub_download(repo_id="Kazenowoko/telos", filename="telos_code.tar.gz", token=HF_TOKEN, local_dir=str(work_dir))
subprocess.run(["tar", "-xzf", str(bundle_path), "-C", str(work_dir)], check=True)
print("  Codebase bundle unpacked successfully!", flush=True)

# Download weights and binary dataset
snapshot_download(
    repo_id="Kazenowoko/telos",
    local_dir=str(work_dir),
    token=HF_TOKEN,
    allow_patterns=[
        "configs/*",
        "data/python_corpus_1.7b.bin",
        "checkpoints/*/12m/telos_12m_r*/model.safetensors",
        "checkpoints/*/12m/telos_12m_r*/config.json"
    ]
)
print("  Weights & dataset ready!", flush=True)

# 3. Spawn Background Training Process on TPU
print("\n[3/4] Launching 25M Retraining on TPU (LR = 1e-4)...", flush=True)
log_file = work_dir / "training.log"
with open(log_file, "a") as f_log:
    f_log.write("\n=== 25M TPU RETRAINING SESSION LAUNCHED ===\n")

proc = subprocess.Popen(
    [sys.executable, "scripts/colab/train_25m_upscaled.py", "--ratios", "r1", "r10", "r20", "r25", "--hf-repo", "Kazenowoko/telos", "--device", "tpu"],
    stdout=open(log_file, "a"),
    stderr=subprocess.STDOUT,
    cwd=str(work_dir),
    env=dict(os.environ, HF_TOKEN=HF_TOKEN)
)

print(f"Background TPU Training Process Spawned Successfully! (PID: {proc.pid})", flush=True)
print(f"Logs streaming to: {log_file}", flush=True)
