"""
Remote Google Colab Execution Driver for 25M Retraining.

Executed remotely via `colab exec -s Untitled1 -f scripts/colab/remote_colab_runner.py`.
1. Clones repository / downloads Hugging Face artifacts (Kazenowoko/telos).
2. Sets up Python dependencies (PyTorch, safetensors, huggingface_hub, tokenizers).
3. Executes 25M Upscaled 3-Paradigm Retraining (AR, MDLM, UNDLM) with fine-tuning LR (1e-4).
4. Runs Contextual Probes Evaluation.
5. Uploads retrained 25M checkpoints back to Hugging Face Hub.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

HF_TOKEN = os.environ.get("HF_TOKEN")

work_dir = Path("/content/telos")
work_dir.mkdir(parents=True, exist_ok=True)
os.chdir(str(work_dir))
sys.path.insert(0, str(work_dir))

print("=" * 80)
print(f"COLAB GPU ENVIRONMENT INITIALIZED: {work_dir}")
print("=" * 80)

# 1. Install Dependencies
print("\n[Step 1/5] Installing Dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub", "safetensors", "pyyaml", "tokenizers", "numpy", "matplotlib"], check=True)

# 2. Download ONLY necessary 12.5M Checkpoints & Configs from Hugging Face Hub
print("\n[Step 2/5] Downloading 12.5M Checkpoints & Configs from HuggingFace (Kazenowoko/telos)...")
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Kazenowoko/telos",
    local_dir=str(work_dir),
    token=HF_TOKEN,
    allow_patterns=[
        "configs/*",
        "checkpoints/*/12m/telos_12m_r*/model.safetensors",
        "checkpoints/*/12m/telos_12m_r*/config.json"
    ]
)
print("  Download complete!")

# 3. Pull training codebase files
print("\n[Step 3/5] Setting up codebase files...")
if not (work_dir / "mdiff").exists() or not (work_dir / "scripts").exists():
    subprocess.run(["git", "clone", "https://github.com/kazenoko-git/telos.git", "/content/telos_git"], check=False)
    if Path("/content/telos_git").exists():
        for item in ["mdiff", "undiff", "ar", "scripts", "notebooks", "evals"]:
            src = Path("/content/telos_git") / item
            dst = work_dir / item
            if src.exists() and not dst.exists():
                shutil.copytree(src, dst)

# 4. Execute 25M Upscaled 3-Paradigm Retraining Suite
print("\n[Step 4/5] Executing 25M Upscaled 3-Paradigm Retraining Suite (LR = 1e-4)...")
from scripts.colab.train_25m_upscaled import run_full_25m_suite

# Train 1:1, 1:10, 1:20, 1:25
run_full_25m_suite(ratios=["r1", "r10", "r20", "r25"], hf_repo="Kazenowoko/telos", device="cuda")

# 5. Upload New 25M Checkpoints to Hugging Face
print("\n[Step 5/5] Uploading Retrained 25M Checkpoints to HuggingFace...")
from huggingface_hub import HfApi
api = HfApi(token=HF_TOKEN)

for paradigm in ["ar", "masked", "uniform"]:
    ckpt_dir = work_dir / "checkpoints" / paradigm / "25m"
    if ckpt_dir.exists():
        print(f"  Uploading checkpoints/{paradigm}/25m...")
        api.upload_folder(
            folder_path=str(ckpt_dir),
            path_in_repo=f"checkpoints/{paradigm}/25m",
            repo_id="Kazenowoko/telos",
            repo_type="model",
            allow_patterns=["*.safetensors", "*.json"]
        )

print("\n" + "=" * 80)
print("SUCCESS: 25M RETRAINING & HUGGINGFACE EXPORT FULLY COMPLETE!")
print("=" * 80)
