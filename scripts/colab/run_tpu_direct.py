"""
Direct In-Kernel TPU Retraining Driver for Colab.
"""
import os
import sys
import subprocess
import importlib
from pathlib import Path

work_dir = Path("/content/telos")
os.chdir(str(work_dir))
if str(work_dir) not in sys.path:
    sys.path.insert(0, str(work_dir))

HF_TOKEN = os.environ.get("HF_TOKEN")

# Download and unpack latest code bundle
from huggingface_hub import hf_hub_download
bundle_path = hf_hub_download(repo_id="Kazenowoko/telos", filename="telos_code.tar.gz", token=HF_TOKEN, local_dir=str(work_dir))
subprocess.run(["tar", "-xzf", str(bundle_path), "-C", str(work_dir)], check=True)
print("Latest code bundle unpacked successfully!", flush=True)

# Unload any previously imported modules so updated code takes effect immediately
for mod in list(sys.modules.keys()):
    if mod.startswith("mdiff") or mod.startswith("undiff") or mod.startswith("ar") or mod.startswith("scripts"):
        del sys.modules[mod]

import torch
import torch_xla.core.xla_model as xm
device = xm.xla_device()
print(f"Acquired TPU Device in Kernel: {device}", flush=True)

import scripts.colab.train_25m_upscaled as trainer
print("=" * 80, flush=True)
print("STARTING 25M RETRAINING ON TPU FOR 1:1 (r1) AND 1:10 (r10)...", flush=True)
print("=" * 80, flush=True)

trainer.run_full_25m_suite(ratios=["r1", "r10"], hf_repo="Kazenowoko/telos", device="tpu")
