"""
Download updated 100M COROSred (alpha_min=0.75) separately into
checkpoints/corosred/unified/100m_5b_alpha075 without overwriting existing runs.
"""

import os
import sys
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

TOKEN = ""
REPO_ID = "Kazenowoko/telos"

dest_dir = Path("checkpoints/corosred/unified/100m_5b_alpha075")
dest_dir.mkdir(parents=True, exist_ok=True)

remote_prefix = "checkpoints/corosred/unified/100m_5b"
files = [
    "config.json",
    "checkpoint_final.pt",
    "checkpoint_step_24500.pt",
    "checkpoint_step_21000.pt",
    "checkpoint_step_17500.pt",
    "checkpoint_step_14000.pt",
    "checkpoint_step_10500.pt",
    "checkpoint_step_7000.pt",
    "checkpoint_step_3500.pt"
]

print(f"Downloading updated 100M COROSred model to separate directory: {dest_dir} ...")

for fname in files:
    remote_path = f"{remote_prefix}/{fname}"
    target_path = dest_dir / fname
    print(f"Downloading {fname}...")
    cached_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=remote_path,
        token=TOKEN,
        repo_type="model"
    )
    # Copy from cache to target destination
    shutil.copy2(cached_path, target_path)
    print(f"✓ Saved {fname} ({target_path.stat().st_size / (1024**2):.1f} MB)")

print("\n✓ Download completed successfully!")
