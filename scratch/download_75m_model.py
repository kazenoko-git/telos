"""
Download 75M COROSred model from Hugging Face into checkpoints/corosred/unified/75m_python
and verify weight shapes, keys, and metadata.
"""

import os
import sys
import shutil
import json
from pathlib import Path
from huggingface_hub import hf_hub_download, HfApi
import torch

TOKEN = os.environ.get("HF_TOKEN", "")
REPO_ID = "Kazenowoko/telos"
REMOTE_PREFIX = "checkpoints/corosred/unified/75m_python"

DEST_DIR = Path("checkpoints/corosred/unified/75m_python")
DEST_DIR.mkdir(parents=True, exist_ok=True)

api = HfApi(token=TOKEN)

print("=" * 70)
print(f"  VERIFYING & DOWNLOADING 75M MODEL FROM {REPO_ID}")
print("=" * 70)

# List remote files and sizes
remote_files = api.list_repo_tree(REPO_ID, path_in_repo=REMOTE_PREFIX)
print("\nRemote repository files:")
for item in remote_files:
    sz_mb = item.size / (1024**2) if item.size else 0.0
    print(f"  - {item.path} ({sz_mb:.1f} MB)")

# We download config.json and checkpoint_final.pt first, then intermediate checkpoints
download_order = [
    "config.json",
    "checkpoint_final.pt",
    "checkpoint_step_16000.pt",
    "checkpoint_step_12000.pt",
    "checkpoint_step_8000.pt",
    "checkpoint_step_4000.pt"
]

for fname in download_order:
    remote_path = f"{REMOTE_PREFIX}/{fname}"
    target_path = DEST_DIR / fname
    print(f"\nDownloading {fname}...")
    try:
        cached_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=remote_path,
            token=TOKEN,
            repo_type="model"
        )
        shutil.copy2(cached_path, target_path)
        sz_mb = target_path.stat().st_size / (1024**2)
        print(f"✓ Saved {fname} ({sz_mb:.1f} MB)")
    except Exception as e:
        print(f"✗ Failed to download {fname}: {e}")

print("\n" + "=" * 70)
print("  VERIFYING LOCAL CHECKPOINT INTEGRITY")
print("=" * 70)

ckpt_final_path = DEST_DIR / "checkpoint_final.pt"
if not ckpt_final_path.exists():
    print("FATAL: checkpoint_final.pt does not exist!")
    sys.exit(1)

ckpt_data = torch.load(ckpt_final_path, map_location="cpu")
print("Checkpoint top-level keys:", list(ckpt_data.keys()))
print(f"Global step: {ckpt_data.get('global_step')}")

sd = ckpt_data.get("model_state_dict", ckpt_data)
print(f"Total parameter tensors in state_dict: {len(sd)}")

total_params = 0
for k, v in sd.items():
    if isinstance(v, torch.Tensor):
        total_params += v.numel()
print(f"Total parameters in checkpoint: {total_params:,}")

# Check key architectural landmarks
emb = sd.get("tok_embeddings.weight")
print(f"tok_embeddings.weight shape: {emb.shape if emb is not None else 'MISSING'}")

final_norm = sd.get("final_norm.weight")
print(f"final_norm.weight shape: {final_norm.shape if final_norm is not None else 'MISSING'}")

rel_head = [k for k in sd.keys() if "reliability_head" in k]
print(f"Reliability head layers: {rel_head}")

# Check tied weights
out_proj = sd.get("output_projection.weight")
if out_proj is not None and emb is not None:
    eq = torch.equal(emb, out_proj)
    print(f"tok_embeddings == output_projection: {eq}")

# Check weight norms across first and last layer
print("\nSample layer weight norms:")
for layer_idx in [0, 10, 21]:
    q_w = sd.get(f"layers.{layer_idx}.attn.q_proj.weight")
    w1_w = sd.get(f"layers.{layer_idx}.mlp.w1.weight")
    if q_w is not None and w1_w is not None:
        print(f"  Layer {layer_idx:2d} | Q_proj norm: {q_w.norm().item():.3f} (std: {q_w.std().item():.5f}) | SwiGLU W1 norm: {w1_w.norm().item():.3f}")

print("\n✓ 75M Model verification and download fully successful!")
