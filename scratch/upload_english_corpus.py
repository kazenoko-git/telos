"""
Upload data/english_corpus_5b.bin (9.3 GB) and metadata to Hugging Face Kazenowoko/telos
"""

import os
import sys
import time
from pathlib import Path
from huggingface_hub import HfApi

TOKEN = ""
REPO_ID = "Kazenowoko/telos"

api = HfApi(token=TOKEN)

bin_file = Path("data/english_corpus_5b.bin")
meta_file = Path("data/english_corpus_5b.bin.json")

if not bin_file.exists():
    raise FileNotFoundError(f"{bin_file} does not exist!")

size_gb = bin_file.stat().st_size / (1024**3)
print(f"Uploading {bin_file} ({size_gb:.2f} GB) to https://huggingface.co/{REPO_ID}/blob/main/{bin_file} ...")
t0 = time.time()

api.upload_file(
    path_or_fileobj=str(bin_file),
    path_in_repo=str(bin_file),
    repo_id=REPO_ID,
    repo_type="model",
    commit_message="add: 5B-token English corpus (FineWeb-Edu)",
)
elapsed = time.time() - t0
print(f"✓ Binary upload successful in {elapsed:.1f}s ({size_gb * 1024 / max(elapsed, 0.1):.1f} MB/s)!")

if meta_file.exists():
    print(f"Uploading {meta_file}...")
    api.upload_file(
        path_or_fileobj=str(meta_file),
        path_in_repo=str(meta_file),
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="add: 5B-token English corpus metadata sidecar",
    )
    print("✓ Metadata upload successful!")

print("All uploads completed successfully!")
