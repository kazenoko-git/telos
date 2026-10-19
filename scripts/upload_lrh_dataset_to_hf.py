#!/usr/bin/env python3
"""Upload pre-tokenized binary datasets (10M / 100M) to Hugging Face Hub.

Usage:
    .venv/bin/python3 scripts/upload_lrh_dataset_to_hf.py --repo-id Kazenowoko/telos --token <HF_TOKEN>
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_token_from_env() -> str | None:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                if line.startswith("HF_TOKEN="):
                    return line.strip().split("=", 1)[1]
    return None


def main():
    parser = argparse.ArgumentParser(description="Upload pre-tokenized binary files to Hugging Face")
    parser.add_argument("--repo-id", type=str, default="Kazenowoko/telos", help="Hugging Face repo ID")
    parser.add_argument("--repo-type", type=str, default="model", choices=["model", "dataset"])
    parser.add_argument("--data-dir", type=str, default="data/qwen25_coder")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face write token")
    args = parser.parse_args()

    token = args.token or get_token_from_env()
    if not token:
        print("ERROR: No Hugging Face token provided. Provide via --token or set HF_TOKEN in .env.")
        sys.exit(1)

    api = HfApi(token=token)
    data_dir = PROJECT_ROOT / args.data_dir
    if not data_dir.exists():
        print(f"ERROR: Data directory {data_dir} does not exist. Run scripts/prepare_lrh_tokenized_dataset.py first.")
        sys.exit(1)

    files_to_upload = [
        data_dir / "qwen25_coder_10m.bin",
        data_dir / "qwen25_coder_100m.bin",
        data_dir / "metadata.json",
    ]

    print("=" * 80)
    print("UPLOADING PRE-TOKENIZED LRH DATASET TO HUGGING FACE")
    print("=" * 80)
    print(f"Target Repo: {args.repo_id} (type: {args.repo_type})")
    print(f"Source Directory: {data_dir}")

    for file_path in files_to_upload:
        if not file_path.exists():
            print(f"Skipping missing file: {file_path.name}")
            continue

        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"\nUploading {file_path.name} ({size_mb:.2f} MB)...")
        api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=f"data/{file_path.name}",
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            commit_message=f"Upload pre-tokenized {file_path.name} ({size_mb:.1f} MB)",
        )
        print(f"Successfully uploaded {file_path.name} -> {args.repo_id}/data/{file_path.name}")

    print("\n" + "=" * 80)
    print("ALL FILES UPLOADED SUCCESSFULLY!")
    print("=" * 80)
    print(f"You can now download these in Kaggle / Colab in 5 seconds via:")
    print(f"  from huggingface_hub import hf_hub_download")
    print(f"  bin_path = hf_hub_download(repo_id='{args.repo_id}', filename='data/qwen25_coder_10m.bin', repo_type='{args.repo_type}')")


if __name__ == "__main__":
    main()
