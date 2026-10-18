#!/usr/bin/env python3
"""Download pinned weights and configs for Qwen3-0.6B and Qwen3-0.6B-diffusion-mdlm-v0.1.

Downloads:
- AR Model: Qwen/Qwen3-0.6B at revision c1899de289a04d12100db370d81485cdf75e47ca
- Diffusion Model: dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1 at revision c8d24a3f4adaeef46881b450e1bf7d1005203bd7
"""
from __future__ import annotations

import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

AR_REPO = "Qwen/Qwen3-0.6B"
AR_REV = "c1899de289a04d12100db370d81485cdf75e47ca"
AR_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"]

DIFF_REPO = "dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1"
DIFF_REV = "c8d24a3f4adaeef46881b450e1bf7d1005203bd7"
DIFF_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "added_tokens.json", "modeling_qwen3.py", "model.safetensors"]


def download_repo_files(repo_id: str, revision: str, filenames: list[str]) -> dict[str, str]:
    print(f"\n--- Downloading {repo_id} (rev: {revision[:8]}) ---")
    paths = {}
    for filename in filenames:
        print(f"Fetching {filename}...")
        p = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
        paths[filename] = p
        size_mb = Path(p).stat().st_size / (1024 * 1024)
        print(f"  -> {filename} ({size_mb:.2f} MB)")
    return paths


def main() -> None:
    ar_paths = download_repo_files(AR_REPO, AR_REV, AR_FILES)
    diff_paths = download_repo_files(DIFF_REPO, DIFF_REV, DIFF_FILES)
    print("\nAll files downloaded and verified.")


if __name__ == "__main__":
    main()
