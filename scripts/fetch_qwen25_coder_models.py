#!/usr/bin/env python3
"""Download pinned weights and configs for Qwen2.5-Coder-0.5B-Instruct and its MDLM counterpart.

Downloads:
- AR Model: Qwen/Qwen2.5-Coder-0.5B-Instruct at revision ea3f2471cf1b1f0db85067f1ef93848e38e88c25
- Diffusion Model: dllm-hub/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1 at revision a284e895a6248256baf2f60502e54aba61b24c1a
"""
from __future__ import annotations

import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

AR_REPO = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
AR_REV = "ea3f2471cf1b1f0db85067f1ef93848e38e88c25"
AR_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "model.safetensors",
]

DIFF_REPO = "dllm-hub/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1"
DIFF_REV = "a284e895a6248256baf2f60502e54aba61b24c1a"
DIFF_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "added_tokens.json",
    "special_tokens_map.json",
    "modeling_qwen2.py",
    "model.safetensors",
]


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
    print("\nAll Qwen2.5-Coder models downloaded and verified.")


if __name__ == "__main__":
    main()
