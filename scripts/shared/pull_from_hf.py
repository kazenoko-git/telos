"""
Automated downloader: Pulls 25M retrained checkpoints from Hugging Face Hub (Kazenowoko/telos)
directly into local repository directories.
"""
import os
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download

def pull_checkpoints(ratios: list[str] = ["r1", "r10"], hf_repo: str = "Kazenowoko/telos"):
    print("=" * 80)
    print(f"PULLING RETRAINED 25M CHECKPOINTS FROM HUGGINGFACE ({hf_repo})...")
    print("=" * 80)
    
    patterns = []
    for r in ratios:
        patterns.append(f"checkpoints/*/25m/telos_25m_{r}/*")
        
    snapshot_download(
        repo_id=hf_repo,
        local_dir="./",
        allow_patterns=patterns
    )
    print("✅ Download Complete! Saved checkpoints:")
    for r in ratios:
        for p in ["ar", "masked", "uniform"]:
            d = Path(f"checkpoints/{p}/25m/telos_25m_{r}")
            if d.exists():
                print(f"  - {d}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratios", nargs="+", default=["r1", "r10"])
    args = parser.parse_args()
    pull_checkpoints(args.ratios)
