"""
Hugging Face Hub Upload Script for télos.

Uploads tokenizers, unified configuration files, and 12.5M source checkpoints
to Hugging Face Hub so they can be seamlessly downloaded in Google Colab, cloud VMs,
or inference environments.
"""

import os
import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo


def upload_telos_to_hub(repo_id: str, token: str | None = None, include_checkpoints: bool = True):
    """
    Uploads tokenizers, configs, and checkpoint weights to Hugging Face Hub.
    """
    api = HfApi(token=token or os.environ.get("HF_TOKEN"))
    print(f"Creating / verifying HuggingFace repository: {repo_id}...")
    create_repo(repo_id=repo_id, repo_type="model", token=token, exist_ok=True)
    
    project_root = Path(__file__).resolve().parents[2]
    
    # 1. Upload Tokenizer
    print("\n[1/3] Uploading Tokenizers...")
    for tok_name in ["tokenizer_mac.json", "tokenizer_0.json"]:
        tok_file = project_root / "configs" / tok_name
        if tok_file.exists():
            api.upload_file(
                path_or_fileobj=str(tok_file),
                path_in_repo=f"configs/{tok_name}",
                repo_id=repo_id,
                repo_type="model"
            )
            print(f"  Uploaded configs/{tok_name}")
            
    # 2. Upload Unified Configs
    print("\n[2/3] Uploading Unified Configs...")
    configs_dir = project_root / "configs" / "unified"
    if configs_dir.exists():
        api.upload_folder(
            folder_path=str(configs_dir),
            path_in_repo="configs/unified",
            repo_id=repo_id,
            repo_type="model",
            allow_patterns=["*.yaml", "*.json"]
        )
        print("  Uploaded configs/unified/")
        
    # 3. Upload 12.5M Source Checkpoints
    if include_checkpoints:
        print("\n[3/3] Uploading 12.5M Checkpoints across AR, MDLM, and UNDLM...")
        for paradigm in ["ar", "masked", "uniform"]:
            ckpt_12m_dir = project_root / "checkpoints" / paradigm / "12m"
            if ckpt_12m_dir.exists():
                print(f"  Uploading checkpoints/{paradigm}/12m/...")
                api.upload_folder(
                    folder_path=str(ckpt_12m_dir),
                    path_in_repo=f"checkpoints/{paradigm}/12m",
                    repo_id=repo_id,
                    repo_type="model",
                    allow_patterns=["*.safetensors", "*.json"]
                )
                
    # 4. Upload Model Card / README
    readme_content = f"""---
language:
- en
license: apache-2.0
tags:
- code-autocomplete
- discrete-diffusion
- masked-diffusion
- autoregressive
- pytorch
- mlx
pipeline_tag: text-generation
---

# τέλος (télos) — Unified 3-Paradigm Scaling Research

This repository contains tokenizer models, unified benchmark configurations, and pretrained 12.5M and 25M checkpoints for **télos**:
1. **Autoregressive Baseline (AR)** (Causal next-token prediction)
2. **Masked Discrete Diffusion (MDLM)** (Absorbing $[\\text{{MASK}}]$ diffusion with $1/t$ reweighted ELBO)
3. **Uniform Noise Diffusion (UNDLM)** (Reversible discrete vocabulary corruption)

## Repository Structure
- `configs/`: Tokenizer files (`tokenizer_mac.json`) and unified hyperparameter YAML configs (`configs/unified/`)
- `checkpoints/`: Model weights (`model.safetensors`) organized by paradigm (`ar`, `masked`, `uniform`) and scale (`12m`, `25m`)

## Usage in Google Colab / PyTorch / MLX

```python
from huggingface_hub import snapshot_download

snapshot_download(repo_id="{repo_id}", local_dir="./")
```
"""
    readme_path = project_root / "HF_README.md"
    readme_path.write_text(readme_content)
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model"
    )
    if readme_path.exists():
        readme_path.unlink()
        
    print(f"\nSUCCESS: All files uploaded to https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Upload télos artifacts to HuggingFace Hub")
    parser.add_argument("--repo-id", type=str, default="kazenoko/telos", help="Hugging Face Model Repo ID")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face API Token")
    parser.add_argument("--skip-checkpoints", action="store_true", help="Skip uploading checkpoints")
    args = parser.parse_args()

    upload_telos_to_hub(repo_id=args.repo_id, token=args.token, include_checkpoints=not args.skip_checkpoints)


if __name__ == "__main__":
    main()
