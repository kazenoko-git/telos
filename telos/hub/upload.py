"""HuggingFace Hub Export and Upload Script for télos MDLM.

Converts PyTorch model weights to safetensors format, writes model card README,
and uploads model repository to HuggingFace Hub.
"""

import argparse
import json
from pathlib import Path
import torch
from safetensors.torch import save_file
from huggingface_hub import HfApi


MODEL_CARD_TEMPLATE = """---
language:
- python
license: apache-2.0
tags:
- masked-diffusion
- code-autocomplete
- pytorch
- mdlm
pipeline_tag: text-generation
---

# τέλος (télos) — Masked Diffusion Language Model for Code Autocomplete

**télo** is a small Masked Diffusion Language Model (MDLM) built from scratch for narrow-domain Python code completion.

## Model Architecture
- **Paradigm**: Masked Diffusion (absorbing discrete diffusion, non-autoregressive)
- **Attention**: Full bidirectional self-attention (no causal mask)
- **Positional Encoding**: Rotary Positional Embeddings (RoPE)
- **Activation**: SwiGLU (~2.67x expansion)
- **Normalization**: RMSNorm
- **Time Conditioning**: Omitted (time-agnostic ELBO)
- **Weights**: Tied input/output embeddings

## Usage

Since `transformers.generate()` assumes autoregressive decoding, use the standalone `TelosModel` API:

```python
from telos.hub import TelosModel

model = TelosModel.from_pretrained("{repo_id}")

completion = model.complete(
    "def fibonacci(n: int) -> int:\\n    \\\"\\\"\\\"Calculate nth Fibonacci number.\\\"\\\"\\\"\\n",
    max_tokens=128,
    num_steps=64,
    temperature=0.8
)
print(completion)
```
"""


def export_and_upload(model_dir: str, repo_id: str):
    """Exports model weights to safetensors format and uploads to HF Hub."""
    model_dir = Path(model_dir)
    assert model_dir.exists(), f"Model directory {model_dir} does not exist"

    # 1. Load PyTorch checkpoint
    ckpt_path = model_dir / "checkpoint_final.pt"
    if not ckpt_path.exists():
        ckpt_path = list(model_dir.glob("*.pt"))[0]

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # 2. Save safetensors
    safetensors_path = model_dir / "model.safetensors"
    save_file(state_dict, safetensors_path)
    print(f"Exported weights to {safetensors_path}")

    # 3. Create Model Card
    model_card = MODEL_CARD_TEMPLATE.format(repo_id=repo_id)
    with open(model_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(model_card)

    # 4. Upload to HuggingFace Hub
    print(f"Uploading files to HuggingFace Hub repository: {repo_id}...")
    api = HfApi()
    api.create_repo(repo_id=repo_id, exist_ok=True)
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=repo_id,
        allow_patterns=["*.safetensors", "*.json", "*.md"]
    )
    print(f"Upload complete! Model available at https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Export and upload télos model to HuggingFace Hub")
    parser.add_argument("--model-dir", type=str, required=True, help="Directory containing checkpoint")
    parser.add_argument("--repo-id", type=str, required=True, help="HuggingFace repository ID (e.g. username/telos-85m)")
    args = parser.parse_args()

    export_and_upload(args.model_dir, args.repo_id)


if __name__ == "__main__":
    main()
