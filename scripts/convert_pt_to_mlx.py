"""
télos — PyTorch .pt Checkpoint to MLX .safetensors Converter
============================================================
Converts a trained PyTorch .pt checkpoint into a self-contained 
MLX model bundle (model.safetensors, config.json, tokenizer.json).
"""

import sys
import json
import shutil
import argparse
from pathlib import Path
import torch
import safetensors.torch


def convert(pt_path: str, output_dir: str):
    pt_path = Path(pt_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading PyTorch checkpoint: {pt_path}...")
    ckpt = torch.load(pt_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)

    # Convert state_dict keys and transpositions to MLX format
    mlx_state_dict = {}
    for k, v in state_dict.items():
        key = k
        if key.startswith("model."):
            key = key[6:]
        
        # Convert torch.bfloat16 or torch.float32 tensor to float32 numpy array for safetensors
        tensor = v.detach().to(torch.float32).cpu()

        # MLX Linear weight transposition mapping
        if "w2.weight" in key or "w3.weight" in key or "mlp.w3.weight" in key or "out_proj.weight" in key:
            tensor = tensor.T.contiguous()

        mlx_state_dict[key] = tensor

    # Save safetensors
    out_safetensors = output_dir / "model.safetensors"
    safetensors.torch.save_file(mlx_state_dict, str(out_safetensors))
    print(f"Saved MLX weights -> {out_safetensors}")

    # Save config.json
    model_cfg = ckpt.get("config", {}).get("model", {
        "vocab_size": 8192,
        "d_model": 512,
        "n_layers": 6,
        "n_heads": 8,
        "n_kv_heads": 2,
        "seq_len": 512,
    })
    with open(output_dir / "config.json", "w") as f:
        json.dump(model_cfg, f, indent=2)
    print(f"Saved model config -> {output_dir / 'config.json'}")

    # Copy tokenizer
    tok_src = Path("configs/tokenizer_mac.json")
    if not tok_src.exists():
        tok_src = Path("configs/tokenizer_0.json")
    if tok_src.exists():
        shutil.copy(tok_src, output_dir / "tokenizer.json")
        print(f"Copied tokenizer -> {output_dir / 'tokenizer.json'}")

    print(f"\nConversion complete! Bundle saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PyTorch .pt checkpoint to MLX bundle")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to .pt checkpoint file")
    parser.add_argument("--out", type=str, required=True, help="Output directory path")
    args = parser.parse_args()
    convert(args.ckpt, args.out)
