#!/usr/bin/env bash
# ==============================================================================
# Lightning AI TPU v6e-1 Environment Setup Script
# Installs torch-xla, libtpu, huggingface_hub, pyyaml, safetensors, and dependencies.
# ==============================================================================

set -eo pipefail

echo "======================================================================"
echo " Setting up Lightning AI TPU v6e-1 Environment..."
echo "======================================================================"

# Upgrade pip
python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch CPU + PyTorch-XLA for TPU v6e-1
echo "Installing PyTorch & PyTorch-XLA..."
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html

# Install HuggingFace and utilities
echo "Installing HuggingFace Hub, PyYAML, Safetensors, and utilities..."
python3 -m pip install huggingface_hub pyyaml safetensors tokenizers numpy requests tqdm

# Verify TPU Device Detection
echo "Verifying PyTorch-XLA TPU Detection..."
python3 -c "
import torch
import torch_xla.core.xla_model as xm
device = xm.xla_device()
print(f'>> SUCCESS: Detected TPU XLA Device -> {device}')
"

echo "======================================================================"
echo " TPU Environment Setup Complete!"
echo "======================================================================"
