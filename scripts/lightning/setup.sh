#!/bin/bash
set -e

echo "==============================================================================="
echo "  télos (τέλος) - Lightning AI TPU v6e-1 Environment Bootstrap"
echo "==============================================================================="

# Export miniconda PATH if not already in PATH
export PATH="/system/conda/miniconda3/envs/cloudspace/bin:/system/conda/miniconda3/bin:$PATH"

# 1. Install PyTorch & PyTorch-XLA for TPU v6e (Trillium architecture)
echo "[1/4] Installing PyTorch and PyTorch/XLA (TPU wheels)..."
pip install "torch==2.8.0" "torch_xla[tpu]==2.8.0" -f https://storage.googleapis.com/libtpu-releases/index.html

# 2. Install essential dependencies for télos
echo "[2/4] Installing supporting dependencies (safetensors, huggingface_hub, pyyaml, numpy)..."
pip install safetensors huggingface_hub pyyaml numpy

# 3. Create persistent compilation cache directory
echo "[3/4] Configuring persistent XLA compilation cache directory..."
mkdir -p /tmp/xla_cache
export XLA_PERSISTENT_CACHE_PATH=/tmp/xla_cache

# 4. Verify TPU v6e Hardware Initialization
echo "[4/4] Verifying TPU hardware detection..."
python -c "import torch_xla.core.xla_model as xm; dev = xm.xla_device(); print(f'[TPU Verified] Hardware online: {dev}')"

echo "==============================================================================="
echo "  Setup Complete! You can now run:"
echo "  export HF_TOKEN='your_huggingface_token'"
echo "  python scripts/lightning/train_25m_lightning.py --ratios r15 r20 r25 r30 r35"
echo "==============================================================================="
