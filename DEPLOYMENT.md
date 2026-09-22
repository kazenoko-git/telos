# Télos Deployment Guide

This guide explains how to install, build, test, and deploy **télos** across local environments, hardware clusters, and package registries.

## 1. System Requirements

- **Python**: 3.10, 3.11, 3.12, 3.13, or 3.14
- **Hardware Targets**:
  - **Apple Silicon (MLX)**: macOS 14.0+ with unified memory (Metal acceleration)
  - **NVIDIA GPU (CUDA)**: Linux with CUDA 12.0+ and PyTorch
  - **TPU (XLA)**: Google Cloud TPU v4 / v5e / v6e with PyTorch-XLA
  - **CPU**: Standard Linux, macOS, or Windows for testing

## 2. Installation

### From PyPI

```bash
# Standard install
pip install telos

# With Apple Silicon MLX acceleration
pip install "telos[mlx]"

# Full developer installation
pip install "telos[all]"
```

### From Source

```bash
git clone https://github.com/kazenoko-git/telos.git
cd telos
uv sync
uv pip install -e .
```

Verify the installation:
```bash
telos --help
```

## 3. Building Distribution Packages

To build source distributions (`.tar.gz`) and binary wheels (`.whl`):

```bash
# Install build tools
pip install build twine

# Build source archive and wheel
python -m build

# Check package validity
twine check dist/*
```

Built artifacts will appear in `dist/`.

## 4. Running Verification Tests

Run the test suite to verify model contracts, causality checks, and numerical stability:

```bash
# Using pytest directly
pytest tests/ -q

# Or using the telos CLI
telos test
```

## 5. Hardware Deployment Modes

### Apple Silicon (MLX)
Use the `--hardware mlx` flag for zero-config Metal acceleration:
```bash
telos train --paradigm corosred --params 50M --tokens 2.5B --hardware mlx
```

### NVIDIA GPUs (CUDA)
Use `--hardware cuda` with optional kernel compilation:
```bash
telos train --paradigm corosred --params 50M --tokens 2.5B --hardware cuda --devices 4 --compile
```

### Google Cloud TPU (PyTorch-XLA)
Use `--hardware xla` with automatic multi-core cadence synchronization:
```bash
telos train --paradigm corosred --params 50M --tokens 2.5B --hardware xla --devices 8
```
