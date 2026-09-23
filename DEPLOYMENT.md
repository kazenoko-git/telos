# Télos Deployment Guide

This guide describes how to install, build, verify, and publish **télos** (`telos-ml`) across local systems, hardware accelerators, and package registries.

## 1. System Requirements

- **Python**: 3.10, 3.11, 3.12, 3.13, or 3.14
- **Supported Accelerators**:
  - **Apple Silicon (MLX)**: macOS 14.0+ with unified memory (Metal acceleration)
  - **NVIDIA GPU (CUDA)**: Linux with CUDA 12.0+ and PyTorch
  - **TPU (XLA)**: Google Cloud TPU v4 / v5e / v6e with PyTorch-XLA
  - **CPU**: Standard Linux, macOS, or Windows for evaluation and testing

## 2. Package Installation

The official distribution package on PyPI is named `telos-ml`. The Python import name remains `telos`, and the command line binary remains `telos`.

### From PyPI

```bash
# Standard installation
pip install telos-ml

# With Apple Silicon Metal acceleration (MLX)
pip install "telos-ml[mlx]"

# Complete development environment
pip install "telos-ml[all]"
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
telos --version
telos --help
```

## 3. Verification Suite

Run verification checks before building distribution packages:

```bash
# Run unit and regression tests
uv run pytest tests/ -q

# Run unified multi-paradigm verification suite
uv run telos test
```

## 4. Package Building & PyPI Publication

### Build Distribution Artifacts

Télos uses `hatchling` as its build backend. Build distributions with `uv` or `build`:

```bash
# Clean previous builds
rm -rf dist

# Build source distribution (.tar.gz) and wheel (.whl)
uv build
```

Expected output files:
- `dist/telos_ml-1.0.0-py3-none-any.whl`
- `dist/telos_ml-1.0.0.tar.gz`

### Validate Distribution Metadata

```bash
# Check distribution integrity
twine check dist/*

# Inspect wheel archive contents
python3 -m zipfile -l dist/telos_ml-1.0.0-py3-none-any.whl

# Inspect source distribution contents
tar -ztvf dist/telos_ml-1.0.0.tar.gz
```

### Rehearse in an Isolated Environment

Verify the wheel in a temporary environment before public release:

```bash
# Create temporary isolated environment
uv venv /tmp/telos-smoke
source /tmp/telos-smoke/bin/activate

# Install wheel
uv pip install dist/telos_ml-1.0.0-py3-none-any.whl

# Test CLI and import
telos --version
python3 -c "import telos; print(telos.__version__)"

# Deactivate and remove smoke environment
deactivate
rm -rf /tmp/telos-smoke
```

### Publish to Package Registries

#### Publish to TestPyPI (Staging)

```bash
uv publish --publish-url https://test.pypi.org/legacy/
```

Or with twine:
```bash
twine upload --repository testpypi dist/*
```

#### Publish to Official PyPI (Production)

```bash
uv publish
```

Or with twine:
```bash
twine upload dist/*
```

## 5. Hardware Deployment Modes

### Apple Silicon (MLX)

Execute zero-configuration Metal training with unified memory:
```bash
telos train --paradigm corosred --params 50M --tokens 2.5B --hardware mlx
```

### NVIDIA GPUs (CUDA)

Execute multi-GPU training with optional kernel compilation:
```bash
telos train --paradigm corosred --params 50M --tokens 2.5B --hardware cuda --devices 4 --compile
```

### Google Cloud TPU (PyTorch-XLA)

Execute distributed TPU training across all cores with cadence synchronization:
```bash
telos train --paradigm corosred --params 50M --tokens 2.5B --hardware xla --devices 8
```

## 6. Apple Foundation Models Deployment

On Apple Silicon running macOS 15.0+, Télos interfaces directly with on-device Apple Foundation Models (AFM-3 Core / Core Advanced):

```bash
# Check availability status
telos afm status

# Execute prompt generation
telos afm generate "Explain rotary position embeddings in two sentences."
```
