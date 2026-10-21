# télos (τέλος)

[![PyPI Version](https://img.shields.io/pypi/v/telos-ml.svg)](https://pypi.org/project/telos-ml/)
[![Python Versions](https://img.shields.io/pypi/pyversions/telos-ml.svg)](https://pypi.org/project/telos-ml/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/kazenoko-git/telos/blob/main/LICENSE)
[![Hardware Support](https://img.shields.io/badge/hardware-Apple%20Silicon%20(MLX)%20%7C%20CUDA%20%7C%20TPU%20(XLA)-orange)](https://github.com/kazenoko-git/telos)

**Télos** is a high-performance Python framework for discrete diffusion and autoregressive language modeling. It features zero-config dimensional training across Apple Silicon (MLX), NVIDIA GPUs (CUDA), and Google Cloud TPUs (PyTorch-XLA), alongside native access to Apple on-device Foundation Models.

---

## Installation

### Standard Installation
Install the base package via `pip` or `uv`:

```bash
pip install telos-ml
```

### With Apple Silicon (MLX) Acceleration
For native Apple Silicon Metal acceleration on macOS (M1/M2/M3/M4):

```bash
pip install "telos-ml[mlx]"
```

### With Development & Evaluation Tools
```bash
pip install "telos-ml[dev]"
```

---

## Key Features

- **Discrete Diffusion & Autoregressive Paradigms**: Unified implementations of Masked Diffusion Language Models (MDLM), Uniform Noise Diffusion (UNDLM), Causal Autoregressive (AR), and Confidence-Routed Selective ReDiffusion (COROSred).
- **Multi-Backend Hardware Acceleration**:
  - **Apple Silicon**: Native Metal execution via Apple MLX (`mlx.core`).
  - **NVIDIA CUDA**: Scaled PyTorch training with FlashAttention-2, mixed-precision (bfloat16), and master weight accumulation.
  - **Google Cloud TPU**: Multi-core single-chip and Pod scaling with PJRT runtime.
- **Apple Foundation Models (AFM) Bridge**: Direct Python access to Apple's native on-device language models on macOS Sequoia via compiled Swift IPC.
- **Evaluation & Benchmarks**: 100 contextual probes, code execution suites (HumanEval, MBPP), cybersecurity audits, and tool-use verification.
- **High-Throughput Data Preparation**: Memory-mapped binary token streaming (`.bin`) with custom Byte-Level BPE tokenization.

---

## Quickstart

### 1. Command Line Interface (CLI)

Télos ships with a unified command-line dispatcher:

```bash
# Prepare dataset into high-speed binary token format
telos dataprep --source /path/to/code_corpus --output data/corpus.bin

# Train a 12M parameter model on automatically detected hardware
telos train --paradigm mdlm --params 12M --tokens 1B

# Run hardware throughput benchmark (capped at 5 minutes)
telos bench --hardware auto

# Evaluate a trained checkpoint across evaluation suites
telos eval --checkpoint checkpoints/mdlm/checkpoint_final.pt

# Query on-device Apple Foundation Model status (macOS only)
telos afm status
```

---

### 2. Python API

#### Defining and Initializing Models
```python
from telos.models import TelosConfig, TelosTransformer

# Configure a compact transformer architecture
config = TelosConfig(
    vocab_size=8192,
    d_model=512,
    n_layers=12,
    n_heads=8,
    n_kv_heads=8,
    max_seq_len=512,
    tied_embeddings=True,
    is_causal=False,  # Set to False for bidirectional diffusion, True for AR
)

# PyTorch Model
model = TelosTransformer(config)
```

#### Apple Silicon Metal (MLX)
```python
from telos.models import MLXTelosTransformer

# Direct Apple Silicon Metal transformer
mlx_model = MLXTelosTransformer(config)
```

#### Apple Foundation Models (macOS Sequoia)
Access Apple's system language model directly from Python:

```python
from telos import afm

# Verify system availability
status = afm.status()
if status["available"]:
    # Synchronous generation
    response = afm.generate("Explain discrete diffusion models in two sentences.")
    print(response)

    # Streaming generation
    for chunk in afm.stream("Write a Python binary search function."):
        print(chunk, end="", flush=True)
```

#### Loading Model Checkpoints Securely
Télos enforces `weights_only=True` to prevent arbitrary code execution during deserialization:

```python
from telos.eval.runner import load_model_from_checkpoint

# Safely loads weights into optimal runtime backend (MLX or PyTorch)
model, backend, vocab_size = load_model_from_checkpoint("checkpoints/model.pt")
```

---

## Hardware Support

| Backend | Hardware | Framework | Key Features |
| :--- | :--- | :--- | :--- |
| **`mlx`** | Apple Silicon (M1/M2/M3/M4) | Apple MLX | Zero-copy unified memory, Metal acceleration |
| **`cuda`** | NVIDIA GPUs (Ampere, Ada, Hopper) | PyTorch | `torch.compile`, FlashAttention-2, bfloat16 |
| **`xla`** | Google Cloud TPU (v2/v3/v4/v5) | PyTorch-XLA | PJRT runtime, distributed data-parallel |
| **`afm`** | Apple Silicon (macOS 15.0+) | Swift Bridge | SystemLanguageModel on-device inference |

---

## Documentation & Resources

- **GitHub Repository**: [github.com/kazenoko-git/telos](https://github.com/kazenoko-git/telos)
- **Deployment & Cloud Training Guide**: [DEPLOYMENT.md](https://github.com/kazenoko-git/telos/blob/main/DEPLOYMENT.md)
- **Changelog**: [CHANGELOG.md](https://github.com/kazenoko-git/telos/blob/main/CHANGELOG.md)
- **Issue Tracker**: [github.com/kazenoko-git/telos/issues](https://github.com/kazenoko-git/telos/issues)

---

## License

Télos is licensed under the [Apache 2.0 License](https://github.com/kazenoko-git/telos/blob/main/LICENSE).
