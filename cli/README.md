# Télos CLI & Deployment Guide (`telos`)

This document is the official command line interface (CLI) and package deployment guide for **τέλος** (télos). It details installation, data preparation, zero-config model training, multi-domain evaluation, hardware benchmarking, Apple Foundation Models (`telos afm`), and PyPI distribution.

---

## 1. Installation

Install Télos directly via `pip` or `uv`:

```bash
# Standard installation
pip install telos-ml

# With Apple Silicon Metal support (MLX)
pip install "telos-ml[mlx]"

# Complete development environment
pip install "telos-ml[all]"
```

For local development in editable mode:
```bash
git clone https://github.com/kazenoko-git/telos.git
cd telos
uv sync
uv pip install -e .
```

Verify installation:
```bash
telos --version
telos --help
```

---

## 2. Master CLI Commands Overview (`telos <command>`)

Télos provides a unified command line interface with 6 primary subcommands:

| Command | Module | Description |
| :--- | :--- | :--- |
| **`telos dataprep`** | `telos.dataprep` | High-efficiency data processing for raw text, code directories, JSONL, or Hugging Face datasets into chunked binary memory-mapped arrays (`.bin`). |
| **`telos train`** | `telos.train` | Zero-config dimensional model trainer (AR, MDLM, UNDLM, COROSred, custom). No YAML configuration required. |
| **`telos eval`** | `telos.eval` | Multi-domain evaluation suite across 100 contextual probes, 512 functional code execution tasks, anti-cheat span masking, linguistics, and tool-use. |
| **`telos bench`** | `telos.bench` | Hardware throughput, step latency, and memory benchmark engine strictly capped at 5 minutes. |
| **`telos afm`** | `telos.afm` | Apple on-device Foundation Models: availability status and text generation on Apple Silicon. |
| **`telos test`** | `telos.testing` | Unified verification suite verifying model contracts, causality, losses, and samplers. |

---

## 3. Zero-Config Dimensional Training (`telos train`)

User directly specifies the fundamental training dimensions from the command line:

### The 6 Fundamental Training Dimensions

1. **Amount of Parameters**: `--params` (e.g. `12M`, `25M`, `50M`, `100M`, `500M`, or raw integer). An analytical geometry solver automatically computes optimal $(d_{\text{model}}, n_{\text{layers}}, n_{\text{heads}})$. Custom architectures can be explicitly specified via `--d-model` and `--n-layers`.
2. **Amount of Training Tokens**: `--tokens` (e.g. `2.5B`, `300M`, `50M`). Total steps are automatically calculated from effective batch tokens per step.
3. **Batch Size**: `--effective-batch` (sequences or token count) with automatic gradient accumulation calculation, OR direct `--batch-size` + `--grad-accum`.
4. **Tokenizer**: `--tokenizer` (path to custom JSON, Hugging Face model, or default) with automatic `--vocab-size` inference.
5. **Hardware Target**: `--hardware` (`auto`, `mlx`, `cuda`, `mps`, `xla`, `cpu`). Auto-detects Apple Silicon Metal, NVIDIA GPUs, or Cloud TPUs.
6. **Hardware Count**: `--devices` (e.g. `1`, `4`, `8`, or `auto` for all available devices).

### Automatic Training Dynamics
- **Max LR**: Auto-scaled with model width: $\text{max\_lr} = 6.0 \times 10^{-4} \times \sqrt{256 / d_{\text{model}}}$.
- **Min LR**: Auto-calculated as $0.1 \times \text{max\_lr}$ (standard cosine floor).
- **Warmup Steps**: Auto-calculated as $\max(50, \min(2000, 0.02 \times \text{max\_steps}))$.
- **Weight Decay**: Default $0.1$.
- **PyTorch Compilation**: `--compile` / `--no-compile` toggles `torch.compile` kernel fusion on CUDA.

### Cadence & Device Synchronization Controls
- `--default-cadence`: Base step cadence across all cadence operations (defaults to **25** on TPU/XLA, and **1** on CUDA/CPU).
- `--lr-cadence`: Step frequency for learning rate updates.
- `--sched-step` / `--sched-cadence`: Step frequency for COROSred dynamic schedule updates.
- `--cadence`: Step frequency for cross-replica metric synchronization and EMAs.

### CLI Training Examples

```bash
# 1. Train a 25M MDLM model on 300M tokens on Apple Silicon (MLX)
telos train --paradigm mdlm --params 25M --tokens 300M --effective-batch 32

# 2. Train a 50M UNDLM model on 4x NVIDIA GPUs (CUDA) with torch.compile
telos train --paradigm undlm --params 50M --tokens 500M --hardware cuda --devices 4 --compile

# 3. Train Unified COROSred 50M on Cloud TPU (8 cores)
telos train --paradigm corosred --params 50M --tokens 2.5B --causal-ratio 0.65 --mask-prob 0.20 --hardware xla --devices 8 --data data/python_corpus_5b.bin

# 4. Train 100M AR Baseline on 5B Tokens on Cloud TPU
telos train --paradigm ar --params 100M --tokens 5B --hardware xla --devices 8 --data data/python_corpus_5b.bin
```

---

## 4. Multi-Domain Evaluation (`telos eval`)

```bash
# 1. Run Complete Multi-Domain Evaluation (Code + English Linguistics + Tool-Use)
telos eval --checkpoint checkpoints/corosred/model.safetensors --type all --mode full

# 2. Python Code Contextual Probes Suite (1,000 contextual probes with 95% Bootstrap CI)
telos eval --checkpoint checkpoints/corosred/model.safetensors --type code --language python --mode probes --num-probes 1000

# 3. Functional Execution Benchmark (512 sandboxed tasks with Pass@1, AST validity)
telos eval --checkpoint checkpoints/corosred/model.safetensors --type code --mode functional --suite private_unseen

# 4. Anti-Cheat & Suffix-Copy Span Masking Suite (K in {1, 2, 4, 8, 16})
telos eval --checkpoint checkpoints/corosred/model.safetensors --type code --mode anticheat

# 5. Native Apple Foundation Models (AFM-3 Core Advanced) Evaluation
telos eval --model afm-3-core-advanced --backend swift --type all

# 6. IBM Granite 4.2 3B Evaluation
telos eval --model ibm-granite-4.2-3b --backend mlx --type all
```

---

## 5. Hardware Benchmarking (`telos bench`)

Strictly capped at 5 minutes to prevent compute waste:

```bash
# Benchmark 50M COROSred model throughput on Apple Silicon
telos bench --paradigm corosred --params 50M --hardware mlx

# Benchmark 100M AR baseline on CUDA
telos bench --paradigm ar --params 100M --hardware cuda
```

---

## 6. Apple Foundation Models (`telos afm`)

Access Apple's on-device Foundation Models (AFM 3) on Apple Silicon:

```bash
# Check on-device model availability
telos afm status

# Generate completion via native Swift bridge
telos afm generate "Explain rotary position embeddings in two sentences."
```

Python API:
```python
import telos.afm

# Probe availability
status = telos.afm.probe()
print("AFM Available:", status.available)

# Generate completion
text = telos.afm.generate("Write a haiku about gradients.")
print(text)
```

---

## 7. Python API Overview

```python
import telos

# 1. Prepare Dataset
telos.prepare_dataset(
    input_path="data/raw_python/",
    output_bin="data/python_corpus.bin",
    vocab_size=8192
)

# 2. Train Model
telos.run_train(
    paradigm="corosred",
    params="50M",
    tokens="2.5B",
    hardware="mlx"
)

# 3. Evaluate Checkpoint
results = telos.evaluate(
    checkpoint="checkpoints/corosred/model.safetensors",
    benchmark_type="code",
    mode="full"
)
print("Pass@1:", results["functional"]["pass_at_1_pct"])
```

---

## 8. Packaging & PyPI Distribution

```bash
# Build wheels and source distribution
rm -rf dist
uv build

# Inspect archive contents
python -m zipfile -l dist/telos_ml-*.whl
tar tzf dist/telos_ml-*.tar.gz

# Rehearse in isolated smoke venv
uv venv /tmp/telos-smoke
uv pip install --python /tmp/telos-smoke/bin/python \
  --extra-index-url https://pypi.org/simple/ dist/telos_ml-1.0.0-py3-none-any.whl
/tmp/telos-smoke/bin/telos --version
/tmp/telos-smoke/bin/telos afm status

# Publish to TestPyPI
uv publish --publish-url https://test.pypi.org/legacy/

# Publish to PyPI
uv publish
```
