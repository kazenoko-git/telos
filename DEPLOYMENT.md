# DEPLOYMENT — télos (τέλος) PyPI Package & CLI Guide

This document details how to install, prepare data, train, evaluate, benchmark, and deploy **τέλος** — a Discrete Diffusion & Autoregressive Language Modeling package for Python code autocomplete and non-monotonic generation.

---

## 1. Installation

Install Télos directly via `pip` or `uv`:

```bash
# Install from source (or PyPI wheel)
pip install telos

# Or with uv
uv pip install telos
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
telos --help
```

---

## 2. Master CLI Overview (`telos <command>`)

Télos provides a unified command line interface with 5 core commands:

| Command | Module | Description |
| :--- | :--- | :--- |
| **`telos dataprep`** | `telos.dataprep` | High-efficiency data processing for raw text, code directories, JSONL, or Hugging Face datasets into chunked binary memory-mapped arrays (`.bin`). |
| **`telos train`** | `telos.train` | Zero-config dimensional model trainer (AR, MDLM, UNDLM, COROSred Phase A & B, custom). No YAML config required. |
| **`telos eval`** | `telos.eval` | High-end evaluation suite with 100 contextual probes across 8 categories, target CE, average rank, and qualitative code sampling. |
| **`telos bench`** | `telos.bench` | Dedicated throughput, latency, and memory benchmark engine strictly capped at at most 5 minutes. |
| **`telos test`** | `telos.testing` | Unified test suite verifying model contracts, causality, losses, and samplers. |

---

## 3. Zero-Config Dimensional Training (`telos train`)

**No YAML configs required.** The user directly specifies the fundamental training dimensions from the command line:

### The 6 Fundamental Training Dimensions

1. **Amount of Parameters**: `--params` (e.g. `12M`, `25M`, `50M`, `100M`, `500M`, or raw integer). An analytical geometry solver automatically computes optimal $(d_{\text{model}}, n_{\text{layers}}, n_{\text{heads}})$.
2. **Amount of Training Tokens**: `--tokens` (e.g. `2.5B`, `300M`, `50M`). Total steps are automatically calculated from effective batch tokens per step (or pass `--max-steps`).
3. **Batch Size**: `--effective-batch` (sequences or token count) with automatic gradient accumulation calculation, OR direct `--batch-size` + `--grad-accum`.
4. **Tokenizer**: `--tokenizer` (path to custom JSON, Hugging Face model, or default) with automatic `--vocab-size` inference.
5. **Hardware Target**: `--hardware` (`auto`, `mlx`, `cuda`, `mps`, `xla`, `cpu`). Auto-detects Apple Silicon Metal, NVIDIA GPUs, or Cloud TPUs.
6. **Hardware Count**: `--devices` (e.g. `1`, `4`, `8`, or `auto` for all available devices).

### Automatic Training Dynamics
- **Max LR**: Auto-scaled with model width: $\text{max\_lr} = 6.0 \times 10^{-4} \times \sqrt{256 / d_{\text{model}}}$.
- **Min LR**: Auto-calculated as $0.1 \times \text{max\_lr}$ (standard cosine floor).
- **Warmup Steps**: Auto-calculated as $\max(50, \min(2000, 0.02 \times \text{max\_steps}))$.
- **Weight Decay**: Default $0.1$.
*(All overridable via `--max-lr`, `--min-lr`, `--warmup-steps`, `--weight-decay`)*

### Checkpoint Controls
- `--checkpoint-dir`: Storage directory (default: `checkpoints/<paradigm>`).
- `--save-every`: Save checkpoint cadence in steps (default: auto-calculated as $10\%$ of steps).

### CLI Training Examples

```bash
# 1. Train a 25M MDLM model on 300M tokens on Apple Silicon (MLX)
telos train --paradigm mdlm --params 25M --tokens 300M --effective-batch 32

# 2. Train a 50M UNDLM model on 4x NVIDIA GPUs (CUDA)
telos train --paradigm undlm --params 50M --tokens 500M --hardware cuda --devices 4

# 3. Train COROSred 2-Phase Model
telos train --paradigm corosred --phase A --params 12M --tokens 50M
telos train --paradigm corosred --phase B --params 12M --tokens 100M

# 4. Train AR Baseline on Cloud TPU Pod (PyTorch-XLA)
telos train --paradigm ar --params 100M --tokens 2.5B --hardware xla --devices 8

# 5. Config File Bypass (for legacy experiments or reproducibility)
telos train --config configs/unified/25m/telos_25m_r10.yaml
```

> [!TIP]
> **TPU Runtime Notes & Kaggle Troubleshooting**:
> 1. Ensure `export PJRT_DEVICE=TPU` is set in your environment (standard on Kaggle/Colab TPU runtimes).
> 2. **Resolving `/dev/vfio/*: Device or resource busy`**: On Cloud/Kaggle TPU VMs, each TPU chip (`/dev/vfio/0`, `/dev/vfio/1`, etc.) can only be locked by a single process at a time. If an earlier process crashed or a notebook cell was interrupted, the device lock remains held. Run:
>    ```bash
>    fuser -k -9 /dev/vfio/* 2>/dev/null || true
>    ```
>    or in the Kaggle UI: click **Session -> Restart Session**.
> 3. **Avoid Notebook Kernel Contention**: Do not import `torch_xla` or initialize TPU tensors in interactive notebook cells if you execute training via shell `!telos train ...`. Otherwise the long-lived Jupyter kernel retains `/dev/vfio/*`, blocking child processes.
> 4. **Multi-Phase Pipeline**: When chaining Phase A and Phase B via shell, insert a 5-second pause (`sleep 5`) to allow the kernel driver to release VFIO descriptors, or run both phases within a single Python script using `from telos.train.cli import train`.

---

## 4. High-Efficiency Data Preparation (`telos dataprep`)

Converts any source corpus into contiguous, memory-mapped binary token arrays (`.bin`) with constant low RAM usage:

```bash
# 1. Process a directory of source code files recursively
telos dataprep --corpus src/ --output data/python_corpus.bin

# 2. Process a JSONL file
telos dataprep --corpus data/train.jsonl --text-key content --output data/corpus.bin

# 3. Stream from a Hugging Face dataset
telos dataprep --dataset codeparrot/codeparrot-clean --output data/python_corpus.bin

# 4. Train a new ByteLevel BPE tokenizer on the corpus
telos dataprep --corpus src/ --train-tokenizer --vocab-size 8192 --output data/corpus.bin

# 5. Generate a synthetic stream for testing
telos dataprep --synthetic --tokens 100000 --output data/synthetic_corpus.bin
```

---

## 5. Model Evaluation Suite (`telos eval`)

Runs the comprehensive 101 contextual probes benchmark or qualitative generation sampling on any MLX (`.safetensors`) or PyTorch (`.pt`) checkpoint:

```bash
# 1. Run 100 contextual probes benchmark (across 8 syntactic categories)
telos eval --checkpoint checkpoints/mdlm/model.safetensors --mode probes

# 2. Run qualitative code completion sampling
telos eval --checkpoint checkpoints/12m/telos_12m_r1/model.safetensors --mode sample
```

The probes suite outputs category breakdowns for Top-1 (%), Top-5 (%), Average Rank, and Target Cross-Entropy, saving a detailed JSON report to `logs/`.

---

## 6. Hardware Throughput Benchmarks (`telos bench`)

Measures steps/sec, tokens/sec, step latency percentiles (mean, p50, p95), and unified memory usage.
**Guaranteed to run for at most 5 minutes (300 seconds):**

```bash
# 1. Benchmark MDLM on Apple Silicon
telos bench --paradigm mdlm --params 25M --duration 30

# 2. Benchmark on multi-GPU CUDA
telos bench --paradigm undlm --params 50M --hardware cuda --devices 4 --duration 60
```

Results are printed as a publication-quality table and saved to `logs/benchmark_<paradigm>_<backend>.json`.

---

## 7. Programmatic Python API

All functionality is also accessible programmatically:

```python
import telos

# 1. Data Preparation
telos.dataprep(
    corpus="src/",
    output_path="data/corpus.bin",
    vocab_size=8192
)

# 2. Zero-Config Model Training
trainer = telos.train(
    paradigm="mdlm",
    params="25M",
    tokens="300M",
    effective_batch=32,
    hardware="auto"
)

# 3. Model Evaluation
results = telos.evaluate(
    checkpoint="checkpoints/mdlm/model.safetensors",
    mode="probes"
)
print("Top-1 Accuracy:", results["overall"]["top1_acc_pct"])

# 4. Benchmarking
bench_results = telos.benchmark(
    paradigm="ar",
    params="12M",
    duration=15.0
)
```

---

## 8. Packaging & Publishing to PyPI

### Build Wheels and Source Distribution
Build the production package artifacts using `uv build` (or `python -m build`):

```bash
# Build tar.gz and .whl into dist/
uv build
```

The wheel automatically bundles the default ByteLevel BPE tokenizer (`telos/assets/tokenizer_0.json`) so installed packages function with zero manual downloads.

### Optional Hardware Acceleration Targets
- Standard Linux/CUDA/TPU install:
  ```bash
  pip install telos
  ```
- Apple Silicon Metal acceleration (`mlx`):
  ```bash
  pip install "telos[mlx]"
  ```
- Complete development environment with tests:
  ```bash
  pip install "telos[all]"
  ```

### Publish to PyPI
```bash
# Upload to PyPI via twine (or uv publish)
uv publish
# Or:
twine upload dist/*
```
