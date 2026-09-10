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
| **`telos train`** | `telos.train` | Zero-config dimensional model trainer (AR, MDLM, UNDLM, COROSred Phase A, B & C, custom). No YAML config required. |
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
- **PyTorch Compilation**: `--compile` / `--no-compile` toggles `torch.compile` kernel fusion on CUDA (auto-enabled in unified GPU profiles).
*(All overridable via `--max-lr`, `--min-lr`, `--warmup-steps`, `--weight-decay`, `--compile`)*

### Checkpoint Controls
- `--checkpoint-dir`: Storage directory (default: `checkpoints/<paradigm>`).
- `--save-every`: Save checkpoint cadence in steps (default: auto-calculated as $10\%$ of steps).

### CLI Training Examples

```bash
# 1. Train a 25M MDLM model on 300M tokens on Apple Silicon (MLX)
telos train --paradigm mdlm --params 25M --tokens 300M --effective-batch 32

# 2. Train a 50M UNDLM model on 4x NVIDIA GPUs (CUDA) with torch.compile
telos train --paradigm undlm --params 50M --tokens 500M --hardware cuda --devices 4 --compile

# 3. Train Unified COROSred Continuous Multi-Objective Loop from Scratch (Recommended)
# One continuous loop combining Causal AR, Infilling, and LRH routing from init with zero checkpoint dependency:
telos train --paradigm corosred --params 50M --tokens 1.0B

# 4. Train Unified COROSred with Custom Schedule or Dynamic Loss Rebalancing
telos train --paradigm corosred --params 100M --tokens 2.5B --alpha-min 0.20 --hold-frac 0.20 --adaptive-rebalance

# 5. Legacy Phase-Based COROSred (Phase A -> B -> C) if explicitly desired
telos train --paradigm corosred --phase A --params 50M --tokens 1.0B --legacy-phases
telos train --paradigm corosred --phase C --params 50M --tokens 1.0B --legacy-phases --init-checkpoint checkpoints/corosred/phase_a/checkpoint_final.pt

# 6. Train AR Baseline on Cloud TPU Pod (PyTorch-XLA)
telos train --paradigm ar --params 100M --tokens 2.5B --hardware xla --devices 8

# 7. Config File Bypass (for legacy experiments or reproducibility)
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
# 1. Benchmark Unified Continuous COROSred on Apple Silicon (MLX)
telos bench --paradigm corosred --hardware mlx --duration 15

# 2. Benchmark Autoregressive (AR) causal baseline on Apple Silicon (MLX)
telos bench --paradigm ar --hardware mlx --duration 15

# 3. Benchmark MDLM diffusion baseline on Apple Silicon (MLX)
telos bench --paradigm mdlm --params 25M --hardware mlx --duration 30

# 4. Benchmark on multi-GPU CUDA with torch.compile
telos bench --paradigm undlm --params 50M --hardware cuda --devices 4 --duration 60
```

Results are printed as a publication-quality table and saved to `logs/benchmark_<paradigm>_<backend>_<timestamp>.json`.

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

---

## 9. Kaggle TPU VM (v5e-8 & v3-8) Optimization & Benchmark Guide

When training or benchmarking on Google Cloud or Kaggle Cloud TPUs (v5e-8 or v3-8), Télos includes dedicated optimizations to achieve **600k+ tokens/sec** while eliminating host CPU bottlenecks and memory overflows:

### 1. Eliminating the 650% CPU Bottleneck
TPU VMs often suffer from severe host CPU pegging (600%–800% CPU usage) due to two underlying causes:
1. **OpenMP Thread Over-Subscription**: PyTorch-XLA spawns 8 worker processes via `xmp.spawn`. Without thread capping, each process spawns 8–16 OpenMP threads ($8 \times 8 = 64$ threads) competing on CPU spinlocks (`kmp_wait_yield`). Télos automatically sets `OMP_NUM_THREADS=1` and calls `torch.set_num_threads(1)` inside spawned workers.
2. **Zero-Copy Memory-Mapped Streaming**: Pretokenized `uint16` binary datasets are wrapped directly via `torch.from_numpy()` without intermediate CPU array reallocation.
3. **Asynchronous Static Graphs**: Inner microbatch loss computation uses pure PyTorch tensor operations with zero `.item()` calls, keeping the XLA HLO execution graph 100% static and asynchronous.

### 2. Sizing Effective Batch Size & Preventing OOM on Higher Models
TPU v5e chips provide **16 GB HBM** per tensor core. To strictly maintain a **medium Effective Batch Size = 256 sequences** ($131,072$ tokens/step) across 8 TPU cores without memory exhaustion:
- **15M / 25M / 50M** ($d_{\text{model}} \le 512$): `batch_size=32`, `grad_accum=1` $\implies 32 \times 1 \times 8 = \mathbf{256\text{ sequences}}$ (~3.5 GB HBM per core, reaches **600k tok/s** at ~4.58 steps/s).
- **100M+** ($d_{\text{model}} \ge 768$): `batch_size=16`, `grad_accum=2` $\implies 16 \times 2 \times 8 = \mathbf{256\text{ sequences}}$. Microbatch is halved to 16, cutting peak activation memory in half and preventing OOM while preserving the exact same effective batch size.
- **Explicit 384 Sequences**: Pass `--effective-batch 384` to automatically resolve to `batch_size=48, grad_accum=1` (for 50M) and `batch_size=24, grad_accum=2` (for 100M).

### Recommended All-in-One Kaggle Notebook Cell (`%%bash`)

Running as a bash cell avoids Jupyter kernel `/dev/vfio/*` device lock retention and protects Kaggle's pre-installed PyTorch-XLA binaries from pip conflicts:

```bash
%%bash
# 1. Terminate any hung or zombie TPU device locks from previous runs
fuser -k -9 /dev/vfio/* 2>/dev/null || true

# 2. Configure PyTorch-XLA Environment
export PJRT_DEVICE=TPU
unset LIBTPU_INIT_ARGS
export OMP_NUM_THREADS=1
unset TPU_PROCESS_ADDRESSES
unset CLOUD_TPU_TASK_ID

# 3. Clone or pull latest codebase
cd /kaggle/working
if [ ! -d "telos" ]; then
  git clone https://github.com/kazenoko-git/telos.git
  cd telos
else
  cd telos
  git fetch origin
  git reset --hard origin/main
fi

# 4. Install dependencies without overwriting Kaggle's pre-installed torch/torch_xla
pip install -q pyyaml tokenizers datasets safetensors huggingface_hub
pip install -q --no-deps -e .

# 5. Run Hardware Benchmark across all 8 TPU cores (50M Unified COROSred, 60s)
telos bench --paradigm corosred --params 50M --hardware xla --devices 8 --duration 60
```

### In-Kernel Python Alternative

If calling Télos directly within a Python notebook cell:

```python
import os
os.environ["PJRT_DEVICE"] = "TPU"
os.environ.pop("LIBTPU_INIT_ARGS", None)
os.environ["OMP_NUM_THREADS"] = "1"

import telos

# Run 60s benchmark across all 8 TPU cores
telos.benchmark(
    paradigm="corosred",
    params="50M",
    hardware="xla",
    devices=8,
    duration=60.0
)
```

---

## 10. NVIDIA CUDA (H100, A100, RTX 4090) High-Performance Training Guide

To achieve **3.0M to 4.5M+ tokens/sec on a single H100** (and **30M+ tokens/sec on 8x H100 SXM5**), Télos automatically configures four key optimizations:

### 1. Asynchronous CUDA Graph Execution
- **Zero Inner-Loop Synchronizations**: Elimination of blocking `.item()` calls, dynamic boolean slicing, and heavy sorting inside inner microbatches. The CUDA command queue remains 100% full, preventing SM starvation.
- **FlashAttention-2**: `F.scaled_dot_product_attention` executes fused tile computations directly on Hopper/Ampere Tensor Cores.

### 2. VRAM-Aware Microbatch & Large Effective Batch Sizing (Up to 384+)
The 80GB HBM3 on H100 SXM5 provides 3.35 TB/s of memory bandwidth and vast activation capacity. Because FlashAttention-2 computes attention tiles in SRAM without materializing the $O(T^2)$ attention matrix in HBM, activation memory is linear in $T$ and $d_{\text{model}}$:

| Model Budget | Batch Size | Grad Accum | Effective Batch | Tokens / Step | VRAM Footprint (80GB H100) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **15M** ($d=384, L=10$) | **384** | **1** | **384 sequences** | **196,608 tokens** | $\approx 15.2 \text{ GB}$ ($19\%$ HBM) |
| **50M** ($d=512, L=14$) | **192** | **2** | **384 sequences** | **196,608 tokens** | $\approx 28.5 \text{ GB}$ ($35\%$ HBM) |
| **50M** ($d=512, L=14$) | **384** | **1** | **384 sequences** | **196,608 tokens** | $\approx 29.1 \text{ GB}$ ($36\%$ HBM) |
| **100M** ($d=768, L=14$) | **128** | **3** | **384 sequences** | **196,608 tokens** | $\approx 29.8 \text{ GB}$ ($37\%$ HBM) |
| **100M** ($d=768, L=14$) | **384** | **1** | **384 sequences** | **196,608 tokens** | $\approx 42.4 \text{ GB}$ ($53\%$ HBM) |

- **H100 / A100 (80GB)**: Télos automatically defaults to **Effective Batch 384** ($196,608$ tokens/step). For $d_{\text{model}} \le 384$, it dispatches `batch_size = 384` directly with zero gradient accumulation overhead.
- **Explicit Override**: You can pass `--batch-size 384` or `--effective-batch 384` (or even 512) to saturate memory and keep Hopper Tensor Cores 100% busy.

### 3. Gradient Checkpointing Disabled for Small/Medium Models
- On GPUs with $\ge 24\text{GB}$ VRAM training models $\le 100\text{M}$, activations are retained in VRAM. This saves **33% to 50% compute FLOPs** by completely bypassing backward activation recomputation.

### 4. PyTorch 2.0 `torch.compile` Fusion
- Pass `--compile` to fuse RMSNorm, SwiGLU (`SiLU(x * W1) * V`), and RoPE into single CUDA kernels via Inductor:

```bash
# High-throughput 15M training on single H100 with batch 384 (196k tok/step)
telos train --paradigm corosred --params 15M --tokens 750M --batch-size 384 --hardware cuda --compile

# 50M training on single H100 with batch 384
telos train --paradigm corosred --params 50M --tokens 1.5B --batch-size 384 --hardware cuda --compile

# 100M training on single H100 with effective batch 384 (192 microbatch x 2 accum)
telos train --paradigm ar --params 100M --tokens 4.0B --batch-size 192 --grad-accum 2 --hardware cuda --compile

# Multi-GPU training across 8x H100 via torchrun (30M+ tok/s)
torchrun --nproc_per_node=8 -m telos.train.cli --paradigm corosred --params 100M --tokens 2.5B --effective-batch 768 --hardware cuda --devices 8 --compile
```



