# DEPLOYMENT — télos (τέλος) MDLM & UNDLM

This document details how to set up, reproduce, train, evaluate, and deploy **τέλος** — a Discrete Diffusion Language Model for Python code autocomplete and non-monotonic generation (supporting both Masked Discrete Diffusion and Uniform Noise Diffusion paradigms).

---

## 1. Model Scaling & Hardware Specifications

| Suite | Model Size | Architecture | Vocab / Context | Backend | Hardware Target |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **12M Suite** | **~12.5 Million** | 13L, $d=256$, 4 Heads | 8,192 BPE / 512 | MLX / PyTorch-XLA / CUDA | Apple Silicon / TPU / CUDA GPU |
| **25M Suite** | **~26.1 Million** | 13L, $d=384$, 6 Heads | 8,192 BPE / 512 | MLX / PyTorch-XLA / CUDA | Apple Silicon / TPU / CUDA GPU |
| **50M Suite** | **~50 Million** | 13L, $d=512$, 8 Heads | 8,192 BPE / 512 | MLX / PyTorch-XLA / CUDA | Apple Silicon / TPU / CUDA GPU |

---

## 2. Environment Setup

### Prerequisites
- Python >= 3.10
- `uv` package manager (`curl -LsSf https://ast.sh/uv/install.sh | sh`)

### Installation & Dependencies
```bash
git clone https://github.com/kazenoko-git/telos.git
cd telos
uv sync
```

For Universal Training via Notebooks:
Open [`notebooks/shared/Unified_Training_Suite.ipynb`](notebooks/shared/Unified_Training_Suite.ipynb) which automatically detects MLX (Apple Silicon), PyTorch XLA (Google Cloud/Kaggle TPU v5e/v6e), or PyTorch CUDA (NVIDIA GPUs), syncing dataset/tokenizer and weights from Hugging Face automatically.

### For Apple Silicon (MLX Metal GPU Acceleration & Unified Memory)
```bash
uv pip install mlx tokenizers torch pyyaml
```

> **Memory Footprint & Unified Memory Optimization (< 6GB Target):**
> - **12.5M Suite**: Runs natively in `~4.9–5.3 GB` peak memory without gradient checkpointing (`~87k tok/s`).
> - **25M Suite**: Set `gradient_checkpointing: true` (or `use_grad_checkpoint: true` under `model:`) in config YAMLs. This reduces transient activation memory from `~7.0–7.4 GB` down to `~1.75–2.34 GB` peak (`~49k tok/s`), guaranteeing safe execution well within 6GB Unified Memory limits.
> - **Inference**: Samplers execute confidence margin and top-$k$ unmasking directly in Metal kernels, avoiding high-bandwidth host-to-device transfers.

---

## 3. Data Preparation & Tokenization

Prepare the tokenized dataset matrix before executing training runs:

```bash
# 1. Prepare raw python corpus
uv run python scripts/shared/prepare_data.py --config configs/masked/25m/phase_b_25m_1to40_mlx.yaml

# 2. Generate publication-quality probe scaling curves
uv run python scripts/shared/generate_probe_graphs.py
```

---

---

## 4. Unified Training Pipelines & Hardware Modularity

The single entry point for all training, evaluation, and benchmarking is `scripts/train.py`. The engine automatically detects available hardware and scales execution.

### Hardware Modularity & Auto-Scaling

| Hardware Target | Examples | Auto-Detected Behaviors |
| :--- | :--- | :--- |
| **Apple Silicon (MLX)** | M1–M4/M5 (16GB, 24GB, 36GB, 64GB, 128GB+) | Dynamically adjusts `mx.eval()` frequency based on unified memory. Low RAM (<24GB) uses eager microbatch evaluation; High RAM (>=36GB) disables intermediate graph syncs for maximum Metal throughput. |
| **NVIDIA CUDA Multi-GPU** | 1×–8× GPUs (2× T4, 4× RTX PRO 6000, 8× A100/H100) | Auto-detects device count, wraps model in multi-GPU parallelization, auto-selects `bf16` (Ampere/Ada/Hopper) or `fp16` + `GradScaler` (Turing/T4). |
| **Google Cloud TPU (PyTorch-XLA)** | v3e-8, v5e-8, v6e-1, v6e-16 | Auto-detects `xm.xrt_world_size()`, shards dataset streaming across TPU cores, coordinates cross-core all-reduce via `xm.optimizer_step()`. |

---

### Running Training via Unified CLI

```bash
# 1. Apple Silicon (MLX) - Local Training
uv run python scripts/train.py --paradigm mdlm --backend mlx --config configs/unified/25m/telos_25m_r10.yaml

# 2. NVIDIA Multi-GPU (CUDA) - e.g., 4x RTX PRO 6000 or 2x T4
python scripts/train.py --paradigm undlm --backend pytorch --device cuda --config configs/unified/50m/telos_50m_r35.yaml

# 3. Cloud TPU Pod (PyTorch-XLA) - e.g., v5e-8 or v6e-16
python scripts/train.py --paradigm ar --backend pytorch --device xla --config configs/unified/100m/telos_100m_r1.yaml

# 4. COROSred 2-Phase Training (Phase A: Reliability Head; Phase B: MDLM)
python scripts/train.py --paradigm corosred --phase A --backend mlx --config configs/corosred/phase_a.yaml
python scripts/train.py --paradigm corosred --phase B --backend mlx --config configs/corosred/phase_b.yaml
```

---

### Hardware Throughput Benchmark Mode (--benchmark)

Run an immediate throughput benchmark that automatically measures steps/second, tokens/second, step latency, and memory footprint:

```bash
# Run a 5-minute benchmark on local Apple Silicon
uv run python scripts/train.py --paradigm mdlm --backend mlx --config configs/unified/25m/telos_25m_r10.yaml --benchmark

# Run a benchmark on multi-GPU CUDA or TPU
python scripts/train.py --paradigm undlm --backend pytorch --device cuda --config configs/unified/50m/telos_50m_r35.yaml --benchmark
```

> **Benchmark Guarantees:**
> - Runs for **at most 5 minutes** (300s) before terminating cleanly.
> - Bypasses checkpoint disk I/O overhead.
> - Automatically outputs a structured performance summary and saves metrics to `logs/benchmark_<paradigm>_<backend>.json`.



---

## 5. Evaluation & Contextual Probes Benchmark

Run the 101 contextual probes benchmark (measuring rank, target cross-entropy, top-1/5 accuracies across 8 syntactic categories):

```bash
# Run probes on an MLX checkpoint
uv run python evals/masked/Evaluation.py --checkpoint checkpoints/masked/25m/kappa_25m_1to40_mlx --mode probes

# Run qualitative code generation samples
uv run python evals/masked/Evaluation.py --checkpoint checkpoints/masked/25m/kappa_25m_1to40_mlx --mode sample
```

Outputs will be automatically saved to `evals/masked/probes/`.

---

## 6. Inference Deployment & Web Interface

All three paradigms provide a high-level standalone **`TelosModel.from_pretrained()`** API within their respective Model Hub modules.

### A. Masked Diffusion Language Models (MDLM)
Generates code via confidence-based iterative unmasking:

```python
from mdiff.hub import TelosModel

# Load pretrained MDLM checkpoint
model = TelosModel.from_pretrained("checkpoints/masked/12m/telos_12m_r15")

# Generate code completion
completion = model.complete(
    prompt="def sort_array(arr: list[int]) -> list[int]:\n",
    max_tokens=64,
    num_steps=64,
    temperature=0.3,
    schedule="linear"  # Options: "linear" or "cosine"
)
print(completion)
```

### B. Uniform Noise Diffusion Models (UNDLM)
Generates code via iterative reversible denoising with self-correction:

```python
from undiff.hub import TelosModel

# Load pretrained UNDLM checkpoint
model = TelosModel.from_pretrained("checkpoints/uniform/12m/telos_12m_r15")

# Generate code completion
completion = model.complete(
    prompt="def fibonacci(n: int) -> int:\n",
    max_tokens=64,
    num_steps=64,
    temperature=0.8,
    schedule="linear"  # Options: "linear" or "cosine"
)
print(completion)
```

### C. Autoregressive Baseline Models (AR)
Generates code via causal left-to-right next-token prediction:

```python
from ar.hub import TelosModel

# Load pretrained AR checkpoint
model = TelosModel.from_pretrained("checkpoints/ar/12m/telos_12m_r15")

# Generate code completion
completion = model.complete(
    prompt="import os\ndef get_current_directory():\n",
    max_tokens=64,
    temperature=0.7
)
print(completion)
```

### D. Interactive Research Web Interface
For live interactive generation and diffusion step inspections:
- **Interactive Research Demo**: [https://telos.research.wingit.tech](https://telos.research.wingit.tech)


