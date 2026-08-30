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

## 4. Training Pipelines & Benchmarking

### Option A: Google Cloud / Kaggle TPU v5e-8 (PyTorch-XLA SPMD)
To train 25M upscaled or 12.5M models across all 8 TPU cores using single-process SPMD data-parallel sharding:
1. **Execute Full 3-Paradigm 25M Suite via Script**:
   ```bash
   python scripts/colab/train_25m_upscaled.py --ratios r1 r10 r15 r20 r25 r30 r35 --hf-repo Kazenowoko/telos --device tpu
   ```
   *Performance Note*: The training engine automatically stages the dataset directly into TPU HBM (`~1.6 GB`), eliminating CPU-side gather bottlenecks. Logging and step synchronization use asynchronous step closures (`xm.add_step_closure`) to maintain continuous MXU pipeline saturation.
2. **Or Execute Interactively in Kaggle / Colab**:
   Open [`notebooks/shared/Unified_Training_Suite.ipynb`](notebooks/shared/Unified_Training_Suite.ipynb) which automatically detects the TPU environment, initializes the SPMD 8-chip mesh, and distributes global batches without multi-process forking.

### Option B: Kaggle & Cloud Multi-GPU Training (2x Tesla T4 / CUDA DataParallel)
To train on Kaggle (GPU T4 x2) or cloud NVIDIA GPUs with multi-GPU parallelization:
1. **Set Accelerator to `GPU T4 x2` and toggle Internet ON** in Kaggle notebook settings.
2. **Execute via CLI**:
   ```bash
   git clone https://github.com/kazenoko-git/telos.git
   cd telos
   pip install -q safetensors huggingface_hub pyyaml einops tokenizers
   python scripts/colab/train_25m_upscaled.py --device cuda --ratios r1 r10 --hf-repo Kazenowoko/telos
   ```
   *Note*: The training engine automatically detects `2x T4 GPUs`, scales the global batch size, wraps the model in `torch.nn.DataParallel`, and uses `FP16` + `GradScaler` for maximum Turing Tensor Core throughput.

### Option B: Local Apple Silicon Unified Training (MLX)
To train all three paradigms under identical configurations locally via MLX:
```bash
jupyter notebook notebooks/shared/Unified_Training_Suite.ipynb
```

### Option C: Throughput & Optimization Benchmark Suite
To test steps/sec, tokens/sec, and memory footprint across model scales (5M–100M) and batch sizes:
```bash
jupyter notebook notebooks/shared/Optimization_Test_Suite.ipynb
# Or run standalone CLI:
python scripts/shared/run_optimization_suite.py
```

### Option D: Lightning AI TPU v6e-1 Trillium (PyTorch-XLA Single-Chip 32GB HBM)
To train 25M upscaled models on Lightning AI single-chip TPU v6e:
1. **Connect via SSH to Lightning AI Studio and run bootstrap**:
   ```bash
   git clone https://github.com/kazenoko-git/telos.git
   cd telos
   bash scripts/lightning/setup.sh
   ```
2. **Execute 25M 3-Paradigm Training Pipeline**:
   ```bash
   export HF_TOKEN="your_huggingface_token"
   python scripts/lightning/train_25m_lightning.py --ratios r15 r20 r25 r30 r35 --hf_repo Kazenowoko/telos
   ```

### Option E: Paradigm-Specific Local Notebooks
- **Masked Discrete Diffusion (MDLM)**: `notebooks/masked/Training_Suites.ipynb`
- **Uniform Noise Diffusion (UNDLM)**: `notebooks/uniform/Training_Suites.ipynb`
- **Autoregressive Baseline (AR)**: `notebooks/ar/Training_Suites.ipynb`


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


