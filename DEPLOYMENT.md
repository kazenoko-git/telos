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

For Apple Silicon (MLX Metal GPU acceleration):
```bash
uv pip install mlx tokenizers torch pyyaml
```

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

### Option A: Google Colab Cloud TPU (v5e-1 / v6e-1 PyTorch-XLA)
To train 25M upscaled or 12.5M models on Google Cloud TPUs:
1. **Provision TPU Session via Colab CLI**:
   ```bash
   uv run colab --auth adc new -s Untitled1_TPU --tpu v5e1
   ```
2. **Execute Full 3-Paradigm 25M Retraining on TPU**:
   ```bash
   uv run colab --auth adc exec -s Untitled1_TPU --timeout 86400 -f scripts/colab/run_tpu_direct.py
   ```

### Option B: Google Colab / Cloud GPU Training (PyTorch CUDA with SSH)
To train on cloud NVIDIA GPUs (T4 / A100):
1. **Upload weights, tokenizer & dataset to Hugging Face**:
   ```bash
   python scripts/shared/upload_to_hf.py --repo-id Kazenowoko/telos
   ```
2. **Execute in Google Colab**:
   Open [`notebooks/colab/Unified_25M_Upscaling_Colab.ipynb`](notebooks/colab/Unified_25M_Upscaling_Colab.ipynb) or run via Colab SSH:
   ```bash
   python scripts/colab/train_25m_upscaled.py --ratios r1 r10 r20 r25 --hf-repo kazenoko/telos
   ```

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

### Option D: Paradigm-Specific Local Notebooks
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


