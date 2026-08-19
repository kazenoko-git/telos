# DEPLOYMENT — télos (τέλος) MDLM & UNDLM

This document details how to set up, reproduce, train, evaluate, and deploy **télos (τέλος)** — a Discrete Diffusion Language Model for Python code autocomplete and non-monotonic generation (supporting both Masked Discrete Diffusion and Uniform Noise Diffusion paradigms).

---

## 1. Model Scaling & Hardware Specifications

| Suite | Model Size | Architecture | Vocab / Context | Backend | Hardware Target |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **12M Suite** | **~12.5 Million** | 6L, $d=256$, 8 Heads | 8,192 BPE / 512 | Apple MLX | Apple Silicon (MPS/Metal GPU) |
| **25M Suite** | **~25 Million** | 8L, $d=512$, 8 Heads | 8,192 BPE / 512 | Apple MLX | Apple Silicon (MPS/Metal GPU) |
| **50M Suite** | **~50 Million** | 8L, $d=768$, 12 Heads | 8,192 BPE / 512 | MLX / PyTorch-XLA | Apple Silicon / TPU v4/v5/v6e |

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

## 4. Training Pipelines

### Option A: Masked Diffusion (MDLM) Local Training
To train the 12M, 25M, or 50M parameter scaling suites locally on Apple Silicon Metal GPU:

```bash
# Open and run the master training notebook
jupyter notebook notebooks/masked/Training_Suites.ipynb

# Or execute RoPE adaptation & fine-tuning across models
jupyter notebook notebooks/masked/RoPE_Finetune_Suite.ipynb
```

### Option B: Uniform Noise Diffusion (UNDLM) Training
To train using the uniform noise diffusion paradigm:

```bash
# Uniform noise pipelines reside in undiff/
jupyter notebook notebooks/uniform/Training_Suites.ipynb
```

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

### Standalone Python Inference

```python
from mdiff.hub import TelosModel

# Load model pipeline from canonical checkpoint
model = TelosModel.from_pretrained("checkpoints/masked/50m/kappa_50m_1to35_mlx")

# Generate code completion
completion = model.complete(
    prompt="def sort_array(arr: list[int]) -> list[int]:\n",
    max_tokens=64,
    num_steps=64,
    temperature=0.3,
    schedule="linear"
)
print(completion)
```
