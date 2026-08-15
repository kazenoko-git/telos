# DEPLOYMENT — télos (τέλος) MDLM

This document details how to set up, reproduce, train, evaluate, and deploy **télos (τέλος)** — a Masked Diffusion Language Model for Python code autocomplete and non-monotonic generation.

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
uv run python scripts/prepare_data.py --config configs/phase_b_50m_1to40_mlx.yaml

# 2. Train BPE Tokenizer (if creating custom vocabulary)
uv run python scripts/train_tokenizer.py --config configs/phase_b_50m_1to40_mlx.yaml
```

---

## 4. Training Pipelines

### Option A: Apple MLX Scaling Suites (Jupyter Notebook / Script)
To train the 12M, 25M, or 50M parameter scaling suites locally on Apple Silicon Metal GPU:

```bash
# Open and run the master training notebook
jupyter notebook notebooks/Training_Suites.ipynb

# Or execute RoPE adaptation & fine-tuning across all 15 models
jupyter notebook notebooks/RoPE_Finetune_Suite.ipynb
```
Or execute via script entry points:
```bash
uv run python scripts/train_mlx.py --config configs/phase_b_50m_1to40_mlx.yaml
```

### Option B: TPU Scaling Suites (PyTorch-XLA)
For training on Google Cloud TPU / Kaggle TPU:

```bash
uv run python scripts/train_tpu_50m_suite.py
```

---

## 5. Evaluation & Contextual Probes Benchmark

Run the 101 contextual probes benchmark (measuring rank, target cross-entropy, top-1/5 accuracies across 8 syntactic categories):

```bash
# Run probes on an MLX checkpoint
uv run python notebooks/Evaluation.py --checkpoint checkpoints/phase_b_50m_1to40_mlx --mode probes

# Run qualitative code generation samples
uv run python notebooks/Evaluation.py --checkpoint checkpoints/phase_b_50m_1to40_mlx --mode sample
```

Outputs will be automatically saved to `probes_output/`.

---

## 6. Inference Deployment & Web Interface

### Standalone Python Inference

```python
from telos.hub import TelosModel

# Load model pipeline
model = TelosModel.from_pretrained("checkpoints/phase_b_50m_1to35_mlx_20260811_231951")

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

### Interactive Gradio Application

Launch the local web application for live interactive code completions:

```bash
uv run python telos/hub/gradio_app.py
```

### HuggingFace Hub Publishing

Upload trained checkpoint assets to the HuggingFace Hub:

```bash
python -m telos.hub.upload --model-dir checkpoints/phase_b_50m_1to40_mlx --repo-id kazenoko/telos-50m-coder
```
