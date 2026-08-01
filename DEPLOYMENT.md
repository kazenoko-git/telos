# DEPLOYMENT — τέλος (télos) MDLM

This document details how to set up, reproduce, train, and deploy **télos (τέλος)** — a Masked Diffusion Language Model for Python code autocomplete.

---

## 1. Hardware & Compute Accelerator Comparison

| Accelerator | Hardware Type | Peak Compute | VRAM / Memory | Target 7,500 Step Time | Setup Required |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **TPU v5e-8** | Google TPU Pod (8 Chips) | **~1,570 TFLOPS** (`bf16`) | **128 GB HBM** | **~2 to 3 Minutes** 🚀 | `torch_xla` (Built into Kaggle) |
| **NVIDIA H100** | 1x SXM GPU | ~989 TFLOPS (`bf16`) | 80 GB HBM3 | ~5 to 8 Minutes | Native CUDA |
| **T4 x2 GPUs** | 2x NVIDIA T4 GPUs | ~130 TFLOPS (`fp16`) | 32 GB VRAM | ~18 to 20 Minutes | Native CUDA |
| **Local M5 Pro** | Apple Silicon MPS | ~40 TFLOPS (`bf16`) | 24 GB Unified | ~60 Minutes | Native MPS |

---

## 2. Environment Setup

### Prerequisites
- Python >= 3.13
- `uv` package manager (`curl -LsSf https://ast.sh/uv/install.sh | sh`)

### Installation
```bash
git clone https://github.com/kazenoko-git/telos.git
cd telos
uv sync
```

---

## 3. Training on Kaggle TPU v5e-8 (Sub-3 Minute Warp Speed)

Kaggle provides **20 free TPU hours per week** (TPU v5e-8 with 128GB VRAM).

1. Create a Kaggle Notebook $\rightarrow$ set **Accelerator** to **TPU v5e-8**.
2. Run in notebook cell:
```bash
!git clone https://github.com/kazenoko-git/telos.git
%cd telos
!pip install -e .
!python scripts/train.py --config configs/phase_a.yaml --device tpu
```

---

## 4. Phase A: Local Training & Validation (M5 Pro MacBook Pro)

Phase A runs locally on Apple Silicon Metal MPS.

```bash
uv run python scripts/prepare_data.py --config configs/phase_a.yaml
uv run python scripts/train_tokenizer.py --config configs/phase_a.yaml
uv run python scripts/train.py --config configs/phase_a.yaml
```

---

## 5. Phase B: Production Training Run (H100 / Cloud GPU)

Phase B trains the production ~80.2M Deep & Narrow + GQA model on 1.7 Billion tokens.

```bash
python scripts/prepare_data.py --config configs/phase_b.yaml --dataset bigcode/the-stack-v2-dedup
python scripts/train_tokenizer.py --config configs/phase_b.yaml
python scripts/train.py --config configs/phase_b.yaml --device cuda
```

---

## 6. Model Publishing & Public Deployment

### Step 1: Export Weights & Upload to HuggingFace Hub
```bash
python -m telos.hub.upload --model-dir checkpoints/phase_b --repo-id kazenoko/telos-80m
```

### Step 2: Standalone Programmatic Inference
```python
from telos.hub import TelosModel

model = TelosModel.from_pretrained("kazenoko/telos-80m")

code = model.complete(
    "def fibonacci(n: int) -> int:\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n",
    max_tokens=128,
    num_steps=64,
    temperature=0.4
)
print(code)
```

### Step 3: Interactive Web Demo (HuggingFace Spaces)
Deploy `telos/hub/gradio_app.py` to a free-tier CPU HuggingFace Space for interactive code completion generation with step-by-step unmasking speed controls.
