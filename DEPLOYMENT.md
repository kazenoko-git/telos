# DEPLOYMENT — τέλος (télos) MDLM

This document details how to set up, reproduce, train, and deploy **télos (τέλος)** — a Masked Diffusion Language Model for Python code autocomplete.

---

## 1. Model Phase & Hardware Comparison

| Parameter | Phase A (Local M5 Pro) | Phase B (Standard TPU/H100) | **Phase C (7-Hour Flagship Overtrained)** |
| :--- | :--- | :--- | :--- |
| **Model Size** | **~12.48 Million** ($1.25 \times 10^7$) | **~80.2 Million** ($80.2 \times 10^6$) | **~232.4 Million** ($2.32 \times 10^8$) |
| **Architecture** | Deep & Narrow (10L, $d=320$) | Deep & Narrow (16L, $d=672$) | **Deep & Narrow (16L, $d=1152$)** |
| **Attention** | GQA (8 Query, 2 KV) | GQA (12 Query, 2 KV) | **GQA (16 Query, 4 KV)** |
| **Vocabulary** | 4,096 BPE Tokens | 8,192 BPE Tokens | **8,192 BPE Tokens** |
| **Context Length** | 256 tokens | 512 tokens | **512 tokens** |
| **Token Budget** | 120 Million tokens | 1.70 Billion tokens | **10.0 Billion tokens** (Overtrained 43:1 ratio) |
| **Target Steps** | 7,500 steps (~60 mins) | 13,000 steps (~50 mins) | **75,000 steps (~6.9 Hours on TPU v5e-8)** |
| **Hardware** | Apple Silicon MPS | H100 / TPU v5e-8 | **Kaggle TPU v5e-8 (128GB VRAM)** |

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

## 3. Phase C: 7-Hour Flagship Overtrained Model (~232.4M Params / 10B Tokens)

Phase C trains our largest, overtrained **232.4M parameter** model on **10 Billion Python tokens** for ~7 hours on Kaggle TPU v5e-8 ($0 cost).

### Commands for Kaggle TPU Notebook:
```bash
# Step 1: Clone repo & install
!git clone https://github.com/kazenoko-git/telos.git
%cd telos
!pip install -e .

# Step 2: Stream 10B Python tokens & train tokenizer
!python scripts/prepare_data.py --config configs/phase_c.yaml --dataset bigcode/the-stack-v2-dedup
!python scripts/train_tokenizer.py --config configs/phase_c.yaml

# Step 3: Run Phase C Flagship Training (~6.9 Hours on TPU v5e-8)
!python scripts/train.py --config configs/phase_c.yaml --device tpu
```

---

## 4. Phase A & Phase B Summary

### Phase A (Local M5 Pro)
```bash
uv run python scripts/prepare_data.py --config configs/phase_a.yaml
uv run python scripts/train_tokenizer.py --config configs/phase_a.yaml
uv run python scripts/train.py --config configs/phase_a.yaml
```

### Phase B (50-Minute TPU/GPU Run)
```bash
python scripts/prepare_data.py --config configs/phase_b.yaml --dataset bigcode/the-stack-v2-dedup
python scripts/train_tokenizer.py --config configs/phase_b.yaml
python scripts/train.py --config configs/phase_b.yaml --device tpu
```

---

## 5. Model Publishing & Public Deployment

### Step 1: Export Weights & Upload to HuggingFace Hub
```bash
python -m telos.hub.upload --model-dir checkpoints/phase_c --repo-id kazenoko/telos-230m
```

### Step 2: Standalone Programmatic Inference
```python
from telos.hub import TelosModel

model = TelosModel.from_pretrained("kazenoko/telos-230m")

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
