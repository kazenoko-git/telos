# DEPLOYMENT.md — télos (τέλος) MDLM

This document details how to set up, reproduce, train, and deploy **télos** — a Masked Diffusion Language Model (MDLM) for Python code autocomplete.

---

## 1. Environment Setup

### Local Setup (Apple Silicon / M5 Pro / MPS)

1. Ensure Python >= 3.13 and `uv` package manager are installed.
2. Install dependencies:
   ```bash
   uv sync
   ```
3. Verify PyTorch Metal Performance Shaders (MPS) availability:
   ```bash
   python -c "import torch; print('MPS Available:', torch.backends.mps.is_available())"
   ```

---

## 2. Phase A: Local Training & Validation (M5 Pro)

Phase A validates the MDLM pipeline on local hardware before allocating remote GPU compute.

### Step 1: Prepare Tokenizer & Dataset
Extract a small Python function corpus (~30M tokens) and train a Byte-Pair Encoding (BPE) tokenizer (vocab size 4,096):
```bash
python scripts/prepare_data.py --config configs/phase_a.yaml
python scripts/train_tokenizer.py --config configs/phase_a.yaml
```

### Step 2: Run Training & Pipeline Validation
Execute training for 5,000 steps (~5-10 mins):
```bash
python scripts/train.py --config configs/phase_a.yaml
```

### Step 3: Run Unit Test Suite
Ensure loss reweighting ($1/t$), bidirectional attention, checkpoint continuity, and sampling function properly:
```bash
pytest tests/ -v
```

---

## 3. Phase B: Final Training Run (H100 via Lightning AI)

Phase B executes a single compute-optimal training run (~85M params, ~1.7B tokens) within a strict ≤2 hour budget.

### Setup on Lightning AI H100 Studio

1. Spin up a GPU Studio with 1x NVIDIA H100 SXM (80GB).
2. Clone repository & install dependencies:
   ```bash
   git clone <repo-url> telos
   cd telos
   pip install -e .
   ```
3. Prepare dataset (or download pre-processed tokens):
   ```bash
   python scripts/prepare_data.py --config configs/phase_b.yaml
   ```
4. Run full training with mixed precision (`bf16`) & dynamic checkpointing:
   ```bash
   python scripts/train.py --config configs/phase_b.yaml
   ```

---

## 4. Model Publishing & Deployment

### Step 1: Export & Upload to HuggingFace Hub
Upload model safetensors, tokenizer, and config:
```bash
python -m telos.hub.upload --model-dir checkpoints/phase_b/best --repo-id kazenoko/telos-85m
```

### Step 2: Local / Programmatic Inference
Run predictions using the custom standalone inference API:
```python
from telos.hub import TelosModel

model = TelosModel.from_pretrained("kazenoko/telos-85m")
code = model.complete(
    "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n",
    max_tokens=128,
    num_steps=64,
    temperature=0.8
)
print(code)
```

### Step 3: Deploy Interactive Web Demo (HuggingFace Spaces)
Deploy the Gradio app to a free CPU HuggingFace Space:
1. Create a new Space on HuggingFace named `telos-demo` (SDK: Gradio).
2. Upload `telos/hub/gradio_app.py` as `app.py` alongside `requirements.txt`.
3. The Space will automatically run interactive masked diffusion generation with step-by-step unmasking visualization.
