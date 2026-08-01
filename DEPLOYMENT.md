# DEPLOYMENT — τέλος (télos) MDLM

This document details how to set up, reproduce, train, and deploy **télos (τέλος)** — a Masked Diffusion Language Model for Python code autocomplete.

---

## 1. Architecture Specifications

| Parameter | Phase A (1-Hour Local Run) | Phase B (H100 GPU Run) |
| :--- | :--- | :--- |
| **Model Size** | **~12.48 Million** ($1.25 \times 10^7$) | **~80.2 Million** ($80.2 \times 10^6$) |
| **Architecture** | Deep & Narrow (10 layers, $d=320$) | Deep & Narrow (16 layers, $d=672$) |
| **Attention** | Grouped-Query Attention (8 Query, 2 KV) | Grouped-Query Attention (12 Query, 2 KV) |
| **Vocabulary** | 4,096 BPE Tokens | 8,192 BPE Tokens |
| **Context Length** | 256 tokens | 512 tokens |
| **Token Budget** | 120 Million tokens | 1.70 Billion tokens |
| **Target Steps** | 7,500 steps (~60 mins on M5 Pro) | 13,000 steps (~2 hrs on H100) |
| **Precision** | `bf16` mixed precision (MPS) | `bf16` mixed precision (CUDA) |

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

## 3. Phase A: Local Training & Validation (M5 Pro MacBook Pro)

Phase A runs a 1-hour extended validation run locally on Apple Silicon.

### Step 1: Stream Online Python Dataset (120M Tokens)
Streams 120 Million tokens of real, AST-valid Python functions with docstrings directly from HuggingFace (`codeparrot/codeparrot-clean`):
```bash
uv run python scripts/prepare_data.py --config configs/phase_a.yaml
```

### Step 2: Train BPE Tokenizer
Trains a ByteLevel BPE tokenizer with a 4,096 vocabulary:
```bash
uv run python scripts/train_tokenizer.py --config configs/phase_a.yaml
```

### Step 3: Run 1-Hour Extended Training
Executes 7,500 training steps (~60 minutes on M5 Pro MPS):
```bash
uv run python scripts/train.py --config configs/phase_a.yaml
```

### Step 4: Run Automated Test Suite
Ensure all 9 unit tests pass:
```bash
uv run --with pytest pytest tests/ -v
```

---

## 4. Phase B: Production Training Run (H100 via Lightning AI / Cloud GPU)

Phase B trains the production ~80.2M Deep & Narrow + GQA model on 1.7 Billion tokens.

### Step 1: Prepare Production Dataset
Stream 1.7B tokens from HuggingFace's permissively-licensed Python repositories:
```bash
python scripts/prepare_data.py --config configs/phase_b.yaml --dataset bigcode/the-stack-v2-dedup
```

### Step 2: Train Production Tokenizer (8,192 Vocab)
```bash
python scripts/train_tokenizer.py --config configs/phase_b.yaml
```

### Step 3: Execute Production Training Run
```bash
python scripts/train.py --config configs/phase_b.yaml
```

---

## 5. Model Publishing & Public Deployment

### Step 1: Export Weights & Upload to HuggingFace Hub
Convert checkpoint to `safetensors` format and upload to HuggingFace Hub:
```bash
python -m telos.hub.upload --model-dir checkpoints/phase_b --repo-id kazenoko/telos-80m
```

### Step 2: Standalone Programmatic Inference
Run prediction using the `TelosModel` standalone API:
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
