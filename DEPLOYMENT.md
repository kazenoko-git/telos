# DEPLOYMENT — τέλος (télos) MDLM

This document details how to set up, reproduce, train, and deploy **télos (τέλος)** — a Masked Diffusion Language Model for Python code autocomplete, natural language instructions, and shell command execution.

---

## 1. Model Phase & Hardware Specifications

| Parameter | Phase A (Local M5 Pro) | **Phase B (Flagship Autocomplete)** | **Phase C (14-Hour Multi-Domain Flagship)** |
| :--- | :--- | :--- | :--- |
| **Model Size** | **~12.48 Million** ($1.25 \times 10^7$) | **~232.4 Million** ($2.32 \times 10^8$) | **~365.1 Million** ($3.65 \times 10^8$) |
| **Architecture** | Deep & Narrow (10L, $d=320$) | Deep & Narrow (16L, $d=1152$) | **Deep & Narrow (20L, $d=1280$)** |
| **Attention** | GQA (8 Query, 2 KV) | GQA (16 Query, 4 KV) | **GQA (20 Query, 4 KV)** |
| **Vocabulary** | 4,096 BPE Tokens | 8,192 BPE Tokens | **16,384 BPE Tokens** (Multi-Domain) |
| **Context Length** | 256 tokens | 512 tokens | **512 tokens** |
| **Domain Mixture** | Pure Python | **100% Pure Python Code** | **60% Python, 25% English, 15% Shell** |
| **Token Budget** | 120 Million tokens | **8.0 Billion tokens** (34:1 overtraining ratio) | **25.0 Billion tokens** (Overtrained 68:1 ratio) |
| **Target Steps** | 7,500 steps (~60 mins) | **60,000 steps (~12.5 Mins on TPU v6e-1)** | **150,000 steps (~13.8 Hours on TPU v5e-8)** |
| **Hardware** | Apple Silicon MPS | Lightning AI TPU v6e-1 / 4x T4 | **Lightning AI TPU v6e-1 / Kaggle TPU** |

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

## 3. Phase B: Flagship Pure Python Autocomplete Model (~232.4M Params / 8B Tokens)

Executes a 12.5-minute flagship pure Python autocomplete model on Lightning AI TPU v6e-1.

```bash
# Step 1: Clone repo & install
git clone https://github.com/kazenoko-git/telos.git
cd telos
pip install -e .

# Step 2: Stream 8.0B ungated Python tokens & train 8k tokenizer (~32GB text file)
python scripts/prepare_data.py --config configs/phase_b.yaml --raw
python scripts/train_tokenizer.py --config configs/phase_b.yaml

# Step 3: Execute Phase B Training (~12.5 Mins on TPU v6e-1 or ~1.2 Hours on 2x T4)
python scripts/train.py --config configs/phase_b.yaml --device tpu
```

---

## 4. Phase C: 14-Hour Multi-Domain Flagship Model (~365.1M Params / 25B Tokens)

Phase C trains our largest **365.1M parameter** multi-domain model on **25 Billion tokens** across Python, English instructions, and UNIX/Windows shell commands.

```bash
# Step 1: Clone & prepare dataset
python scripts/prepare_data.py --config configs/phase_c.yaml --raw
python scripts/train_tokenizer.py --config configs/phase_c.yaml

# Step 2: Execute Phase C Training (~13.8 Hours across Kaggle/Lightning TPU sessions)
python scripts/train.py --config configs/phase_c.yaml --device tpu
```

---

## 5. Model Publishing & Public Deployment

### Step 1: Export Weights & Upload to HuggingFace Hub
```bash
python -m telos.hub.upload --model-dir checkpoints/phase_b --repo-id kazenoko/telos-230m-python
```

### Step 2: Standalone Programmatic Inference
```python
from telos.hub import TelosModel

model = TelosModel.from_pretrained("kazenoko/telos-230m-python")

# Python Code Completion
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
