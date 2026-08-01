# DEPLOYMENT — τέλος (télos) MDLM

This document details how to set up, reproduce, train, and deploy **télos (τέλος)** — a Masked Diffusion Language Model for Python code autocomplete, natural language instructions, and shell command execution.

---

## 1. Model Phase & Hardware Specifications

| Parameter | Phase A (Local M5 Pro) | **Phase B (3-Hour Python Run)** | **Phase C (14-Hour Multi-Domain Flagship)** |
| :--- | :--- | :--- | :--- |
| **Model Size** | **~12.48 Million** ($1.25 \times 10^7$) | **~141.3 Million** ($1.41 \times 10^8$) | **~365.1 Million** ($3.65 \times 10^8$) |
| **Architecture** | Deep & Narrow (10L, $d=320$) | Deep & Narrow (16L, $d=896$) | **Deep & Narrow (20L, $d=1280$)** |
| **Attention** | GQA (8 Query, 2 KV) | GQA (14 Query, 2 KV) | **GQA (20 Query, 4 KV)** |
| **Vocabulary** | 4,096 BPE Tokens | 8,192 BPE Tokens | **16,384 BPE Tokens** (Multi-Domain) |
| **Context Length** | 256 tokens | 512 tokens | **512 tokens** |
| **Domain Mixture** | Pure Python | Pure Python Code | **60% Python, 25% English, 15% Shell** |
| **Token Budget** | 120 Million tokens | 4.0 Billion tokens | **25.0 Billion tokens** (Overtrained 68:1 ratio) |
| **Target Steps** | 7,500 steps (~60 mins) | 30,000 steps (~2.8 Hours) | **150,000 steps (~13.8 Hours on TPU v5e-8)** |
| **Hardware** | Apple Silicon MPS | Kaggle TPU v5e-8 / 2x T4 | **Kaggle TPU v5e-8 (128GB VRAM)** |

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

## 3. Phase B: Upgraded 3-Hour Pure Python Run (~141.3M Params)

Executes a 3-hour pure Python training run on Kaggle TPU v5e-8 or 2x T4 GPUs.

```bash
# Step 1: Clone repo & install
!git clone https://github.com/kazenoko-git/telos.git
%cd telos
!pip install -e .

# Step 2: Stream 4.0B Python tokens & train 8k tokenizer
!python scripts/prepare_data.py --config configs/phase_b.yaml --dataset bigcode/the-stack-v2-dedup
!python scripts/train_tokenizer.py --config configs/phase_b.yaml

# Step 3: Execute Phase B Training (~2.8 Hours on TPU v5e-8)
!python scripts/train.py --config configs/phase_b.yaml --device tpu
```

---

## 4. Phase C: 14-Hour Multi-Domain Flagship Model (~365.1M Params / 25B Tokens)

Phase C trains our largest **365.1M parameter** multi-domain model on **25 Billion tokens** across Python, English instructions, and UNIX/Windows shell commands.

```bash
# Step 1: Clone & prepare dataset
!python scripts/prepare_data.py --config configs/phase_c.yaml
!python scripts/train_tokenizer.py --config configs/phase_c.yaml

# Step 2: Execute Phase C Training (~13.8 Hours across 2 Kaggle TPU sessions)
!python scripts/train.py --config configs/phase_c.yaml --device tpu
```

---

## 5. Model Publishing & Public Deployment

### Step 1: Export Weights & Upload to HuggingFace Hub
```bash
python -m telos.hub.upload --model-dir checkpoints/phase_c --repo-id kazenoko/telos-365m
```

### Step 2: Standalone Programmatic Inference
```python
from telos.hub import TelosModel

model = TelosModel.from_pretrained("kazenoko/telos-365m")

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
