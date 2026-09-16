"""
Direct Prompt Comparison Suite across 50M CUDA, 75M TPU, and 100M TPU.
Runs both Causal Code Completions and Bidirectional Infilling directly.
"""

import json
from pathlib import Path
import torch
import torch.nn.functional as F

from telos.eval.runner import load_model_from_checkpoint, _generate_greedy_completion, clean_functional_completion
from telos.data.tokenizer import load_tokenizer

MODELS = [
    ("50M COROSred (CUDA, Lightning)", "checkpoints/corosred/50m_lightning/checkpoint_final.pt"),
    ("75M COROSred (TPU)", "checkpoints/corosred/unified/75m_python/checkpoint_final.pt"),
    ("100M COROSred (TPU, alpha=0.5)", "checkpoints/corosred/unified/100m_5b/checkpoint_final.pt"),
]

tokenizer = load_tokenizer("configs/tokenizer_mac.json")

# Load models into memory
loaded_models = []
for name, ckpt_path in MODELS:
    m, b, _ = load_model_from_checkpoint(ckpt_path)
    m.eval()
    loaded_models.append((name, m, b))

# -------------------------------------------------------------
# Part 1: Causal Code Generation Prompts
# -------------------------------------------------------------
CAUSAL_PROMPTS = [
    (
        "Task 18 (Private Unseen Benchmark: algo_task_19)",
        'def algo_task_19_numeric_op(val: int) -> int:\n    """Computes modular arithmetic transformation (val * 19 + 361) % 10007."""\n'
    ),
    (
        "Task 20 (Private Unseen Benchmark: algo_task_21)",
        'def algo_task_21_numeric_op(val: int) -> int:\n    """Computes modular arithmetic transformation (val * 21 + 441) % 10007."""\n'
    ),
    (
        "Fibonacci Sequence",
        'def fibonacci(n: int) -> int:\n    """Returns the n-th Fibonacci number."""\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n'
    ),
    (
        "List Reversal",
        'def reverse_list(items: list) -> list:\n    """Returns a new list with elements in reverse order."""\n'
    ),
    (
        "Stack Data Structure",
        'class Stack:\n    """A simple LIFO stack implementation."""\n    def __init__(self):\n        self.items = []\n\n    def push(self, item):\n'
    )
]

print("\n" + "=" * 90)
print("  PART 1: DIRECT CAUSAL GREEDY COMPLETIONS (T=0.0)")
print("=" * 90)

for title, prompt in CAUSAL_PROMPTS:
    print(f"\n>>> PROMPT: {title}")
    print("-" * 70)
    print(prompt.rstrip())
    print("-" * 70)
    
    for name, model, backend in loaded_models:
        raw = _generate_greedy_completion(model, tokenizer, backend, prompt, max_new_tokens=48)
        clean = clean_functional_completion(prompt, raw)
        # Format single-line or multi-line display cleanly
        lines = [l for l in clean.splitlines() if l.strip()]
        disp = "\n    ".join(lines) if lines else "<EMPTY / EOS>"
        print(f"[{name}]\n    {disp}\n")

# -------------------------------------------------------------
# Part 2: Bidirectional Infilling Prompts ([MASK])
# -------------------------------------------------------------
INFILL_PROMPTS = [
    (
        "Contextual Attribute Infill",
        "def compute_area(width: float, height: float) -> float:\n    return width * ",
        "\n"
    ),
    (
        "Syntactic Keyword Infill",
        "with open(filepath, 'r') ",
        " file:\n    content = file.read()\n"
    ),
    (
        "Idiomatic Import Infill",
        "import torch.nn as ",
        "\n\nclass Model(nn.Module):\n    pass\n"
    ),
    (
        "Suffix-Clued Getter Infill",
        "def get_",
        "(self):\n    return self._data\n"
    ),
]

print("\n" + "=" * 90)
print("  PART 2: BIDIRECTIONAL [MASK] INFILLING PREDICTIONS")
print("=" * 90)

MASK_ID = 1  # 1 is [MASK]

for title, prefix, suffix in INFILL_PROMPTS:
    print(f"\n>>> INFILL PROMPT: {title}")
    p_ids = tokenizer.encode(prefix).ids
    s_ids = tokenizer.encode(suffix).ids
    input_ids = p_ids + [MASK_ID] + s_ids
    mask_pos = len(p_ids)
    
    print(f"Context: {prefix!r} + [MASK] + {suffix!r}")
    print("-" * 70)
    
    for name, model, backend in loaded_models:
        x = torch.tensor([input_ids], dtype=torch.long)
        with torch.no_grad():
            out = model(x, mask_override=False)
            logits = out[0] if isinstance(out, tuple) else out
            mask_logits = logits[0, mask_pos]
            probs = F.softmax(mask_logits, dim=-1)
            
            top5_probs, top5_ids = torch.topk(probs, 5)
            top1_tok = tokenizer.decode([top5_ids[0].item()])
            top5_str = ", ".join([f"{tokenizer.decode([tid.item()])!r} ({p.item():.1%})" for p, tid in zip(top5_probs, top5_ids)])
            
            print(f"[{name}]")
            print(f"  Top-1: {top1_tok!r} ({top5_probs[0].item():.1%})")
            print(f"  Top-5: {top5_str}")
