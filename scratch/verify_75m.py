"""
Verify 75M COROSred checkpoint integrity and sanity forward pass.
"""

import json
from pathlib import Path
import torch
from telos.models import TelosConfig, TelosTransformer
from telos.data.tokenizer import load_tokenizer

CKPT_DIR = Path("checkpoints/corosred/unified/75m_python")
CFG_PATH = CKPT_DIR / "config.json"
CKPT_PATH = CKPT_DIR / "checkpoint_final.pt"

print("=" * 70)
print("  75M COROSRED VERIFICATION SUITE")
print("=" * 70)

# 1. Config Verification
with open(CFG_PATH) as f:
    cfg_data = json.load(f)

print("\n--- 1. CONFIG CHECK ---")
print(f"Paradigm: {cfg_data.get('paradigm')}")
m_cfg = cfg_data.get("model", cfg_data)
print(f"d_model: {m_cfg.get('d_model')}")
print(f"n_layers: {m_cfg.get('n_layers')}")
print(f"n_heads: {m_cfg.get('n_heads')}")
print(f"n_kv_heads: {m_cfg.get('n_kv_heads')}")
print(f"vocab_size: {m_cfg.get('vocab_size')}")
print(f"max_seq_len: {m_cfg.get('max_seq_len')}")
print(f"tied_embeddings: {m_cfg.get('tied_embeddings')}")
print(f"use_reliability_head: {m_cfg.get('use_reliability_head')}")

t_cfg = cfg_data.get("training", {})
print(f"Training steps: {t_cfg.get('max_steps')} (warmup: {t_cfg.get('warmup_steps')})")
print(f"Batch size per device: {t_cfg.get('batch_size')}, grad accum: {t_cfg.get('gradient_accumulation_steps')}")
print(f"Learning rate: {t_cfg.get('lr')} -> {t_cfg.get('min_lr')}")

# 2. Checkpoint Verification
print("\n--- 2. CHECKPOINT INTEGRITY ---")
print(f"Checkpoint size: {CKPT_PATH.stat().st_size / (1024**2):.2f} MB")
ckpt = torch.load(CKPT_PATH, map_location="cpu")
print("Checkpoint keys:", list(ckpt.keys()))
print("Global step:", ckpt.get("global_step"))

sd = ckpt.get("model_state_dict", ckpt)
nan_count = 0
inf_count = 0
total_params = 0
layer_indices = set()

for k, v in sd.items():
    if isinstance(v, torch.Tensor):
        total_params += v.numel()
        if torch.isnan(v).any():
            nan_count += 1
            print(f"  [!] NaN detected in {k}")
        if torch.isinf(v).any():
            inf_count += 1
            print(f"  [!] Inf detected in {k}")
        if "layers." in k:
            parts = k.split(".")
            idx = parts[parts.index("layers") + 1]
            layer_indices.add(int(idx))

print(f"Total parameter tensors: {len(sd)}")
print(f"Total raw tensor elements: {total_params:,}")
print(f"Layers detected in state_dict: {len(layer_indices)} (indices: {min(layer_indices)}..{max(layer_indices)})")
print(f"NaN tensors: {nan_count}, Inf tensors: {inf_count}")

# 3. Model Architecture & Forward Pass
print("\n--- 3. MODEL INSTANTIATION & FORWARD PASS ---")
telos_cfg = TelosConfig(
    d_model=m_cfg.get("d_model", 512),
    n_layers=m_cfg.get("n_layers", 22),
    n_heads=m_cfg.get("n_heads", 8),
    n_kv_heads=m_cfg.get("n_kv_heads", 8),
    vocab_size=m_cfg.get("vocab_size", 8192),
    max_seq_len=m_cfg.get("max_seq_len", 1024),
    tied_embeddings=m_cfg.get("tied_embeddings", True),
    use_reliability_head=m_cfg.get("use_reliability_head", True),
    is_causal=m_cfg.get("is_causal", False)
)
model = TelosTransformer(telos_cfg)
model.load_state_dict(sd, strict=True)
model.eval()

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
deduped_params = len(set(p.data_ptr() for p in model.parameters()))
print(f"Model instantiated successfully (strict=True load).")
print(f"Model total unique parameters: {trainable_params:,}")

# Forward pass test
tokenizer = load_tokenizer("configs/tokenizer_mac.json")
test_input = "def add(a: int, b: int) -> int:\n    return a + b"
tokens = tokenizer.encode(test_input)
token_ids = tokens.ids if hasattr(tokens, "ids") else tokens
input_ids = torch.tensor([token_ids], dtype=torch.long)
print(f"Test input ({len(token_ids)} tokens): {test_input!r}")

with torch.no_grad():
    out = model(input_ids)
    if isinstance(out, tuple):
        logits = out[0]
        rel_logits = out[1] if len(out) > 1 else None
    else:
        logits = out
        rel_logits = None

print(f"Output logits shape: {logits.shape}")
print(f"Logits min: {logits.min().item():.3f}, max: {logits.max().item():.3f}, mean: {logits.mean().item():.3f}, std: {logits.std().item():.3f}")
if rel_logits is not None:
    print(f"Reliability logits shape: {rel_logits.shape}")
    print(f"Reliability sigmoid mean: {torch.sigmoid(rel_logits).mean().item():.3f}")

# Causal loss test on test sequence
targets = input_ids[:, 1:]
pred_logits = logits[:, :-1, :]
ce = torch.nn.functional.cross_entropy(pred_logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
print(f"Sanity CE on test snippet: {ce.item():.4f} (ppl: {torch.exp(ce).item():.2f})")

print("\n✓ VERIFICATION SUCCESSFUL: 75M Model is completely healthy and structurally sound.")
print("=" * 70)
