"""Verify the TPU resume fix in trainer_pytorch.py load_checkpoint: fp32 master
weights must be synced from loaded model params (not stale init).

Runs on CPU (is_tpu=False) but exercises the same master-weights sync code path
via the use_master_weights flag by monkeypatching. Since the master sync logic
guards on getattr(self, 'use_master_weights', False), we verify directly.
"""
import torch
import torch.nn as nn

import sys
sys.path.insert(0, ".")

from telos.models.transformer import TelosTransformer
from telos.training.trainer_pytorch import UnifiedPyTorchTrainer

cfg = {
    "model": {"vocab_size": 64, "d_model": 32, "n_layers": 2, "n_heads": 4,
              "n_kv_heads": 4, "seq_len": 16, "max_seq_len": 16},
    "training": {"precision": "bf16", "max_steps": 3, "batch_size": 2},
    "checkpoint": {},
    "data": {"synthetic": True},
    "paradigm": "ar",
}

model = TelosTransformer(
    vocab_size=64, d_model=32, n_layers=2, n_heads=4, n_kv_heads=4,
    is_causal=True, max_seq_len=16, seq_len=16,
)
trainer = UnifiedPyTorchTrainer(paradigm="ar", model=model, cfg=cfg, device_type="cpu")

# --- Simulate the TPU master-weights state on CPU ---
# Build the same structures __init__ builds on TPU
trainer.use_master_weights = True
trainer.param_to_master = [(p, p.detach().clone().float().requires_grad_(True))
                           for _, p in trainer.model.named_parameters() if p.requires_grad]

# Simulate a trained checkpoint: weights different from init
with torch.no_grad():
    for p in trainer.model.parameters():
        p.add_(0.5)

# Save it
sd = {k: v.clone() for k, v in trainer.model.state_dict().items()}
torch.save({"global_step": 1, "model_state_dict": sd,
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "scheduler_state_dict": trainer.scheduler.state_dict(),
            "config": cfg}, "scratch/fake_resume_ckpt.pt")

# Reset model to init-like state, then load
with torch.no_grad():
    for p in trainer.model.parameters():
        p.sub_(0.5)

trainer.load_checkpoint("scratch/fake_resume_ckpt.pt")

# --- The assertion: masters must now equal loaded (trained) weights ---
ok = True
for p, mp in trainer.param_to_master:
    d = (mp.data - p.data.float()).abs().max().item()
    if d > 1e-6:
        ok = False
        print(f"MISMATCH: master vs model max diff {d}")
loaded_delta = (sd["tok_embeddings.weight"] - model.tok_embeddings.weight).abs().max().item()
print(f"model weights restored from checkpoint: {loaded_delta < 1e-6}")
print(f"master weights synced from loaded params: {ok}")
assert ok and loaded_delta < 1e-6
print("\nRESUME FIX VERIFIED")
