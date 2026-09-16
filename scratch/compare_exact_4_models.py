"""
Compare the exact 4 models specified by the user:
1. 15M COROSred (TPU)
2. 50M COROSred (CUDA, 50M Lightning)
3. 75M COROSred (TPU)
4. 100M COROSred (TPU, alpha_min=0.5)

Evaluates:
- Architecture parameters & training specs
- 100 Contextual Probes (Causal & Infill Top-1, CE, Rank)
- Held-out 131K token Validation Causal CE & Infill CE
"""

import json
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np

from telos.eval.runner import load_model_from_checkpoint, _evaluate_single_probe_type
from telos.eval.probes import load_contextual_probes, PROBE_SUITE_100
from telos.data.tokenizer import load_tokenizer

MODELS = {
    "15M COROSred (TPU)": "checkpoints/corosred/unified/15m/checkpoint_final.pt",
    "50M COROSred (CUDA, Lightning)": "checkpoints/corosred/50m_lightning/checkpoint_final.pt",
    "75M COROSred (TPU)": "checkpoints/corosred/unified/75m_python/checkpoint_final.pt",
    "100M COROSred (TPU, alpha=0.5)": "checkpoints/corosred/unified/100m_5b/checkpoint_final.pt",
}

print("=" * 80)
print("  DIRECT 4-MODEL COMPARISON BENCHMARK")
print("=" * 80)

tokenizer = load_tokenizer("configs/tokenizer_mac.json")
probes_list = load_contextual_probes(num_probes=100)

results = {}

# Tail split tokens for validation CE
data_bin_path = Path("data/tokenized/python_corpus_5b.bin")
if data_bin_path.exists():
    raw_data = np.memmap(data_bin_path, dtype=np.uint16, mode="r")
    eval_tokens = raw_data[-131072:].astype(np.int64)
    chunks = eval_tokens.reshape(128, 1024)
else:
    chunks = None

for name, ckpt_path in MODELS.items():
    print(f"\nEvaluating: {name} ({ckpt_path})...")
    cp = Path(ckpt_path)
    if not cp.exists():
        print(f"  [!] Checkpoint not found: {cp}")
        continue

    model, backend, vocab_size = load_model_from_checkpoint(cp)
    model.eval()

    # Model metadata
    cfg_file = cp.parent / "config.json"
    cfg = {}
    if cfg_file.exists():
        try:
            with open(cfg_file) as f:
                cfg = json.load(f)
        except Exception:
            pass

    m_cfg = cfg.get("model", {})
    t_cfg = cfg.get("training", {})
    c_cfg = cfg.get("corosred", {})

    total_params = sum(p.numel() for p in model.parameters())

    # Probes: Causal
    causal_res = _evaluate_single_probe_type(model, tokenizer, backend, probes_list, probe_type="causal")
    c_top1 = causal_res["overall"]["top1_acc_pct"]
    c_ce = causal_res["overall"]["mean_ce"]
    c_rank = causal_res["overall"]["mean_rank"]

    # Probes: Infill
    infill_res = _evaluate_single_probe_type(model, tokenizer, backend, probes_list, probe_type="infill")
    i_top1 = infill_res["overall"]["top1_acc_pct"]
    i_ce = infill_res["overall"]["mean_ce"]
    i_rank = infill_res["overall"]["mean_rank"]

    # Validation CE on 131K tail slice (sample 16 chunks of 1024 to be fast)
    val_causal_ce = None
    if chunks is not None:
        sample_chunks = chunks[:16]  # 16,384 tokens
        sample_t = torch.from_numpy(sample_chunks.copy()).long()
        with torch.no_grad():
            out = model(sample_t)
            logits = out[0] if isinstance(out, tuple) else out
            loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.shape[-1]), sample_t[:, 1:].reshape(-1))
            val_causal_ce = loss.item()

    results[name] = {
        "params": total_params,
        "device": cfg.get("_device", "unknown"),
        "alpha_min": c_cfg.get("alpha_min", "N/A"),
        "alpha_max": c_cfg.get("alpha_max", "N/A"),
        "steps": t_cfg.get("max_steps", "N/A"),
        "causal_top1": c_top1,
        "causal_ce": c_ce,
        "causal_rank": c_rank,
        "infill_top1": i_top1,
        "infill_ce": i_ce,
        "infill_rank": i_rank,
        "val_causal_ce_16k": val_causal_ce,
    }

print("\n" + "=" * 95)
print(f"{'Model Name':<32} | {'Device':<6} | {'Params':<8} | {'Val CE':<7} | {'Cau T1':<7} | {'Cau CE':<7} | {'Inf T1':<7} | {'Inf CE':<7}")
print("-" * 95)
for name, r in results.items():
    val_str = f"{r['val_causal_ce_16k']:.4f}" if r['val_causal_ce_16k'] else "N/A"
    print(f"{name:<32} | {r['device']:<6} | {r['params']/1e6:>6.1f}M | {val_str:<7} | {r['causal_top1']:>6.1f}% | {r['causal_ce']:>6.2f}  | {r['infill_top1']:>6.1f}% | {r['infill_ce']:>6.2f}")
print("=" * 95)

with open("logs/direct_4model_comparison.json", "w") as f:
    json.dump(results, f, indent=2)
print("✓ Saved results to logs/direct_4model_comparison.json")
