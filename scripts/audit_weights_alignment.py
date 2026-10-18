#!/usr/bin/env python3
"""Audit parameter alignment and weight similarity between AR and MDLM models.

Evaluates:
1. Exact shape matches and output head alignment.
2. Tensor-by-tensor Frobenius norms, cosine similarities, and relative distances.
3. Layer-by-layer progression of weight divergence across depth.
4. Intermediate MLP neuron alignment (checking whether neurons are aligned or unaligned).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from dclm_model import DEFAULT_AR_WEIGHTS, DEFAULT_MDLM_WEIGHTS, DCLMConfig

ROOT = Path(__file__).resolve().parents[1]
OUT_FILE = ROOT / "research/matched_530m_corosred_probe_2026-10-18/weight_alignment.json"


def load_all_weights() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    cfg = DCLMConfig()
    ar_state = {}
    with safe_open(str(DEFAULT_AR_WEIGHTS), framework="pt", device="cpu") as f:
        for k in f.keys():
            t = f.get_tensor(k).float()
            if k == "model.transformer.wte.weight":
                ar_state["wte.weight"] = t
            elif k == "model.transformer.ln_f.weight":
                ar_state["ln_f.weight"] = t
            elif k == "model.transformer.ff_out.weight":
                ar_state["lm_head.weight"] = t[: cfg.vocab_size, :]
            elif k.startswith("model.transformer.blocks."):
                ar_state[k.replace("model.transformer.", "")] = t

    mdlm_state = {}
    with safe_open(str(DEFAULT_MDLM_WEIGHTS), framework="pt", device="cpu") as f:
        for k in f.keys():
            mdlm_state[k] = f.get_tensor(k).float()

    return ar_state, mdlm_state


def compute_alignment() -> dict:
    ar_weights, mdlm_weights = load_all_weights()
    assert set(ar_weights.keys()) == set(mdlm_weights.keys()), "Parameter key mismatch"

    per_tensor = {}
    layer_cosines: dict[int, list[float]] = {i: [] for i in range(16)}

    for name in sorted(ar_weights.keys()):
        w_ar = ar_weights[name]
        w_mdlm = mdlm_weights[name]
        assert w_ar.shape == w_mdlm.shape, f"Shape mismatch for {name}: {w_ar.shape} vs {w_mdlm.shape}"

        norm_ar = float(torch.linalg.norm(w_ar).item())
        norm_mdlm = float(torch.linalg.norm(w_mdlm).item())
        dot = float(torch.sum(w_ar * w_mdlm).item())
        cos = dot / max(1e-12, norm_ar * norm_mdlm)
        diff_norm = float(torch.linalg.norm(w_ar - w_mdlm).item())
        rel_diff = diff_norm / max(1e-12, norm_ar)

        entry = {
            "shape": list(w_ar.shape),
            "numel": w_ar.numel(),
            "norm_ar": norm_ar,
            "norm_mdlm": norm_mdlm,
            "cosine_similarity": cos,
            "relative_frobenius_distance": rel_diff,
        }
        per_tensor[name] = entry

        if name.startswith("blocks."):
            layer_idx = int(name.split(".")[1])
            layer_cosines[layer_idx].append(cos)

    # Layer-wise summary
    layer_summary = {}
    for layer_idx, cosines in layer_cosines.items():
        layer_summary[f"layer_{layer_idx}"] = {
            "mean_cosine": sum(cosines) / len(cosines),
            "min_cosine": min(cosines),
            "max_cosine": max(cosines),
        }

    # Neuron alignment check for MLP in Layer 0 and Layer 8
    # In SwiGLU: ff_proj has shape [10752, 1344] representing 5376 pairs of (linear, gate)
    # Check max cosine similarity of neuron rows
    mlp_diagnostics = {}
    for check_layer in [0, 8]:
        key = f"blocks.{check_layer}.ff_out.weight"
        # ff_out has shape [1344, 5376]; columns are intermediate neuron features
        col_ar = F.normalize(ar_weights[key], dim=0)    # [1344, 5376]
        col_md = F.normalize(mdlm_weights[key], dim=0)  # [1344, 5376]
        # Diagonal cosine (assuming direct neuron alignment without permutation)
        diag_cos = float(torch.sum(col_ar * col_md, dim=0).mean().item())
        # Cosine under greedy best-matching
        # Sample first 256 neurons to avoid 5376x5376 matrix on CPU if slow
        sim_matrix = torch.matmul(col_ar[:, :256].T, col_md[:, :256])  # [256, 256]
        best_match_cos = float(sim_matrix.abs().max(dim=1).values.mean().item())
        random_expected = math.sqrt(2.0 / math.pi) / math.sqrt(1344)  # ~0.0218
        mlp_diagnostics[f"layer_{check_layer}"] = {
            "direct_neuron_cosine": diag_cos,
            "greedy_matched_abs_cosine_256": best_match_cos,
            "random_orthogonal_expected": random_expected,
        }

    global_cosines = [v["cosine_similarity"] for v in per_tensor.values()]
    summary = {
        "total_parameters": sum(v["numel"] for v in per_tensor.values()),
        "total_tensors": len(per_tensor),
        "mean_parameter_cosine": sum(global_cosines) / len(global_cosines),
        "min_parameter_cosine": min(global_cosines),
        "max_parameter_cosine": max(global_cosines),
        "embedding_wte_cosine": per_tensor["wte.weight"]["cosine_similarity"],
        "final_norm_ln_f_cosine": per_tensor["ln_f.weight"]["cosine_similarity"],
        "lm_head_cosine": per_tensor["lm_head.weight"]["cosine_similarity"],
        "layer_summary": layer_summary,
        "mlp_neuron_alignment": mlp_diagnostics,
    }

    return {"summary": summary, "per_tensor": per_tensor}


def main() -> None:
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    report = compute_alignment()
    OUT_FILE.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Alignment audit written to {OUT_FILE}")
    print("\nSummary:")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
