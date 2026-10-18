#!/usr/bin/env python3
"""Audit parameter alignment between Qwen3-0.6B (AR) and Qwen3-0.6B-diffusion-mdlm-v0.1.

Measures:
1. Exact key match and tensor shapes across all layers.
2. Layer-by-layer cosine similarity cos(W_AR, W_DIFF).
3. Relative Frobenius shift ||W_DIFF - W_AR|| / ||W_AR||.
4. Identifies which components remained stable vs drifted during MDLM diffusion fine-tuning.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "research/qwen3_06b_corosred_probe_2026-10-18"
OUT_FILE = OUT_DIR / "weight_alignment.json"

AR_WEIGHTS = (
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/model.safetensors"
)
DIFF_WEIGHTS = (
    Path.home()
    / ".cache/huggingface/hub/models--dllm-hub--Qwen3-0.6B-diffusion-mdlm-v0.1/snapshots/c8d24a3f4adaeef46881b450e1bf7d1005203bd7/model.safetensors"
)


def load_tensor_dict(path: Path) -> dict[str, torch.Tensor]:
    tensors = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k).float()
    return tensors


def audit_alignment() -> dict[str, Any]:
    print("Loading AR weights...")
    ar_weights = load_tensor_dict(AR_WEIGHTS)
    print("Loading Diffusion weights...")
    diff_weights = load_tensor_dict(DIFF_WEIGHTS)

    assert set(ar_weights.keys()) == set(diff_weights.keys()), f"Key mismatch: {set(ar_weights.keys()) ^ set(diff_weights.keys())}"

    per_tensor = {}
    component_cosines: dict[str, list[float]] = {}
    layer_cosines: dict[int, list[float]] = {i: [] for i in range(28)}

    for k in sorted(ar_weights.keys()):
        w_ar = ar_weights[k]
        w_diff = diff_weights[k]
        assert w_ar.shape == w_diff.shape, f"Shape mismatch for {k}: {w_ar.shape} vs {w_diff.shape}"

        w_ar_64 = w_ar.double()
        w_diff_64 = w_diff.double()
        norm_ar = float(torch.linalg.norm(w_ar_64).item())
        norm_diff = float(torch.linalg.norm(w_diff_64).item())
        dot = float(torch.sum(w_ar_64 * w_diff_64).item())
        cos = dot / max(1e-12, norm_ar * norm_diff)
        diff_norm = float(torch.linalg.norm(w_diff_64 - w_ar_64).item())
        rel_diff = diff_norm / max(1e-12, norm_ar)

        per_tensor[k] = {
            "shape": list(w_ar.shape),
            "numel": w_ar.numel(),
            "norm_ar": norm_ar,
            "norm_diff": norm_diff,
            "cosine_similarity": cos,
            "relative_frobenius_shift": rel_diff,
        }

        # Component breakdown
        comp_name = k.split(".")[-2] if "layers" in k else k
        component_cosines.setdefault(comp_name, []).append(cos)

        if "layers." in k:
            layer_idx = int(k.split("layers.")[1].split(".")[0])
            layer_cosines[layer_idx].append(cos)

    # Summarize per layer
    layer_summary = {}
    for layer_idx, cos_list in layer_cosines.items():
        layer_summary[f"layer_{layer_idx}"] = {
            "mean_cosine": sum(cos_list) / len(cos_list),
            "min_cosine": min(cos_list),
            "max_cosine": max(cos_list),
        }

    # Summarize per component
    comp_summary = {}
    for comp, cos_list in component_cosines.items():
        comp_summary[comp] = {
            "mean_cosine": sum(cos_list) / len(cos_list),
            "min_cosine": min(cos_list),
            "max_cosine": max(cos_list),
        }

    all_cosines = [v["cosine_similarity"] for v in per_tensor.values()]
    all_shifts = [v["relative_frobenius_shift"] for v in per_tensor.values()]

    summary = {
        "model_pair": {
            "ar": "Qwen/Qwen3-0.6B",
            "diffusion": "dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1",
        },
        "total_parameters": sum(v["numel"] for v in per_tensor.values()),
        "total_tensors": len(per_tensor),
        "mean_parameter_cosine": sum(all_cosines) / len(all_cosines),
        "min_parameter_cosine": min(all_cosines),
        "max_parameter_cosine": max(all_cosines),
        "mean_relative_shift": sum(all_shifts) / len(all_shifts),
        "embedding_cosine": per_tensor["model.embed_tokens.weight"]["cosine_similarity"],
        "embedding_relative_shift": per_tensor["model.embed_tokens.weight"]["relative_frobenius_shift"],
        "component_summary": comp_summary,
        "layer_summary": layer_summary,
    }

    return {"summary": summary, "per_tensor": per_tensor}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = audit_alignment()
    OUT_FILE.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Audit saved to {OUT_FILE}")
    print("\n=== SUMMARY ===")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
