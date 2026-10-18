#!/usr/bin/env python3
"""Execute the predeclared COROSred experiment matrix on matched 530M models.

Evaluates 7 conditions across 112 cases (24 counterfactual pairs + 64 local code cases):
1. native_ar: Frozen AR weights, causal attention, shifted readout (baseline 1)
2. native_mdlm: Frozen MDLM weights, bidirectional attention, same-position readout (baseline 2)
3. weight_avg_50: Direct linear weight interpolation (0.5 AR + 0.5 MDLM)
4. layer_subst_half: Layers 0-7 from AR, layers 8-15 from MDLM (component substitution)
5. ar_bidirectional: Native AR with bidirectional attention (matching Qwen3 attention probe)
6. mdlm_causal: Native MDLM with causal attention (testing AR generation capability)
7. corosred_two_model: AR drafting + uncertainty routing + MDLM bidirectional repair

Outputs:
- results.jsonl: Per-case evaluation records
- summary.json: Aggregated metrics per condition
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from dclm_model import (
    DEFAULT_AR_WEIGHTS,
    DEFAULT_MDLM_WEIGHTS,
    DCLMConfig,
    DCLMModel,
    load_dclm_model,
)

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "research/matched_530m_corosred_probe_2026-10-18"
CASES_FILE = EXP_DIR / "cases.json"
TOKENIZER_FILE = (
    Path.home()
    / ".cache/huggingface/hub/models--michaelchenkj--datadecide-dclm-530m-53b/snapshots/213f7e96745ee3d397834821fe2b397019bdde98/weights/step57786/tokenizer.json"
)


def build_interpolated_model(model_ar: DCLMModel, model_mdlm: DCLMModel, alpha: float = 0.5) -> DCLMModel:
    """Linearly interpolate weights: alpha * AR + (1 - alpha) * MDLM."""
    cfg = model_ar.cfg
    merged = DCLMModel(cfg)
    state = {}
    ar_sd = model_ar.state_dict()
    md_sd = model_mdlm.state_dict()
    for k in ar_sd.keys():
        state[k] = alpha * ar_sd[k] + (1.0 - alpha) * md_sd[k]
    merged.load_state_dict(state)
    merged.eval()
    for p in merged.parameters():
        p.requires_grad = False
    return merged


def build_layer_substituted_model(model_ar: DCLMModel, model_mdlm: DCLMModel) -> DCLMModel:
    """Layers 0-7 from AR, layers 8-15 from MDLM. Embeddings from AR, head from MDLM."""
    cfg = model_ar.cfg
    hybrid = DCLMModel(cfg)
    ar_sd = model_ar.state_dict()
    md_sd = model_mdlm.state_dict()
    state = {}
    for k in ar_sd.keys():
        if k.startswith("blocks."):
            layer_idx = int(k.split(".")[1])
            state[k] = ar_sd[k] if layer_idx < 8 else md_sd[k]
        elif k in ("wte.weight",):
            state[k] = ar_sd[k]
        else:
            state[k] = md_sd[k]
    hybrid.load_state_dict(state)
    hybrid.eval()
    for p in hybrid.parameters():
        p.requires_grad = False
    return hybrid


def compute_uncertainty_metrics(probs: torch.Tensor) -> dict[str, float]:
    """Compute entropy, top-1 probability, and top-1 vs top-2 margin."""
    probs_f = probs.float()
    top2 = torch.topk(probs_f, 2)
    p1 = float(top2.values[0].item())
    p2 = float(top2.values[1].item())
    margin = p1 - p2
    # Shannon entropy
    entropy = -float(torch.sum(probs_f * torch.log(probs_f.clamp_min(1e-12))).item())
    return {
        "entropy": entropy,
        "top1_prob": p1,
        "margin": margin,
    }


def evaluate_all(cases: list[dict], tok: Tokenizer, device: str = "cpu") -> tuple[list[dict], dict]:
    print(f"Loading native AR model on {device}...")
    ar_model = load_dclm_model(DEFAULT_AR_WEIGHTS, is_ar=True, device=device)
    print(f"Loading native MDLM model on {device}...")
    mdlm_model = load_dclm_model(DEFAULT_MDLM_WEIGHTS, is_ar=False, device=device)

    print("Building weight-averaged model (alpha=0.5)...")
    weight_avg_model = build_interpolated_model(ar_model, mdlm_model, alpha=0.5).to(device)
    print("Building layer-substituted model (0-7 AR, 8-15 MDLM)...")
    layer_subst_model = build_layer_substituted_model(ar_model, mdlm_model).to(device)

    models = {
        "ar": ar_model,
        "mdlm": mdlm_model,
        "weight_avg": weight_avg_model,
        "layer_subst": layer_subst_model,
    }

    conditions = [
        "native_ar",
        "native_mdlm",
        "weight_avg_50",
        "layer_subst_half",
        "ar_bidirectional",
        "mdlm_causal",
        "corosred_two_model",
    ]

    records = []
    print(f"\nBeginning evaluation across {len(cases)} cases and {len(conditions)} conditions...")
    t0 = time.monotonic()

    for idx, c in enumerate(cases):
        ids = list(c["ids"])
        hole = c["hole"]
        target = c["target"]
        alt = c.get("alternative")
        length = len(ids)

        for cond in conditions:
            pred = None
            logp_target = None
            candidate_margin = None
            uncertainty = {}

            if cond == "native_ar":
                # Shifted readout: logits at hole - 1 predict token at hole
                input_t = torch.tensor([ids], dtype=torch.long, device=device)
                with torch.no_grad():
                    logits = models["ar"](input_t, is_causal=True)[0, hole - 1, :50280]
                probs = F.softmax(logits.float(), dim=-1)
                pred = int(probs.argmax().item())
                logp_target = float(torch.log(probs[target].clamp_min(1e-12)).item())
                if alt is not None:
                    candidate_margin = float((logits[target] - logits[alt]).item())
                uncertainty = compute_uncertainty_metrics(probs)

            elif cond == "native_mdlm":
                # Same-position readout: replace hole with mask_id (50280), bidirectional attention
                mids = list(ids)
                mids[hole] = 50280  # mask_id
                input_t = torch.tensor([mids], dtype=torch.long, device=device)
                with torch.no_grad():
                    logits = models["mdlm"](input_t, is_causal=False)[0, hole, :50280]
                probs = F.softmax(logits.float(), dim=-1)
                pred = int(probs.argmax().item())
                logp_target = float(torch.log(probs[target].clamp_min(1e-12)).item())
                if alt is not None:
                    candidate_margin = float((logits[target] - logits[alt]).item())
                uncertainty = compute_uncertainty_metrics(probs)

            elif cond == "weight_avg_50":
                # Average weights tested with bidirectional masked input
                mids = list(ids)
                mids[hole] = 50280
                input_t = torch.tensor([mids], dtype=torch.long, device=device)
                with torch.no_grad():
                    logits = models["weight_avg"](input_t, is_causal=False)[0, hole, :50280]
                probs = F.softmax(logits.float(), dim=-1)
                pred = int(probs.argmax().item())
                logp_target = float(torch.log(probs[target].clamp_min(1e-12)).item())
                if alt is not None:
                    candidate_margin = float((logits[target] - logits[alt]).item())
                uncertainty = compute_uncertainty_metrics(probs)

            elif cond == "layer_subst_half":
                # Hybrid layers tested with bidirectional masked input
                mids = list(ids)
                mids[hole] = 50280
                input_t = torch.tensor([mids], dtype=torch.long, device=device)
                with torch.no_grad():
                    logits = models["layer_subst"](input_t, is_causal=False)[0, hole, :50280]
                probs = F.softmax(logits.float(), dim=-1)
                pred = int(probs.argmax().item())
                logp_target = float(torch.log(probs[target].clamp_min(1e-12)).item())
                if alt is not None:
                    candidate_margin = float((logits[target] - logits[alt]).item())
                uncertainty = compute_uncertainty_metrics(probs)

            elif cond == "ar_bidirectional":
                # AR backbone run with full bidirectional attention and zero placeholder
                input_t = torch.tensor([ids], dtype=torch.long, device=device)
                emb = models["ar"].wte(input_t)
                emb[0, hole, :] = 0.0  # Zero embedding placeholder
                with torch.no_grad():
                    logits = models["ar"](input_embeddings=emb, is_causal=False)[0, hole - 1, :50280]
                probs = F.softmax(logits.float(), dim=-1)
                pred = int(probs.argmax().item())
                logp_target = float(torch.log(probs[target].clamp_min(1e-12)).item())
                if alt is not None:
                    candidate_margin = float((logits[target] - logits[alt]).item())
                uncertainty = compute_uncertainty_metrics(probs)

            elif cond == "mdlm_causal":
                # MDLM backbone run with causal attention and shifted readout
                input_t = torch.tensor([ids], dtype=torch.long, device=device)
                with torch.no_grad():
                    logits = models["mdlm"](input_t, is_causal=True)[0, hole - 1, :50280]
                probs = F.softmax(logits.float(), dim=-1)
                pred = int(probs.argmax().item())
                logp_target = float(torch.log(probs[target].clamp_min(1e-12)).item())
                if alt is not None:
                    candidate_margin = float((logits[target] - logits[alt]).item())
                uncertainty = compute_uncertainty_metrics(probs)

            elif cond == "corosred_two_model":
                # Two-model pipeline: AR drafts causal tokens + evaluates uncertainty;
                # MDLM re-diffuses the hole with bidirectional right context.
                input_t = torch.tensor([ids], dtype=torch.long, device=device)
                with torch.no_grad():
                    ar_logits = models["ar"](input_t, is_causal=True)[0, hole - 1, :50280]
                ar_probs = F.softmax(ar_logits.float(), dim=-1)
                ar_pred = int(ar_probs.argmax().item())
                ar_uncertainty = compute_uncertainty_metrics(ar_probs)

                # MDLM repair pass:
                mids = list(ids)
                mids[hole] = 50280  # mask_id
                m_input = torch.tensor([mids], dtype=torch.long, device=device)
                with torch.no_grad():
                    mdlm_logits = models["mdlm"](m_input, is_causal=False)[0, hole, :50280]
                mdlm_probs = F.softmax(mdlm_logits.float(), dim=-1)
                mdlm_pred = int(mdlm_probs.argmax().item())
                mdlm_uncertainty = compute_uncertainty_metrics(mdlm_probs)

                # For confidence-routing: if AR margin is low (< 0.5) or entropy is high (> 1.0),
                # route to MDLM repair.
                is_routed = ar_uncertainty["margin"] < 0.5 or ar_uncertainty["entropy"] > 1.0
                pred = mdlm_pred if is_routed else ar_pred
                chosen_logits = mdlm_logits if is_routed else ar_logits
                probs = mdlm_probs if is_routed else ar_probs
                logp_target = float(torch.log(probs[target].clamp_min(1e-12)).item())
                if alt is not None:
                    candidate_margin = float((chosen_logits[target] - chosen_logits[alt]).item())
                uncertainty = {
                    "routed": is_routed,
                    "ar_margin": ar_uncertainty["margin"],
                    "ar_entropy": ar_uncertainty["entropy"],
                    "mdlm_margin": mdlm_uncertainty["margin"],
                }

            row = {
                "id": c["id"],
                "group": c["group"],
                "pair": c.get("pair"),
                "condition": cond,
                "target": target,
                "target_text": c["target_text"],
                "prediction": pred,
                "prediction_text": tok.decode([pred]),
                "correct": bool(pred == target),
                "target_logp": logp_target,
                "target_nll": -logp_target if logp_target is not None else None,
                "candidate_margin": candidate_margin,
                "uncertainty": uncertainty,
            }
            records.append(row)

        if (idx + 1) % 16 == 0 or (idx + 1) == len(cases):
            dt = time.monotonic() - t0
            print(f"Evaluated {idx+1}/{len(cases)} cases ({len(records)} rows) in {dt:.1f}s", flush=True)

    # Compute summaries
    summary = {}
    for cond in conditions:
        cond_rows = [r for r in records if r["condition"] == cond]
        local_rows = [r for r in cond_rows if r["group"] == "local_code"]
        paired_rows = [r for r in cond_rows if r["group"].startswith("paired_")]

        # Local metrics
        local_top1 = sum(r["correct"] for r in local_rows) / max(1, len(local_rows))
        local_mean_nll = statistics.mean(r["target_nll"] for r in local_rows)

        # Paired metrics
        pairs: dict[str, list[dict]] = {}
        for r in paired_rows:
            pairs.setdefault(r["pair"], []).append(r)
        
        both_top1 = sum(all(p["correct"] for p in pair_members) for pair_members in pairs.values())
        both_prefer_required = sum(
            all(p["candidate_margin"] is not None and p["candidate_margin"] > 0 for p in pair_members)
            for pair_members in pairs.values()
        )
        prediction_changes = sum(
            pair_members[0]["prediction"] != pair_members[1]["prediction"]
            for pair_members in pairs.values()
            if len(pair_members) == 2
        )

        summary[cond] = {
            "total_evaluated": len(cond_rows),
            "local_code_top1": f"{sum(r['correct'] for r in local_rows)}/{len(local_rows)} ({local_top1*100:.1f}%)",
            "local_code_mean_nll": round(local_mean_nll, 4),
            "counterfactual_pairs_both_top1": f"{both_top1}/{len(pairs)}",
            "counterfactual_pairs_both_prefer_required": f"{both_prefer_required}/{len(pairs)}",
            "counterfactual_prediction_changes": f"{prediction_changes}/{len(pairs)}",
        }

    return records, summary


def main() -> None:
    cases = json.loads(CASES_FILE.read_text())
    tok = Tokenizer.from_file(str(TOKENIZER_FILE))
    # Use MPS if available, else CPU
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Running evaluation on device: {device}")

    records, summary = evaluate_all(cases, tok, device=device)

    # Save results
    results_path = EXP_DIR / "results.jsonl"
    with results_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(records)} case records -> {results_path}")

    summary_path = EXP_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved experiment summary -> {summary_path}")

    print("\n=== EXPERIMENT MATRIX RESULTS ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
