#!/usr/bin/env python3
"""Run COROSred experiment matrix on Qwen3-0.6B AR and its converted MDLM derivative.

Evaluates:
1. native_ar: Frozen Qwen3-0.6B, causal attention, shifted readout.
   Includes strict causal suffix-independence control.
2. native_mdlm: Frozen Qwen3-0.6B-diffusion-mdlm-v0.1, bidirectional attention, mask_token 151669.
3. weight_avg_50: Direct weight interpolation (0.5 AR + 0.5 DIFF).
4. weight_interp_75: Weight interpolation (0.75 AR + 0.25 DIFF).
5. weight_interp_25: Weight interpolation (0.25 AR + 0.75 DIFF).
6. layer_subst_half: Layers 0-13 from AR, layers 14-27 from Diffusion.
7. corosred_two_model: Confidence-routed pipeline (AR draft + uncertainty routing + MDLM bidirectional repair).

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

from qwen3_06b_model import (
    Qwen3_06B_Config,
    Qwen3_06B_Model,
    load_qwen3_06b_model,
)

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "research/qwen3_06b_corosred_probe_2026-10-18"
CASES_FILE = ROOT / "research/qwen3_attention_probe_2026-10-18/full/cases.json"
TOKENIZER_FILE = (
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/tokenizer.json"
)

AR_WEIGHTS = (
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/model.safetensors"
)
DIFF_WEIGHTS = (
    Path.home()
    / ".cache/huggingface/hub/models--dllm-hub--Qwen3-0.6B-diffusion-mdlm-v0.1/snapshots/c8d24a3f4adaeef46881b450e1bf7d1005203bd7/model.safetensors"
)


def build_interpolated_model(model_ar: Qwen3_06B_Model, model_diff: Qwen3_06B_Model, alpha: float) -> Qwen3_06B_Model:
    cfg = model_ar.cfg
    merged = Qwen3_06B_Model(cfg)
    ar_sd = model_ar.state_dict()
    diff_sd = model_diff.state_dict()
    state = {}
    for k in ar_sd.keys():
        state[k] = alpha * ar_sd[k] + (1.0 - alpha) * diff_sd[k]
    merged.load_state_dict(state)
    merged.to(dtype=ar_sd["embed_tokens.weight"].dtype)
    merged.eval()
    for p in merged.parameters():
        p.requires_grad = False
    return merged


def build_layer_substituted_model(model_ar: Qwen3_06B_Model, model_diff: Qwen3_06B_Model, split_layer: int = 14) -> Qwen3_06B_Model:
    cfg = model_ar.cfg
    hybrid = Qwen3_06B_Model(cfg)
    ar_sd = model_ar.state_dict()
    diff_sd = model_diff.state_dict()
    state = {}
    for k in ar_sd.keys():
        if k.startswith("layers."):
            layer_idx = int(k.split(".")[1])
            state[k] = ar_sd[k] if layer_idx < split_layer else diff_sd[k]
        elif k == "embed_tokens.weight":
            state[k] = ar_sd[k]
        else:
            state[k] = diff_sd[k]
    hybrid.load_state_dict(state)
    hybrid.to(dtype=ar_sd["embed_tokens.weight"].dtype)
    hybrid.eval()
    for p in hybrid.parameters():
        p.requires_grad = False
    return hybrid


def compute_uncertainty(probs: torch.Tensor) -> dict[str, float]:
    probs_f = probs.float()
    top2 = torch.topk(probs_f, 2)
    p1 = float(top2.values[0].item())
    p2 = float(top2.values[1].item())
    entropy = -float(torch.sum(probs_f * torch.log(probs_f.clamp_min(1e-12))).item())
    return {
        "entropy": entropy,
        "top1_prob": p1,
        "margin": p1 - p2,
    }


def evaluate_condition(
    cond: str,
    cases: list[dict],
    tok: Tokenizer,
    device: str,
    model: Optional[Qwen3_06B_Model] = None,
    ar_model: Optional[Qwen3_06B_Model] = None,
    diff_model: Optional[Qwen3_06B_Model] = None,
) -> list[dict]:
    records = []
    mask_token_id = 151669  # <|mask|>

    for c in cases:
        ids = list(c["ids"])
        hole = c["hole"]
        target = c["target"]
        alt = c.get("alternative")

        ar_input_ids = list(ids)
        ar_input_ids[hole] = 62

        pred = None
        logp_target = None
        candidate_margin = None
        uncertainty = {}

        if cond == "native_ar":
            assert model is not None
            t_in = torch.tensor([ar_input_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                logits = model(t_in, is_causal=True)[0, hole - 1, :]
            probs = F.softmax(logits.float(), dim=-1)
            pred = int(probs.argmax().item())
            logp_target = float(torch.log(probs[target].clamp_min(1e-12)).item())
            if alt is not None:
                candidate_margin = float((logits[target] - logits[alt]).item())
            uncertainty = compute_uncertainty(probs)

        elif cond in ("native_mdlm", "weight_avg_50", "weight_interp_75", "weight_interp_25", "layer_subst_half"):
            assert model is not None
            m_in = list(ids)
            m_in[hole] = mask_token_id
            t_in = torch.tensor([m_in], dtype=torch.long, device=device)
            with torch.no_grad():
                logits = model(t_in, is_causal=False)[0, hole, :]
            probs = F.softmax(logits.float(), dim=-1)
            pred = int(probs.argmax().item())
            logp_target = float(torch.log(probs[target].clamp_min(1e-12)).item())
            if alt is not None:
                candidate_margin = float((logits[target] - logits[alt]).item())
            uncertainty = compute_uncertainty(probs)

        elif cond == "corosred_two_model":
            assert ar_model is not None and diff_model is not None
            # 1. AR draft
            t_in = torch.tensor([ar_input_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                ar_logits = ar_model(t_in, is_causal=True)[0, hole - 1, :]
            ar_probs = F.softmax(ar_logits.float(), dim=-1)
            ar_pred = int(ar_probs.argmax().item())
            ar_unc = compute_uncertainty(ar_probs)

            # 2. Diffusion repair
            m_in = list(ids)
            m_in[hole] = mask_token_id
            t_m_in = torch.tensor([m_in], dtype=torch.long, device=device)
            with torch.no_grad():
                diff_logits = diff_model(t_m_in, is_causal=False)[0, hole, :]
            diff_probs = F.softmax(diff_logits.float(), dim=-1)
            diff_pred = int(diff_probs.argmax().item())
            diff_unc = compute_uncertainty(diff_probs)

            # Confidence routing heuristic: route if margin < 0.5 or entropy > 1.0
            is_routed = ar_unc["margin"] < 0.5 or ar_unc["entropy"] > 1.0
            pred = diff_pred if is_routed else ar_pred
            chosen_logits = diff_logits if is_routed else ar_logits
            probs = diff_probs if is_routed else ar_probs

            logp_target = float(torch.log(probs[target].clamp_min(1e-12)).item())
            if alt is not None:
                candidate_margin = float((chosen_logits[target] - chosen_logits[alt]).item())
            uncertainty = {
                "routed": is_routed,
                "ar_margin": ar_unc["margin"],
                "ar_entropy": ar_unc["entropy"],
                "diff_margin": diff_unc["margin"],
                "diff_entropy": diff_unc["entropy"],
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

    return records


def evaluate_all(cases: list[dict], tok: Tokenizer, device: str = "cpu") -> tuple[list[dict], dict]:
    print(f"Loading native AR model (Qwen3-0.6B) on {device}...")
    ar_model = load_qwen3_06b_model(AR_WEIGHTS, device=device)
    print(f"Loading native Diffusion model (dLLM Qwen3-0.6B MDLM) on {device}...")
    diff_model = load_qwen3_06b_model(DIFF_WEIGHTS, device=device)

    all_records = []
    conditions = [
        "native_ar",
        "native_mdlm",
        "corosred_two_model",
        "weight_avg_50",
        "weight_interp_75",
        "weight_interp_25",
        "layer_subst_half",
    ]

    for cond in conditions:
        t0 = time.monotonic()
        print(f"\n--- Evaluating condition: {cond} ---")
        if cond == "native_ar":
            records = evaluate_condition(cond, cases, tok, device, model=ar_model)
        elif cond == "native_mdlm":
            records = evaluate_condition(cond, cases, tok, device, model=diff_model)
        elif cond == "corosred_two_model":
            records = evaluate_condition(cond, cases, tok, device, ar_model=ar_model, diff_model=diff_model)
        elif cond == "weight_avg_50":
            interp = build_interpolated_model(ar_model, diff_model, 0.5).to(device)
            records = evaluate_condition(cond, cases, tok, device, model=interp)
            del interp
        elif cond == "weight_interp_75":
            interp = build_interpolated_model(ar_model, diff_model, 0.75).to(device)
            records = evaluate_condition(cond, cases, tok, device, model=interp)
            del interp
        elif cond == "weight_interp_25":
            interp = build_interpolated_model(ar_model, diff_model, 0.25).to(device)
            records = evaluate_condition(cond, cases, tok, device, model=interp)
            del interp
        elif cond == "layer_subst_half":
            hybrid = build_layer_substituted_model(ar_model, diff_model, 14).to(device)
            records = evaluate_condition(cond, cases, tok, device, model=hybrid)
            del hybrid

        if device == "mps":
            torch.mps.empty_cache()

        dt = time.monotonic() - t0
        local_corr = sum(r["correct"] for r in records if r["group"] == "local_code")
        print(f"Condition {cond} finished in {dt:.1f}s | Local Code: {local_corr}/64")
        all_records.extend(records)

    records = all_records

    # Compute summaries
    summary = {}
    for cond in conditions:
        cond_rows = [r for r in records if r["condition"] == cond]
        local_rows = [r for r in cond_rows if r["group"] == "local_code"]
        paired_rows = [r for r in cond_rows if r["group"].startswith("paired_")]

        local_top1 = sum(r["correct"] for r in local_rows) / max(1, len(local_rows))
        local_mean_nll = statistics.mean(r["target_nll"] for r in local_rows)

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

        # Check causal suffix independence on native_ar
        max_causal_delta = 0.0
        if cond == "native_ar":
            deltas = []
            for p_members in pairs.values():
                if len(p_members) == 2:
                    d = abs(p_members[0]["target_logp"] - p_members[1]["target_logp"])
                    deltas.append(d)
            # Both targets differ, so target_logp differ, but candidate_margin or prefix distribution
            # at hole-1 is identical. We verify:
            pred_changes = sum(p[0]["prediction"] != p[1]["prediction"] for p in pairs.values())
            assert pred_changes == 0, f"Causal suffix independence violated: {pred_changes} prediction changes across identical prefixes!"

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
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    cases = json.loads(CASES_FILE.read_text())
    tok = Tokenizer.from_file(str(TOKENIZER_FILE))
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Running evaluation on device: {device}")

    records, summary = evaluate_all(cases, tok, device=device)

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
