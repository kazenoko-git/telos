"""
Infill Masking Ratio Scaling Experiment: 20%, 30%, 50%, 80%
Investigates whether repetition/looping and degeneration creep back in
as the masked span expands toward full free generation.
"""

import json
import torch
import numpy as np
from pathlib import Path
from telos.eval.runner import load_model_from_checkpoint
from telos.data.tokenizer import load_tokenizer
from telos.eval.syntax import check_ast_validity

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def run_experiment():
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    tok = load_tokenizer(str(PROJECT_ROOT / "configs" / "tokenizer_mac.json"))

    with open(PROJECT_ROOT / "evals" / "benchmarks" / "private_unseen_suite.json") as f:
        bench = json.load(f)

    ckpt_path = PROJECT_ROOT / "checkpoints/corosred/unified/100m_5b/checkpoint_final.pt"
    print(f"Loading 100M COROSred from {ckpt_path} on {device}...")
    model, backend, _ = load_model_from_checkpoint(ckpt_path)
    model = model.to(device)
    model.eval()

    mask_token_id = 1
    ratios = [0.20, 0.30, 0.50, 0.80]
    num_tasks = 40

    print("=" * 85)
    print(f"  COROSRED 100M INFILL RATIO SCALING EXPERIMENT ({num_tasks} TASKS)")
    print("  Testing Mask Ratios: 20%, 30%, 50%, 80% (Contiguous Span & Random Masking)")
    print("=" * 85)

    results_by_ratio = {r: {"total": 0, "exact_match": 0, "token_acc": [], "loops": 0, "ast_valid": 0} for r in ratios}
    qualitative_samples = {r: [] for r in ratios}

    for task_idx in range(num_tasks):
        item = bench[task_idx]
        prompt = item["prompt"]
        solution = item["ground_truth_solution"]
        full_code = prompt + solution

        full_ids = tok.encode(full_code).ids
        prompt_ids = tok.encode(prompt).ids
        prompt_len = len(prompt_ids)
        body_len = len(full_ids) - prompt_len

        if body_len < 6:
            continue

        for r in ratios:
            # Mask out r fraction of body tokens (contiguous span in the middle of function body)
            num_mask = max(1, int(round(body_len * r)))
            mask_start = prompt_len + max(0, (body_len - num_mask) // 2)
            mask_end = min(len(full_ids), mask_start + num_mask)

            input_ids = list(full_ids)
            target_ids = input_ids[mask_start:mask_end]

            for pos in range(mask_start, mask_end):
                input_ids[pos] = mask_token_id

            x = torch.tensor([input_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                logits = model(x)

            pred_tokens = torch.argmax(logits[0, mask_start:mask_end], dim=-1).cpu().numpy().tolist()

            # Check accuracy
            correct_tokens = sum(p == t for p, t in zip(pred_tokens, target_ids))
            acc = correct_tokens / max(1, len(target_ids))
            is_exact = (pred_tokens == target_ids)

            # Reconstruct code
            reconstructed_ids = list(input_ids)
            reconstructed_ids[mask_start:mask_end] = pred_tokens
            reconstructed_text = tok.decode(reconstructed_ids)

            # Check AST validity
            is_valid, _ = check_ast_validity(reconstructed_text)

            # Check repetition / looping
            infilled_text = tok.decode(pred_tokens)
            lines = [l.strip() for l in infilled_text.splitlines() if l.strip()]
            has_loop = len(lines) >= 2 and len(set(lines)) < len(lines)
            # Also check token repetition (e.g. same 3 tokens repeating 3+ times)
            tokens_str = [str(t) for t in pred_tokens]
            if len(tokens_str) >= 6:
                for window in range(2, 6):
                    chunks = [tuple(pred_tokens[i:i+window]) for i in range(0, len(pred_tokens) - window + 1, window)]
                    if len(chunks) >= 3 and len(set(chunks)) <= len(chunks) // 2:
                        has_loop = True
                        break

            results_by_ratio[r]["total"] += 1
            if is_exact:
                results_by_ratio[r]["exact_match"] += 1
            results_by_ratio[r]["token_acc"].append(acc)
            if is_valid:
                results_by_ratio[r]["ast_valid"] += 1
            if has_loop:
                results_by_ratio[r]["loops"] += 1

            if task_idx < 3:
                qualitative_samples[r].append({
                    "id": item["id"],
                    "name": item["name"],
                    "mask_len": num_mask,
                    "body_len": body_len,
                    "target": tok.decode(target_ids),
                    "pred": infilled_text,
                    "has_loop": has_loop,
                })

    print("\n" + "-" * 85)
    print(f"{'Mask Ratio':<12} | {'Token Acc (%)':<15} | {'Exact Match (%)':<16} | {'AST Valid (%)':<15} | {'Loop / Rep Rate (%)'}")
    print("-" * 85)
    for r in ratios:
        tot = results_by_ratio[r]["total"]
        avg_acc = np.mean(results_by_ratio[r]["token_acc"]) * 100.0 if tot > 0 else 0.0
        em_pct = (results_by_ratio[r]["exact_match"] / max(1, tot)) * 100.0
        ast_pct = (results_by_ratio[r]["ast_valid"] / max(1, tot)) * 100.0
        loop_pct = (results_by_ratio[r]["loops"] / max(1, tot)) * 100.0
        print(f"{int(r*100)}% Mask      | {avg_acc:6.1f}%          | {em_pct:6.1f}%           | {ast_pct:6.1f}%          | {loop_pct:6.1f}% ({results_by_ratio[r]['loops']}/{tot})")
    print("-" * 85)

    print("\n" + "=" * 85)
    print("  QUALITATIVE INFILLING SAMPLES ACROSS MASK RATIOS (TASK 0 & TASK 1)")
    print("=" * 85)
    for r in ratios:
        print(f"\n>>> [MASK RATIO: {int(r*100)}%]")
        for sample in qualitative_samples[r][:2]:
            print(f"  Task: {sample['id']} ({sample['name']}) | Masked: {sample['mask_len']}/{sample['body_len']} tokens")
            print(f"    Target:  {repr(sample['target'][:60])}")
            print(f"    Infill:  {repr(sample['pred'][:60])}")
            print(f"    Looped:  {sample['has_loop']}")

if __name__ == "__main__":
    run_experiment()
