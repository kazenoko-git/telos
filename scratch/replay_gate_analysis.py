#!/usr/bin/env python3
"""Comprehensive Replay Analysis of the Proposed Confidence Gate on Cached Development Results."""
import json
from pathlib import Path

def main():
    print("=" * 80)
    print("REPLAYING PROPOSED CONFIDENCE GATE AGAINST ALL CACHED DEV RESULTS")
    print("=" * 80)

    # 1. 30-task retained edits results
    path_30tasks = Path("research/corosred_dev_retained_edits_30tasks_2026-10-18/per_task_results.json")
    path_edits_log = Path("research/corosred_dev_retained_edits_30tasks_2026-10-18/edits_log.json")
    
    with open(path_30tasks) as f:
        tasks_30 = json.load(f)
    with open(path_edits_log) as f:
        edits_30 = json.load(f)

    print(f"Loaded 30-task dataset: {len(tasks_30)} tasks, {len(edits_30)} logged proposals.\n")

    for pol in ["single_token_retained", "budget_k3_retained"]:
        p_edits = [e for e in edits_30 if e["policy"] == pol]
        total = len(p_edits)
        no_ops = [e for e in p_edits if not e["is_different_token"]]
        diffs = [e for e in p_edits if e["is_different_token"]]
        
        # Check threshold 0.50
        accepted_50 = [e for e in diffs if e["replacement_prob"] >= 0.50]
        rejected_50 = [e for e in diffs if e["replacement_prob"] < 0.50]
        
        # Breaking edits: originally passing draft but repaired_passed is False
        broken_all = [e for e in diffs if e["ar_passed"] and not e["repaired_passed"]]
        broken_accepted_50 = [e for e in broken_all if e["replacement_prob"] >= 0.50]
        broken_rejected_50 = [e for e in broken_all if e["replacement_prob"] < 0.50]

        # Edits on failing drafts that remained failing
        failing_diffs = [e for e in diffs if not e["ar_passed"]]
        failing_accepted_50 = [e for e in failing_diffs if e["replacement_prob"] >= 0.50]
        failing_rejected_50 = [e for e in failing_diffs if e["replacement_prob"] < 0.50]
        
        print(f"--- Policy: {pol} ---")
        print(f"Total proposals evaluated: {total}")
        print(f"  Natural No-Ops (MDLM reproduced original token): {len(no_ops)}")
        print(f"  Effective proposals (different token proposed): {len(diffs)}")
        print(f"At threshold >= 0.50:")
        print(f"  Proposals accepted: {len(accepted_50)}")
        print(f"  Proposals rejected: {len(rejected_50)}")
        print(f"  Total left unchanged (no-ops + rejected): {len(no_ops) + len(rejected_50)} / {total} ({(len(no_ops)+len(rejected_50))/total*100:.1f}%)")
        print(f"Regressions on originally passing drafts:")
        print(f"  Total harmful proposals: {len(broken_all)}")
        print(f"  Harmful proposals accepted by >= 0.50: {len(broken_accepted_50)}")
        print(f"  Harmful proposals rejected by >= 0.50: {len(broken_rejected_50)}")
        for b in broken_all:
            tid = b["task_id"]
            pos = b["position"]
            orig = repr(b["original_token_str"])
            rep = repr(b["replacement_token_str"])
            prob = b["replacement_prob"]
            margin = b["original_margin"]
            print(f"    * {tid} pos {pos}: {orig} -> {rep} (prob={prob:.4f}, margin={margin:.4f})")
        print(f"Proposals on failing drafts:")
        print(f"  Total failing draft effective proposals: {len(failing_diffs)}")
        print(f"  Accepted by >= 0.50 (all remained failing): {len(failing_accepted_50)}")
        print(f"  Rejected by >= 0.50: {len(failing_rejected_50)}")
        for f_e in failing_accepted_50:
            tid = f_e["task_id"]
            pos = f_e["position"]
            orig = repr(f_e["original_token_str"])
            rep = repr(f_e["replacement_token_str"])
            prob = f_e["replacement_prob"]
            margin = f_e["original_margin"]
            print(f"    * High-conf non-repair in {tid} pos {pos}: {orig} -> {rep} (prob={prob:.4f}, margin={margin:.4f})")
        print()

    # 2. Known Successful Repairs (Oracle Diagnostic)
    path_oracle = Path("research/corosred_oracle_diagnostic_2026-10-18/results.json")
    with open(path_oracle) as f:
        oracle_res = json.load(f)
    successful_repairs = oracle_res["all_successful_repairs"]
    print("=" * 80)
    print("REPLAY AGAINST KNOWN SUCCESSFUL REPAIRS (Oracle Diagnostic)")
    print("=" * 80)
    for r in successful_repairs:
        tid = r["task_id"]
        pos = r["position"]
        orig = repr(r["original_token_str"])
        rep = repr(r["replacement_token_str"])
        prob = r["replacement_prob"]
        m = r["ar_margin"]
        m_rk = r["margin_rank"]
        h_rk = r["entropy_rank"]
        print(f"Repair: Task {tid} pos {pos}: {orig} -> {rep}")
        print(f"  MDLM Replacement Confidence: {prob:.4f}")
        print(f"  AR Margin: {m:.4f} (Margin Rank: {m_rk}, Entropy Rank: {h_rk})")
        
        # Test 1: Confidence floor >= 0.50
        pass_conf_floor = prob >= 0.50
        print(f"  Gate check [MDLM prob >= 0.50]: {'ACCEPTED' if pass_conf_floor else 'REJECTED'}")
        
        # Test 2: MDLM Confidence > AR Confidence
        # For MBPP/58: AR top-1 prob was ~0.45, margin=0.0355
        # For MBPP/3: AR top-1 margin was 0.8954 -> AR prob >= 0.8954
        if tid == "MBPP/58":
            ar_conf = 0.45  # approximate, top-1 vs top-2 diff is 0.0355
            pass_rel = prob > ar_conf
            print(f"  Gate check [MDLM prob ({prob:.3f}) > AR prob (~{ar_conf:.3f})]: {'ACCEPTED' if pass_rel else 'REJECTED'}")
        elif tid == "MBPP/3":
            ar_conf = 0.8954 # lower bound on AR top-1 prob
            pass_rel = prob > ar_conf
            print(f"  Gate check [MDLM prob ({prob:.3f}) > AR prob (>= {ar_conf:.3f})]: {'ACCEPTED' if pass_rel else 'REJECTED'}")
            
        # Test 3: Localization (Rank #1)
        rank_int = int(m_rk.split("/")[0])
        pass_loc = (rank_int == 1)
        print(f"  Gate check [Localization Rank == 1]: {'ACCEPTED' if pass_loc else 'REJECTED (Rank=' + str(rank_int) + ')'}")
        
        overall_accepted = pass_conf_floor and pass_rel and pass_loc
        print(f"  Overall Gate Status: {'ACCEPTED' if overall_accepted else 'REJECTED'}\n")

if __name__ == "__main__":
    main()
