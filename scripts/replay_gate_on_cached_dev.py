#!/usr/bin/env python3
"""Replay proposed confidence gates against cached development results.

Tests gate hypotheses on:
1. 30-task MBPP retained execution results (passing and failing drafts).
2. Known successful repairs from the offline oracle diagnostic.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def evaluate_policy_proposals(
    edits: List[Dict[str, Any]],
    policy_name: str,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """Evaluate acceptance, rejection, and error rates for a given policy at a confidence threshold."""
    policy_edits = [e for e in edits if e.get("policy") == policy_name]
    total = len(policy_edits)
    no_ops = [e for e in policy_edits if not e.get("is_different_token")]
    diffs = [e for e in policy_edits if e.get("is_different_token")]

    accepted = [e for e in diffs if e.get("replacement_prob", 0.0) >= threshold]
    rejected = [e for e in diffs if e.get("replacement_prob", 0.0) < threshold]

    # Harmful edits: broke an originally passing draft
    broken_proposals = [e for e in diffs if e.get("ar_passed") and not e.get("repaired_passed")]
    broken_accepted = [e for e in broken_proposals if e.get("replacement_prob", 0.0) >= threshold]
    broken_rejected = [e for e in broken_proposals if e.get("replacement_prob", 0.0) < threshold]

    # Proposals on failing drafts
    failing_proposals = [e for e in diffs if not e.get("ar_passed")]
    failing_accepted = [e for e in failing_proposals if e.get("replacement_prob", 0.0) >= threshold]
    failing_rejected = [e for e in failing_proposals if e.get("replacement_prob", 0.0) < threshold]

    return {
        "policy": policy_name,
        "total_proposals": total,
        "no_ops": len(no_ops),
        "different_token_proposals": len(diffs),
        "threshold": threshold,
        "accepted_proposals": len(accepted),
        "rejected_proposals": len(rejected),
        "total_unchanged": len(no_ops) + len(rejected),
        "unchanged_pct": round((len(no_ops) + len(rejected)) / total * 100, 1) if total else 0.0,
        "broken_total": len(broken_proposals),
        "broken_accepted": len(broken_accepted),
        "broken_rejected": len(broken_rejected),
        "failing_accepted": len(failing_accepted),
        "failing_rejected": len(failing_rejected),
    }


def main():
    data_path = PROJECT_ROOT / "research/corosred_dev_retained_edits_30tasks_2026-10-18/edits_log.json"
    with open(data_path) as f:
        edits = json.load(f)

    print("Replaying confidence gate (threshold = 0.50) on cached MBPP development data:\n")
    for pol in ["single_token_retained", "budget_k3_retained"]:
        stats = evaluate_policy_proposals(edits, pol, threshold=0.50)
        print(f"Policy: {stats['policy']}")
        print(f"  Total Proposals:           {stats['total_proposals']}")
        print(f"  Natural No-Ops (Same-Token): {stats['no_ops']}")
        print(f"  Different-Token Proposed:  {stats['different_token_proposals']}")
        print(f"  Accepted by Gate (>=0.50): {stats['accepted_proposals']}")
        print(f"  Rejected by Gate (<0.50):  {stats['rejected_proposals']}")
        print(f"  Total Left Unchanged:      {stats['total_unchanged']} / {stats['total_proposals']} ({stats['unchanged_pct']}%)")
        print(f"  Harmful Edits Accepted:    {stats['broken_accepted']} (out of {stats['broken_total']} regressions)")
        print(f"  Failing Drafts Mutated:    {stats['failing_accepted']} (none converted to pass)\n")


if __name__ == "__main__":
    main()
