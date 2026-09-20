"""
Re-scores existing tool-use benchmark evaluation reports with the updated
mathematical equivalence and argument normalization logic.
"""

import json
from pathlib import Path
from telos.eval.tooluse.evaluator import _compare_arguments
from telos.eval.stats import bootstrap_confidence_interval

LOGS_DIR = Path("logs")

for filename in ["eval_report_granite_mlx_tooluse_full.json", "eval_report_afm3_tooluse_full.json"]:
    filepath = LOGS_DIR / filename
    if not filepath.exists():
        continue

    with open(filepath) as f:
        data = json.load(f)

    tooluse_data = data.get("tooluse", {})
    tasks = tooluse_data.get("tasks", [])
    if not tasks:
        continue

    passed_count = 0
    cat_stats = {}
    pass_binary = []

    for t in tasks:
        cat = t.get("category", "General")
        if cat not in cat_stats:
            cat_stats[cat] = {"total": 0, "tool_correct": 0, "syntax_valid": 0, "passed": 0}

        cat_stats[cat]["total"] += 1
        is_tool = bool(t.get("tool_correct"))
        is_syntax = bool(t.get("syntax_valid"))
        if is_tool:
            cat_stats[cat]["tool_correct"] += 1
        if is_syntax:
            cat_stats[cat]["syntax_valid"] += 1

        exp_args = t.get("expected_args") or {}
        act_args = t.get("parsed_args") or {}
        is_args_correct = bool(is_tool and _compare_arguments(exp_args, act_args))
        is_passed = bool(is_tool and is_args_correct)

        t["passed"] = is_passed
        if is_passed:
            passed_count += 1
            cat_stats[cat]["passed"] += 1
            pass_binary.append(1.0)
        else:
            pass_binary.append(0.0)

    total_tasks = len(tasks)
    pass_rate_pct = round((passed_count / max(1, total_tasks)) * 100.0, 2)
    ci = bootstrap_confidence_interval(pass_binary)

    # Update category summaries
    cat_summary = {}
    for c, s in cat_stats.items():
        cnt = s["total"]
        cat_summary[c] = {
            "total": cnt,
            "tool_selection_pct": round((s["tool_correct"] / cnt) * 100.0, 1) if cnt else 0.0,
            "syntax_validity_pct": round((s["syntax_valid"] / cnt) * 100.0, 1) if cnt else 0.0,
            "pass_rate_pct": round((s["passed"] / cnt) * 100.0, 1) if cnt else 0.0,
        }

    tooluse_data["pass_rate_pct"] = pass_rate_pct
    tooluse_data["pass_rate_95ci"] = ci
    tooluse_data["categories"] = cat_summary

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    print(f"✓ Re-scored {filename}: Pass Rate = {pass_rate_pct}% (CI: {ci})")
