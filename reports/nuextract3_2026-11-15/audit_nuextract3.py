"""Recompute and independently audit NuExtract3 evaluation tracks offline.

Verifies:
1. Suite and frozen task plans hashes match expected gold.
2. Every record has a unique, valid case ID in the frozen 512 suite.
3. Extraction exactness and groundedness match recomputed values from raw documents.
4. Deterministic pipeline executor results match recomputed execution from parsed fields.
5. Direct QA answers match recomputed outcome classification from raw responses.
6. Asserts zero prompt leaks of expected answers or private reference fields.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CONTRACT_DIR = REPO / "reports" / "xerj_reader_2026-11-14"
sys.path.insert(0, str(CONTRACT_DIR))

from run_comparative_suite import classify_outcome  # noqa: E402
from specialist_contract import execute, grade_extraction, public_plan  # noqa: E402


def main():
    suite_path = CONTRACT_DIR / "suite_512_cases.json"
    plan_path = CONTRACT_DIR / "pipeline_task_plans.json"

    cases = {c["id"]: c for c in json.loads(suite_path.read_text())}
    plans = {p["case_id"]: p for p in json.loads(plan_path.read_text())["cases"]}

    audit_result = {
        "suite_sha256": hashlib.sha256(suite_path.read_bytes()).hexdigest(),
        "taskplan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "tracks": []
    }

    for path in sorted(HERE.glob("nuextract3_*.json")):
        if any(x in path.stem for x in ["summary", "audit", "smoke"]):
            continue

        data = json.loads(path.read_text())
        records = data.get("records")
        if records is None:
            continue

        ids = [r["case_id"] for r in records]
        assert len(ids) == len(set(ids)), f"Duplicate case IDs in {path.name}"
        assert not set(ids) - cases.keys(), f"Unknown case IDs in {path.name}"

        metadata = data["run_metadata"]
        assert metadata["suite_sha256"] == audit_result["suite_sha256"], f"Suite hash mismatch in {path.name}"
        assert metadata["task_plan_sha256"] == audit_result["taskplan_sha256"], f"Plan hash mismatch in {path.name}"

        track_summary = {
            "artifact": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "records": len(records),
            "complete": len(records) == 512,
            "correct": 0,
            "extraction_exact": 0,
            "extraction_grounded": 0,
            "prompt_reference_leaks": []
        }

        for record in records:
            case = cases[record["case_id"]]
            plan = plans[record["case_id"]]

            if "extraction_grade" in record:
                # Extraction-only & Calculator pipeline track
                fields = record.get("parsed_fields")
                computed_grade = grade_extraction(case["document"], plan, fields)
                for k in ("exact_fields", "grounded"):
                    assert computed_grade[k] == record["extraction_grade"][k], (path.name, record["case_id"], k)

                executor = record.get("executor_result")
                if fields is not None:
                    recomputed_exec = execute(case["document"], public_plan(plan), fields)
                    assert recomputed_exec == executor, (path.name, record["case_id"], "execute")

                answer_correct = classify_outcome(case, executor) in ("correct_answer", "correct_abstention")
                assert answer_correct == record["executor_score"]["answer_correct"], (path.name, record["case_id"], "pipeline_grade")

                track_summary["extraction_exact"] += computed_grade["exact_fields"]
                track_summary["extraction_grounded"] += computed_grade["grounded"]
                track_summary["correct"] += answer_correct
            else:
                # Direct QA track
                answer_correct = classify_outcome(case, record["response"]) in ("correct_answer", "correct_abstention")
                assert answer_correct == record["grade"]["answer_correct"], (path.name, record["case_id"], "direct_qa_grade")
                track_summary["correct"] += answer_correct

            prompt = record.get("prompt") or ""
            if any(leak in prompt for leak in ['"reference_fields":', '"expected":', '"operation":']):
                track_summary["prompt_reference_leaks"].append(record["case_id"])

        assert not track_summary["prompt_reference_leaks"], f"Reference leak detected in {path.name}!"
        audit_result["tracks"].append(track_summary)
        print(f"Replay verified: {path.name} ({len(records)} records, correct={track_summary['correct']})")

    out_path = HERE / "audit.json"
    out_path.write_text(json.dumps(audit_result, indent=2) + "\n")
    print(f"\nAudit complete! Output saved to {out_path}")


if __name__ == "__main__":
    main()
