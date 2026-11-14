#!/usr/bin/env python3
"""Retry only the original Qwen3.5 0.8B HTTP 500 cases.

The source comparison is treated as immutable. Every completed retry is saved
atomically so the run can be resumed without repeating already retried cases.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from model_adapters import get_adapter  # noqa: E402
from run_comparative_suite import check_evidence, classify_outcome  # noqa: E402

MODEL_ID = "qwen3.5-0.8b-mlx"
SOURCE = ROOT / "comparative_512_qwen08b.json"
SUITE = ROOT / "suite_512_cases.json"
OUTPUT = ROOT / "qwen08_http500_retry.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_atomic(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def make_record(case: dict[str, Any], original: dict[str, Any], response: dict[str, Any], index: int) -> dict[str, Any]:
    outcome = classify_outcome(case, response)
    offsets_ok, has_evidence = check_evidence(
        case["document"], response.get("status", "ANSWER"), response.get("evidence", [])
    )
    return {
        "case_id": case["id"],
        "category": case["category"],
        "subset": case.get("subset", "unknown"),
        "index": index,
        "attempts": 1,
        "first_attempt": original,
        "response": response,
        "retries": [],
        "grade": {
            "outcome_classification": outcome,
            "answer_correct": outcome in ("correct_answer", "correct_abstention"),
            "evidence_offsets_valid": offsets_ok,
            "answer_has_evidence": has_evidence,
        },
        "symbolic_formula": None,
        "api_cost_usd": 0.0,
        "latency_ms": response.get("latency_ms", response.get("client_ms", 0.0)),
        "client_ms": response.get("client_ms", 0.0),
        "usage": response.get("usage", {}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds between request starts (default: 2)")
    parser.add_argument("--limit", type=int, help="Retry at most this many remaining cases (for a bounded diagnostic run)")
    parser.add_argument("--continue-after-repeated-500", action="store_true", help="Continue after two consecutive retry HTTP 500 responses")
    args = parser.parse_args()

    source_hash = sha256(SOURCE)
    source = json.loads(SOURCE.read_text())
    original_run = source["model_runs"][MODEL_ID]
    selected = [
        row for row in original_run["cases"]
        if (row.get("response", {}).get("error") or {}).get("http_status") == 500
    ]
    if len(selected) != 53:
        raise SystemExit(f"Expected exactly 53 original HTTP 500 records, found {len(selected)}")
    suite = json.loads(SUITE.read_text())
    suite_by_id = {case["id"]: case for case in suite}
    if len(suite_by_id) != len(suite):
        raise SystemExit("Suite case IDs are not unique")

    if OUTPUT.exists():
        result = json.loads(OUTPUT.read_text())
        if result["benchmark_meta"].get("source_sha256") != source_hash:
            raise SystemExit("Existing retry artifact points to a different source hash; refusing to mix runs")
    else:
        result = {
            "benchmark_meta": {
                "source_artifact": SOURCE.name,
                "source_sha256": source_hash,
                "suite_artifact": SUITE.name,
                "suite_sha256": sha256(SUITE),
                "selected_case_ids": [row["case_id"] for row in selected],
                "selected_count": len(selected),
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "retry_policy": "one retry per original HTTP 500; sequential; no prompt, model, or grading changes",
                "request_interval_seconds": args.interval,
                "stop_after_consecutive_http_500": 2,
            },
            "model_runs": {MODEL_ID: {"model_id": MODEL_ID, "cases": [], "completed_count": 0}},
        }
        save_atomic(OUTPUT, result)

    records = result["model_runs"][MODEL_ID]["cases"]
    already_done = {row["case_id"] for row in records}
    if not already_done.issubset({row["case_id"] for row in selected}):
        raise SystemExit("Existing retry artifact contains IDs outside the 53 selected records")
    todo = [row for row in selected if row["case_id"] not in already_done]
    if args.limit is not None:
        todo = todo[: max(0, args.limit)]
    if not todo:
        print(f"No remaining cases. Retry artifact contains {len(records)}/{len(selected)}.")
        return 0

    adapter = get_adapter(MODEL_ID)
    consecutive_500 = 0
    last_start = 0.0
    for original_row in todo:
        wait = args.interval - (time.monotonic() - last_start) if last_start else 0.0
        if wait > 0:
            time.sleep(wait)
        case_id = original_row["case_id"]
        case = suite_by_id.get(case_id)
        if case is None:
            raise SystemExit(f"Case {case_id} is missing from the frozen suite")
        last_start = time.monotonic()
        response = adapter.query(case)
        record = make_record(case, original_row["response"], response, original_row["index"])
        records.append(record)
        result["model_runs"][MODEL_ID]["completed_count"] = len(records)
        result["model_runs"][MODEL_ID]["last_updated_utc"] = datetime.now(timezone.utc).isoformat()
        result["benchmark_meta"]["finished_utc"] = result["model_runs"][MODEL_ID]["last_updated_utc"] if len(records) == len(selected) else None
        save_atomic(OUTPUT, result)
        status = response.get("status")
        http_status = (response.get("error") or {}).get("http_status")
        print(f"{len(records)}/{len(selected)} {case_id}: {status} http={http_status}", flush=True)
        consecutive_500 = consecutive_500 + 1 if http_status == 500 else 0
        if consecutive_500 >= 2 and not args.continue_after_repeated_500:
            print("Stopped after two consecutive retry HTTP 500 responses; backend failure is reproducible.", flush=True)
            break

    counts = Counter(row["grade"]["outcome_classification"] for row in records)
    print(json.dumps({"completed": len(records), "selected": len(selected), "outcomes": dict(counts), "artifact": str(OUTPUT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
