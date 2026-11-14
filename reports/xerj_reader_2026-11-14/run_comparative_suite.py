#!/usr/bin/env python3
"""Comparative evaluation runner across frozen 512-case benchmark suite.

Orchestrates multi-model evaluation across:
1. XERJ Reader-1 (API baseline)
2. Ternary Bonsai 4 (LM Studio local)
3. Ministral 3 8B Instruct 2512 (LM Studio local)
4. Gemma 4 E4B (LM Studio local)
5. LiquidAI LFM3-Omni-2B (Direct MLX resident worker)
6. Gemini 3.5 Flash-Lite (Google Generative Language API)

Features:
- Checkpoints after every single case for zero data loss and transparent resumption.
- Preserves historical extended-run.json without modification.
- Evaluates identical frozen 512 cases with dual-subset disaggregation (inspected_240 vs new_272).
- Distinguishes 6 explicit outcome classifications:
    - correct_answer
    - correct_abstention
    - incorrect_answer
    - incorrect_abstention
    - malformed_response
    - transport_error
- Strict provenance offset checks and answer evidence requirements.
- Retains symbolic formulas and samples 1/4 executor replays for Reader-1.
- Tracks token counts, API list prices, local runtime latencies, and host memory pressure.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

# Import unified model adapter interfaces
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from model_adapters import (  # noqa: E402
    ADAPTER_FACTORIES,
    BaseAdapter,
    get_adapter,
    key_from_env,
)

# Standard evaluation models evaluated in comparative study
DEFAULT_MODELS = [
    "xerj-reader-1",
    "ternary-bonsai-4b-mlx",
    "ministral-3-8b-instruct-2512",
    "google/gemma-4-e4b",
    "lfm3-omni-2b",
    "gemini-3.5-flash-lite",
]


class RateLimiter:
    """Thread-safe start interval rate limiter for network requests."""

    def __init__(self, interval: float = 1.15):
        self.lock = threading.Lock()
        self.next_start = 0.0
        self.interval = interval

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            start = max(now, self.next_start)
            self.next_start = start + self.interval
        delay = start - time.monotonic()
        if delay > 0:
            time.sleep(delay)


def get_memory_stats() -> Dict[str, Any]:
    """Capture host virtual memory statistics on macOS using vm_stat."""
    try:
        out = subprocess.check_output(["vm_stat"], text=True)
        stats = {}
        for line in out.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                stats[k.strip()] = v.strip().rstrip(".")
        return stats
    except Exception as e:
        return {"error": str(e)}


def equal(actual: Any, expected: Any) -> bool:
    """Check mathematical or structured equivalence between actual and expected."""
    if isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            equal(actual.get(k), v) for k, v in expected.items()
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(equal(a, b) for a, b in zip(actual, expected))
        )
    try:
        s_act = str(actual).strip().rstrip("%")
        s_exp = str(expected).strip().rstrip("%")
        return Decimal(s_act) == Decimal(s_exp)
    except (InvalidOperation, TypeError):
        return str(actual).strip() == str(expected).strip()


def classify_outcome(case: Dict[str, Any], response: Dict[str, Any]) -> str:
    """Classify model outcome into one of mutually exclusive categories."""
    err = response.get("error")
    status = response.get("status")

    if err is not None or status == "ERROR":
        err_kind = response.get("error", {}).get("kind", "") if isinstance(response.get("error"), dict) else ""
        if err_kind in (
            "JSONDecodeError",
            "ContractError",
            "AttributeError",
            "TypeError",
            "WorkerDied",
            "WorkerError",
        ):
            return "contract_error"
        if err_kind == "rate_limit_error" or (isinstance(response.get("error"), dict) and response["error"].get("http_status") == 429):
            return "rate_limit_error"
        return "transport_error"
    if status == "MALFORMED" or status not in (
        "ANSWER",
        "ABSENT",
        "UNKNOWN",
        "CONFLICT",
        "AMBIGUOUS",
    ):
        return "malformed_response"

    expects_answer = "expected" in case
    accepted_statuses = case.get("statuses", ["ANSWER"])

    if expects_answer:
        # Case expected a specific extracted or computed value
        if status in accepted_statuses:
            if equal(response.get("value"), case["expected"]):
                return "correct_answer"
            else:
                return "incorrect_answer"
        else:
            # Model refused or abstained when an answer was expected
            return "incorrect_abstention"
    else:
        # Case expected an abstention (e.g. absent, unknown, conflict, ambiguous)
        if status in accepted_statuses:
            return "correct_abstention"
        elif status == "ANSWER":
            # Model fabricated or hallucinated an answer instead of abstaining
            return "incorrect_answer"
        else:
            # Model returned an unsupported or wrong abstention status
            return "incorrect_abstention"


def check_evidence(
    doc: str, status: str, evidence: List[Dict[str, Any]]
) -> Tuple[bool, bool]:
    """Verify evidence quote character offsets and verify presence for answers."""
    if not evidence:
        # Empty evidence list passes offset validity check
        offsets_ok = True
    else:
        # Each evidence span must match verbatim document characters exactly
        offsets_ok = all(
            isinstance(e.get("start"), int)
            and isinstance(e.get("end"), int)
            and 0 <= e["start"] < e["end"] <= len(doc)
            and doc[e["start"] : e["end"]] == e.get("quote")
            for e in evidence
        )

    # Answers require citation evidence; abstentions are exempt
    has_evidence = (status != "ANSWER") or bool(evidence)
    return offsets_ok, has_evidence


def calculate_api_cost(model_id: str, usage: Dict[str, Any]) -> float:
    """Calculate actual or estimated API dollar charges based on token usage."""
    in_tok = usage.get("input_tokens", 0)
    out_tok = usage.get("output_tokens", 0)

    if model_id == "xerj-reader-1":
        # Documented list price: $0.01 per 1M input tokens. Output and replays free.
        return (in_tok / 1_000_000.0) * 0.01
    elif model_id == "gemini-3.5-flash-lite":
        # Gemini 1.5/2.5 Flash-Lite list price: $0.075 / 1M input, $0.30 / 1M output
        # Without an active API key, zero actual dollars billed
        return 0.0
    else:
        # Local MLX models incur zero API cost
        return 0.0


def replay_reader_program(
    case: Dict[str, Any], program: Any, api_key: str, base_url: str = "https://platform.xerj.ai"
) -> Tuple[Optional[Dict[str, Any]], float, bool]:
    """Execute symbolic program against XERJ deterministic executor endpoint."""
    if not api_key or not program:
        return None, 0.0, False

    payload = {k: case[k] for k in ("document", "question", "type", "ref_date") if k in case}
    payload["program"] = program

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/v1/execute",
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "telos-reader-evaluation/1.6.0",
        },
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            replay_res = json.load(resp)
        elapsed_ms = round((time.monotonic() - t0) * 1000, 2)
        return replay_res, elapsed_ms, True
    except Exception as err:
        elapsed_ms = round((time.monotonic() - t0) * 1000, 2)
        return {"error": str(err)}, elapsed_ms, False


def compute_subset_metrics(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate aggregated scores, category metrics, and latencies for a slice."""
    total = len(cases)
    if total == 0:
        return {
            "total_cases": 0,
            "correct_count": 0,
            "accuracy_pct": 0.0,
            "outcomes": {
                "correct_answer": 0,
                "correct_abstention": 0,
                "incorrect_answer": 0,
                "incorrect_abstention": 0,
                "malformed_response": 0,
                "contract_error": 0,
                "rate_limit_error": 0,
                "transport_error": 0,
            },
            "evidence": {
                "valid_offsets_count": 0,
                "valid_offsets_pct": 0.0,
                "has_evidence_count": 0,
                "has_evidence_pct": 0.0,
            },
            "latencies_ms": {
                "median": 0.0,
                "mean": 0.0,
                "client_median": 0.0,
            },
            "tokens": {
                "input": 0,
                "output": 0,
                "total": 0,
                "total_cost_usd": 0.0,
            },
            "categories": {},
        }

    correct_cases = [c for c in cases if c["grade"]["answer_correct"]]
    correct = len(correct_cases)
    accuracy = round((correct / total) * 100, 2)

    # Count outcome classifications
    outcomes = Counter(c["grade"]["outcome_classification"] for c in cases)

    # Evidence metrics
    valid_offsets = sum(1 for c in cases if c["grade"]["evidence_offsets_valid"])
    has_evidence = sum(1 for c in cases if c["grade"]["answer_has_evidence"])

    # Latencies
    latencies = [c.get("latency_ms", 0.0) for c in cases if c.get("latency_ms")]
    client_latencies = [c.get("client_ms", 0.0) for c in cases if c.get("client_ms")]

    # Token and cost totals
    in_tok = sum(c.get("usage", {}).get("input_tokens", 0) for c in cases)
    out_tok = sum(c.get("usage", {}).get("output_tokens", 0) for c in cases)
    tot_tok = in_tok + out_tok
    total_cost = sum(c.get("api_cost_usd", 0.0) for c in cases)

    # Category performance breakdown
    cat_cases: Dict[str, List[Dict[str, Any]]] = {}
    for c in cases:
        cat_cases.setdefault(c["category"], []).append(c)

    categories = {}
    for cat, items in sorted(cat_cases.items()):
        cat_corr = sum(1 for item in items if item["grade"]["answer_correct"])
        categories[cat] = {
            "total": len(items),
            "correct": cat_corr,
            "accuracy_pct": round((cat_corr / len(items)) * 100, 2),
        }

    return {
        "total_cases": total,
        "correct_count": correct,
        "accuracy_pct": accuracy,
        "outcomes": {
            "correct_answer": outcomes.get("correct_answer", 0),
            "correct_abstention": outcomes.get("correct_abstention", 0),
            "incorrect_answer": outcomes.get("incorrect_answer", 0),
            "incorrect_abstention": outcomes.get("incorrect_abstention", 0),
            "malformed_response": outcomes.get("malformed_response", 0),
            "contract_error": outcomes.get("contract_error", 0),
            "rate_limit_error": outcomes.get("rate_limit_error", 0),
            "transport_error": outcomes.get("transport_error", 0),
        },
        "evidence": {
            "valid_offsets_count": valid_offsets,
            "valid_offsets_pct": round((valid_offsets / total) * 100, 2),
            "has_evidence_count": has_evidence,
            "has_evidence_pct": round((has_evidence / total) * 100, 2),
        },
        "latencies_ms": {
            "median": round(statistics.median(latencies), 2) if latencies else 0.0,
            "mean": round(statistics.mean(latencies), 2) if latencies else 0.0,
            "client_median": (
                round(statistics.median(client_latencies), 2) if client_latencies else 0.0
            ),
        },
        "tokens": {
            "input": in_tok,
            "output": out_tok,
            "total": tot_tok,
            "total_cost_usd": round(total_cost, 8),
        },
        "categories": categories,
    }


def evaluate_single_model(
    model_id: str,
    cases: List[Dict[str, Any]],
    existing_run: Optional[Dict[str, Any]],
    output_path: Path,
    replay_every: int = 4,
    rate_limiter: Optional[RateLimiter] = None,
    max_retries: int = 2,
) -> Dict[str, Any]:
    """Run evaluation for one model with atomic per-case checkpointing."""
    print(f"\n=======================================================", flush=True)
    print(f"Starting evaluation: {model_id} ({len(cases)} cases)", flush=True)
    print(f"=======================================================", flush=True)

    adapter: BaseAdapter = get_adapter(model_id)
    xerj_key = key_from_env("XERJ_API_KEY") if model_id == "xerj-reader-1" else ""

    # Load existing checkpointed cases if available
    completed_cases: List[Dict[str, Any]] = []
    completed_ids = set()
    warm_up_notes: Dict[str, Any] = {}
    memory_before = get_memory_stats()

    if existing_run and "model_runs" in existing_run and model_id in existing_run["model_runs"]:
        prior = existing_run["model_runs"][model_id]
        completed_cases = prior.get("cases", [])
        completed_ids = {c["case_id"] for c in completed_cases}
        warm_up_notes = prior.get("warm_up_notes", {})
        memory_before = prior.get("memory_pressure", {}).get("before", memory_before)
        print(f"Resuming {model_id}: {len(completed_cases)} cases already finished.", flush=True)

    t_start = time.monotonic()
    first_query_time = None

    try:
        for idx, case in enumerate(cases):
            case_id = case["id"]
            if case_id in completed_ids:
                continue

            if rate_limiter:
                rate_limiter.wait()

            attempts = 0
            first_attempt_res = None
            retry_records = []
            final_res = None

            # Execute query with retry separation
            while attempts <= max_retries:
                attempts += 1
                query_t0 = time.monotonic()
                res = adapter.query(case)
                q_dur = round((time.monotonic() - query_t0) * 1000, 2)

                if first_attempt_res is None:
                    first_attempt_res = res
                    if first_query_time is None:
                        first_query_time = q_dur

                # Check if retry is required for transient errors
                err_body = str(res.get("error", {}).get("body", ""))
                is_transient = (
                    res.get("status") == "ERROR"
                    and res.get("error", {}).get("kind") in ("HTTPError", "Timeout", "ConnectionError", "rate_limit_error")
                    and res.get("error", {}).get("http_status") not in (401, 403, 404)
                    and "limit: 500" not in err_body
                    and "per day" not in err_body
                )

                if not is_transient or attempts > max_retries:
                    final_res = res
                    break
                else:
                    retry_records.append({"attempt": attempts, "response": res})
                    http_code = res.get("error", {}).get("http_status")
                    delay = 15.0 if http_code == 429 else (2.0 * attempts)
                    time.sleep(delay)

            # Classify outcome and check evidence
            outcome = classify_outcome(case, final_res)
            offsets_ok, has_ev = check_evidence(
                case["document"], final_res.get("status", "ANSWER"), final_res.get("evidence", [])
            )
            is_correct = outcome in ("correct_answer", "correct_abstention")

            record: Dict[str, Any] = {
                "case_id": case_id,
                "category": case["category"],
                "subset": case.get("subset", "unknown"),
                "index": idx,
                "attempts": attempts,
                "first_attempt": first_attempt_res,
                "response": final_res,
                "retries": retry_records,
                "grade": {
                    "outcome_classification": outcome,
                    "answer_correct": is_correct,
                    "evidence_offsets_valid": offsets_ok,
                    "answer_has_evidence": has_ev,
                },
                "symbolic_formula": final_res.get("symbolic", {}).get("formula"),
                "api_cost_usd": calculate_api_cost(model_id, final_res.get("usage", {})),
                "latency_ms": final_res.get("latency_ms", final_res.get("client_ms", 0.0)),
                "client_ms": final_res.get("client_ms", 0.0),
                "usage": final_res.get("usage", {}),
            }

            # For Reader-1, sample 1/4 executor replays
            if model_id == "xerj-reader-1" and idx % replay_every == 0:
                prog = final_res.get("symbolic", {}).get("program")
                if prog is not None:
                    if rate_limiter:
                        rate_limiter.wait()
                    rep_res, rep_ms, rep_ok = replay_reader_program(case, prog, xerj_key)
                    record["replay"] = rep_res
                    record["replay_client_ms"] = rep_ms
                    if rep_ok and rep_res:
                        matches = (
                            rep_res.get("status") == final_res.get("status")
                            and equal(rep_res.get("value"), final_res.get("value"))
                        )
                        record["replay_matches"] = matches
                    else:
                        record["replay_matches"] = False

            completed_cases.append(record)
            completed_ids.add(case_id)

            # Record warm up latency note on first completed case
            if len(completed_cases) == 1 and not warm_up_notes:
                warm_up_notes["cold_start_client_ms"] = first_query_time or record["client_ms"]

            # Save atomic checkpoint to disk
            if existing_run is None:
                existing_run = {"model_runs": {}}
            if "model_runs" not in existing_run:
                existing_run["model_runs"] = {}

            # Partition cases by subset for live progress updates
            inspected = [c for c in completed_cases if c.get("subset") == "inspected_240"]
            new_cases = [c for c in completed_cases if c.get("subset") == "new_272"]

            existing_run["model_runs"][model_id] = {
                "model_id": model_id,
                "completed_count": len(completed_cases),
                "warm_up_notes": warm_up_notes,
                "memory_pressure": {
                    "before": memory_before,
                    "latest": get_memory_stats(),
                },
                "summary": {
                    "inspected_240": compute_subset_metrics(inspected),
                    "new_272": compute_subset_metrics(new_cases),
                    "combined_512": compute_subset_metrics(completed_cases),
                },
                "cases": completed_cases,
            }

            # Write atomic tempfile and replace
            tmp_path = output_path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(existing_run, indent=2, ensure_ascii=False) + "\n")
            tmp_path.replace(output_path)

            # Progress logging
            if len(completed_cases) % 25 == 0 or len(completed_cases) == len(cases):
                total_corr = sum(1 for c in completed_cases if c["grade"]["answer_correct"])
                print(
                    f"[{model_id}] {len(completed_cases)}/{len(cases)} done · "
                    f"Correct: {total_corr}/{len(completed_cases)} ({total_corr*100/len(completed_cases):.1f}%) · "
                    f"Last: {case_id} ({outcome})",
                    flush=True,
                )

    finally:
        adapter.close()

    # Finalize model run metadata
    wall_duration = round(time.monotonic() - t_start, 2)
    memory_after = get_memory_stats()
    model_latencies = [c["latency_ms"] for c in completed_cases if c.get("latency_ms")]

    warm_up_notes["median_subsequent_latency_ms"] = (
        round(statistics.median(model_latencies[1:]), 2)
        if len(model_latencies) > 1
        else warm_up_notes.get("cold_start_client_ms", 0.0)
    )

    inspected = [c for c in completed_cases if c.get("subset") == "inspected_240"]
    new_cases = [c for c in completed_cases if c.get("subset") == "new_272"]

    model_summary = {
        "model_id": model_id,
        "completed_count": len(completed_cases),
        "wall_duration_seconds": wall_duration,
        "warm_up_notes": warm_up_notes,
        "memory_pressure": {
            "before": memory_before,
            "after": memory_after,
        },
        "summary": {
            "inspected_240": compute_subset_metrics(inspected),
            "new_272": compute_subset_metrics(new_cases),
            "combined_512": compute_subset_metrics(completed_cases),
        },
        "cases": completed_cases,
    }

    if model_id == "xerj-reader-1":
        replayed = [c for c in completed_cases if "replay_matches" in c]
        matched = [c for c in replayed if c["replay_matches"]]
        model_summary["replay_evaluation"] = {
            "sampled_replays": len(replayed),
            "matched_replays": len(matched),
            "match_pct": round((len(matched) / len(replayed)) * 100, 2) if replayed else 0.0,
        }

    return model_summary


def generate_markdown_report(data: Dict[str, Any], output_md_path: Path) -> None:
    """Generate comprehensive markdown summary table from comparative evaluation run."""
    lines = [
        "# Comparative Evaluation Report: 512-Case Benchmark Suite",
        "",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        f"Frozen Cases File: {data['benchmark_meta']['suite_cases_file']} (SHA-256: `{data['benchmark_meta']['suite_sha256']}`)",
        f"Total Benchmark Cases: {data['benchmark_meta']['case_count']} (240 inspected historical + 272 new held-out)",
        "",
        "## Overall Comparative Scoreboard",
        "",
        "| Model | Runtime | Inspected 240 Acc | New 272 Acc | Combined 512 Acc | Offset Valid % | Total Cost | Median Latency |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for model_id, run in data["model_runs"].items():
        summ = run["summary"]
        i_tot = summ["inspected_240"].get("total_cases", 0)
        i_corr = summ["inspected_240"].get("correct_count", 0)
        i_pct = summ["inspected_240"].get("accuracy_pct", 0.0)
        i_acc = f"{i_pct}% ({i_corr}/{i_tot})" if i_tot else "N/A"

        n_tot = summ["new_272"].get("total_cases", 0)
        n_corr = summ["new_272"].get("correct_count", 0)
        n_pct = summ["new_272"].get("accuracy_pct", 0.0)
        n_acc = f"{n_pct}% ({n_corr}/{n_tot})" if n_tot else "N/A"

        c_tot = summ["combined_512"].get("total_cases", 0)
        c_corr = summ["combined_512"].get("correct_count", 0)
        c_pct = summ["combined_512"].get("accuracy_pct", 0.0)
        c_acc = f"**{c_pct}%** ({c_corr}/{c_tot})" if c_tot else "N/A"

        off_val = f"{summ['combined_512']['evidence'].get('valid_offsets_pct', 0.0)}%"
        cost = f"${summ['combined_512']['tokens'].get('total_cost_usd', 0.0):.6f}"
        lat = f"{summ['combined_512']['latencies_ms'].get('median', 0.0)} ms"
        lines.append(
            f"| `{model_id}` | local/API | {i_acc} | {n_acc} | {c_acc} | {off_val} | {cost} | {lat} |"
        )

    lines.extend([
        "",
        "## Outcome Classifications (Combined 512 Cases)",
        "",
        "| Model | Correct Ans | Correct Abst | Incorrect Ans | Incorrect Abst | Malformed | Contract Err | Rate Limit Err | Transport Err |",
        "|---|---|---|---|---|---|---|---|---|",
    ])

    for model_id, run in data["model_runs"].items():
        out = run["summary"]["combined_512"]["outcomes"]
        lines.append(
            f"| `{model_id}` | {out['correct_answer']} | {out['correct_abstention']} | "
            f"{out['incorrect_answer']} | {out['incorrect_abstention']} | "
            f"{out['malformed_response']} | {out.get('contract_error', 0)} | "
            f"{out.get('rate_limit_error', 0)} | {out['transport_error']} |"
        )

    lines.extend([
        "",
        "## Category Accuracies (Combined 512 Cases)",
        "",
        "| Category | " + " | ".join(f"`{m}`" for m in data["model_runs"].keys()) + " |",
        "|---|" + "|".join("---" for _ in data["model_runs"].keys()) + "|",
    ])

    # Collect all categories
    first_run = next(iter(data["model_runs"].values()))
    categories = sorted(first_run["summary"]["combined_512"]["categories"].keys())
    for cat in categories:
        row = [cat]
        for model_id, run in data["model_runs"].items():
            cat_data = run["summary"]["combined_512"]["categories"].get(cat, {})
            corr = cat_data.get("correct", 0)
            tot = cat_data.get("total", 0)
            pct = cat_data.get("accuracy_pct", 0.0)
            row.append(f"{corr}/{tot} ({pct}%)")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    output_md_path.write_text("\n".join(lines) + "\n")
    print(f"Summary markdown written to {output_md_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default="suite_512_cases.json", help="Suite cases JSON filename")
    parser.add_argument("--output", default="comparative_512_v2_run.json", help="Target output JSON filename")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="List of models to evaluate")
    parser.add_argument("--limit", type=int, help="Optional case limit for quick validation")
    parser.add_argument("--replay-every", type=int, default=4, help="Replay cadence for Reader-1")
    parser.add_argument("--merge", nargs="+", help="Merge multiple run JSON files into target output")
    args = parser.parse_args()

    output_path = ROOT / args.output

    if args.merge:
        # Merge mode: combine model_runs from multiple run artifacts
        combined_data: Dict[str, Any] = {"model_runs": {}}
        for src in args.merge:
            src_path = ROOT / src if not Path(src).is_absolute() else Path(src)
            src_json = json.loads(src_path.read_text())
            if "benchmark_meta" in src_json and "benchmark_meta" not in combined_data:
                combined_data["benchmark_meta"] = src_json["benchmark_meta"]
            for m_id, m_run in src_json.get("model_runs", {}).items():
                combined_data["model_runs"][m_id] = m_run

        combined_data.setdefault("benchmark_meta", {})["finished_utc"] = datetime.now(timezone.utc).isoformat()
        output_path.write_text(json.dumps(combined_data, indent=2, ensure_ascii=False) + "\n")
        print(f"Successfully merged {len(args.merge)} files into {output_path}", flush=True)
        generate_markdown_report(combined_data, output_path.with_suffix(".md"))
        return

    cases_path = ROOT / args.cases
    if not cases_path.exists():
        raise FileNotFoundError(f"Cases file not found: {cases_path}")

    raw_cases_bytes = cases_path.read_bytes()
    cases_sha256 = hashlib.sha256(raw_cases_bytes).hexdigest()
    all_cases: List[Dict[str, Any]] = json.loads(raw_cases_bytes)

    if args.limit is not None:
        all_cases = all_cases[: args.limit]

    output_path = ROOT / args.output

    # Load existing benchmark data if resuming
    existing_data: Dict[str, Any] = {}
    if output_path.exists():
        try:
            existing_data = json.loads(output_path.read_text())
            print(f"Loaded existing checkpoint from {output_path}", flush=True)
        except Exception as e:
            print(f"Notice: Existing {output_path} could not be read ({e}); starting fresh.", flush=True)

    # Initialize benchmark metadata
    existing_data.setdefault(
        "benchmark_meta",
        {
            "suite_cases_file": args.cases,
            "suite_sha256": cases_sha256,
            "case_count": len(all_cases),
            "subsets": {
                "inspected_240": sum(1 for c in all_cases if c.get("subset") == "inspected_240"),
                "new_272": sum(1 for c in all_cases if c.get("subset") == "new_272"),
            },
            "started_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    existing_data.setdefault("model_runs", {})

    rate_limiters = {
        "xerj-reader-1": RateLimiter(interval=1.15),
        "gemini-3.5-flash-lite": RateLimiter(interval=4.65),
    }

    # Execute models in sequential batches to prevent VRAM thrashing
    for model_id in args.models:
        # Check if model already completed full set
        model_done = False
        if model_id in existing_data.get("model_runs", {}):
            completed = existing_data["model_runs"][model_id].get("completed_count", 0)
            if completed >= len(all_cases):
                print(f"Skipping already completed model: {model_id} ({completed}/{len(all_cases)})", flush=True)
                model_done = True

        if not model_done:
            limiter = rate_limiters.get(model_id)
            model_summary = evaluate_single_model(
                model_id=model_id,
                cases=all_cases,
                existing_run=existing_data,
                output_path=output_path,
                replay_every=args.replay_every,
                rate_limiter=limiter,
            )
            existing_data["model_runs"][model_id] = model_summary

            # Save full checkpoint
            tmp = output_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(existing_data, indent=2, ensure_ascii=False) + "\n")
            tmp.replace(output_path)

    # Finalize output and write markdown summary
    existing_data["benchmark_meta"]["finished_utc"] = datetime.now(timezone.utc).isoformat()
    output_path.write_text(json.dumps(existing_data, indent=2, ensure_ascii=False) + "\n")
    print(f"\nComparative evaluation complete! Written to {output_path}", flush=True)

    md_output_path = output_path.with_suffix(".md")
    generate_markdown_report(existing_data, md_output_path)


if __name__ == "__main__":
    main()
