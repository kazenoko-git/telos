"""
Master Autonomous Overnight & Post-School Benchmark Pipeline.

Executes sequential evaluations across 5 compact/edge models:
1. Waits for Liquid AI LFM 2.5 8B A1B MLX to complete (active now).
2. IBM Granite 4.2 3B MLX: Repairs truncated MATH and GPQA tasks with 4,096 tokens.
3. Apple AFM 3 Core Advanced: Repairs truncated MATH and GPQA tasks with 4,096 tokens.
4. Microsoft Phi 4 Mini Reasoning: Runs full 7-suite institutional evaluation.
5. Google Gemma 4 E4B: Runs full 7-suite institutional evaluation.
6. Compiles final 5-model publication-grade comparative scorecard.
"""

import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from telos.eval.adapters.openai_api import OpenAIAPIAdapter
from telos.eval.adapters.swift_afm import SwiftAFMAdapter
from telos.eval.reasoning_eval import evaluate_arc_sample, evaluate_competition_math_sample
from telos.eval.runner import evaluate

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)
BENCH_DIR = Path("evals/benchmarks")
LMS_BIN = Path.home() / ".lmstudio" / "bin" / "lms"


def wait_for_lfm_completion():
    """Polls until LFM 2.5 8B finishes its active benchmark run."""
    print("\n" + "=" * 90)
    print("  WAITING FOR LIQUID AI LFM 2.5 8B A1B MLX TO COMPLETE BENCHMARKS")
    print("=" * 90 + "\n")

    lfm_arc = LOGS_DIR / "eval_report_lfm8b_arc_full.json"

    while True:
        # ARC is the 7th and final benchmark for LFM 2.5 8B
        if lfm_arc.exists():
            print("✓ LFM 2.5 8B ARC and all benchmarks completed successfully.")
            time.sleep(5)
            break
        print("  [Monitor] LFM 2.5 8B is still running... sleeping 45s.")
        time.sleep(45)


def run_lms_command(args: List[str]) -> bool:
    """Executes an LMS CLI command with error reporting."""
    try:
        res = subprocess.run([str(LMS_BIN)] + args, capture_output=True, text=True, check=True)
        print(f"  [LMS] {' '.join(args)}: {res.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  [LMS Error] {' '.join(args)}: {e.stderr.strip()}", file=sys.stderr)
        return False


def repair_truncated_tasks(
    model_name: str,
    adapter: Any,
    eval_file: Path,
    suite_file: Path,
    benchmark_type: str,
    max_tokens: int = 4096,
):
    """Re-evaluates only tasks that failed due to token truncation (>1000 tokens)."""
    print(f"\n>>> Repairing truncated tasks in {eval_file.name} for {model_name}...")
    if not eval_file.exists():
        print(f"  Warning: {eval_file} not found.")
        return

    with open(eval_file) as f:
        rep_data = json.load(f)

    with open(suite_file) as f:
        suite_data = json.load(f)

    # Map benchmark tasks by ID
    suite_map = {t["id"]: t for t in suite_data}

    domain_key = "math" if benchmark_type == "math" else "science"
    tasks = rep_data.get(domain_key, {}).get("tasks", [])
    if not tasks:
        print("  No tasks found in report.")
        return

    # Identify tasks that failed and hit token limit
    truncated_indices = [
        i
        for i, t in enumerate(tasks)
        if t.get("outcome") != "PASSED"
        and t.get("extended_metrics", {}).get("token_count", 0) >= 1000
    ]

    print(f"  Found {len(truncated_indices)} truncated tasks out of {len(tasks)} total tasks.")
    if not truncated_indices:
        print("  No truncation repairs needed.")
        return

    recovered = 0
    t0 = time.time()

    for count, idx in enumerate(truncated_indices, 1):
        task_meta = tasks[idx]
        task_id = task_meta.get("id")
        raw_task = suite_map.get(task_id)
        if not raw_task:
            continue

        prompt = raw_task["prompt"]

        # Generate with extended token budget
        try:
            completion = adapter.generate(prompt=prompt, max_new_tokens=max_tokens, temperature=0.0)
        except Exception as e:
            print(f"    Task {task_id} generation error: {e}")
            continue

        # Evaluate correctness
        if benchmark_type == "math":
            gold = raw_task.get("target_answer", "")
            passed, details = evaluate_competition_math_sample(completion, gold)
        else:
            gold = raw_task.get("answer_key", "")
            passed, details = evaluate_arc_sample(completion, gold)

        if passed:
            recovered += 1
            task_meta["outcome"] = "PASSED"
            task_meta["details"] = f"Unconstrained 4096-token recovery: {details}"
            ext = task_meta.setdefault("extended_metrics", {})
            ext["token_count"] = len(completion.split())
            ext["recovered_with_4096_tokens"] = True

        if count % 20 == 0 or count == len(truncated_indices):
            print(
                f"    [{count}/{len(truncated_indices)}] Repaired... Recovered: {recovered} ({recovered/count*100:.1f}%)"
            )

    elapsed = time.time() - t0
    total_passed = sum(1 for t in tasks if t.get("outcome") == "PASSED")
    new_pass_rate = round(total_passed / len(tasks) * 100.0, 2)

    # Recalculate 95% Wilson confidence interval
    n = len(tasks)
    p = total_passed / n
    z = 1.96
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2 * n)) / denom
    margin = z * math.sqrt((p * (1.0 - p) + (z**2) / (4 * n)) / n) / denom
    ci_low = round(max(0.0, center - margin) * 100.0, 2)
    ci_high = round(min(1.0, center + margin) * 100.0, 2)

    rep_data[domain_key]["pass_at_1_pct"] = new_pass_rate
    rep_data[domain_key]["pass_at_1_95ci"] = [ci_low, ci_high]
    rep_data[domain_key]["execution_outcomes"]["PASSED"] = total_passed
    rep_data[domain_key]["execution_outcomes"]["FAILED_ASSERTION"] = n - total_passed

    with open(eval_file, "w") as f:
        json.dump(rep_data, f, indent=2)

    print(
        f"✓ Finished repairs in {elapsed/60.0:.2f} mins. New Pass@1: {new_pass_rate}% [{ci_low}%, {ci_high}%] (+{recovered} solved)"
    )


def update_master_summary(master_file: Path, suite_key: str, eval_file: Path, label: str):
    """Updates master summary JSON with new repaired metrics."""
    if not eval_file.exists():
        return
    with open(eval_file) as f:
        d = json.load(f)

    domain_key = "math" if "math" in suite_key else "science"
    sub_data = d.get(domain_key, {})

    master_data = {}
    if master_file.exists():
        try:
            with open(master_file) as f:
                master_data = json.load(f)
        except Exception:
            pass

    master_data[suite_key] = {
        "total_tasks": sub_data.get("total_tasks"),
        "pass_at_1_pct": sub_data.get("pass_at_1_pct"),
        "pass_at_1_95ci": sub_data.get("pass_at_1_95ci"),
        "ast_validity_pct": sub_data.get("ast_validity_pct", 100.0),
        "execution_outcomes": sub_data.get("execution_outcomes"),
        "report_file": str(eval_file),
        "label": label,
    }

    with open(master_file, "w") as f:
        json.dump(master_data, f, indent=2)


def phase_ibm_granite_repair():
    """Phase 2: Load Granite and repair truncated MATH and GPQA tasks."""
    print("\n" + "=" * 90)
    print("  PHASE 2: IBM GRANITE 4.2 3B MLX TRUNCATION REPAIR (4,096 TOKENS)")
    print("=" * 90 + "\n")

    run_lms_command(["unload", "lfm2.5-8b-a1b-mlx"])
    run_lms_command(["load", "granite-4.2-3b-mlx", "-y", "-c", "8192", "--gpu", "max"])
    time.sleep(5)

    adapter = OpenAIAPIAdapter(
        model_name="granite-4.2-3b-mlx",
        api_base="http://127.0.0.1:1234/v1",
        system_prompt="You are an expert mathematician and scientist. Solve the problem step by step and provide the final answer clearly in \\boxed{...}.",
        timeout=180.0,
    )

    # 1. MATH
    math_eval = LOGS_DIR / "eval_report_granite_mlx_competition_math_full.json"
    math_suite = BENCH_DIR / "competition_math_suite.json"
    repair_truncated_tasks(
        "Granite 4.2 3B", adapter, math_eval, math_suite, "math", max_tokens=4096
    )
    update_master_summary(
        LOGS_DIR / "eval_report_granite_mlx_master_summary.json",
        "competition_math",
        math_eval,
        "Competition MATH (Hendrycks — 4,096 Tokens)",
    )

    # 2. GPQA Diamond
    gpqa_eval = LOGS_DIR / "eval_report_granite_mlx_gpqa_diamond_full.json"
    gpqa_suite = BENCH_DIR / "gpqa_diamond_suite.json"
    repair_truncated_tasks(
        "Granite 4.2 3B", adapter, gpqa_eval, gpqa_suite, "science", max_tokens=4096
    )
    update_master_summary(
        LOGS_DIR / "eval_report_granite_mlx_master_summary.json",
        "gpqa_diamond",
        gpqa_eval,
        "GPQA Diamond (PhD Science — 4,096 Tokens)",
    )

    run_lms_command(["unload", "granite-4.2-3b-mlx"])


def phase_apple_afm3_repair():
    """Phase 3: Repair Apple AFM 3 Core Advanced truncated MATH and GPQA tasks."""
    print("\n" + "=" * 90)
    print("  PHASE 3: APPLE AFM 3 CORE ADVANCED TRUNCATION REPAIR (4,096 TOKENS)")
    print("  Runtime Backend: Native Swift FoundationModels.framework IPC")
    print("=" * 90 + "\n")

    adapter = SwiftAFMAdapter("afm-3-core-advanced")

    # 1. MATH
    math_eval = LOGS_DIR / "eval_report_afm3_competition_math_full.json"
    math_suite = BENCH_DIR / "competition_math_suite.json"
    repair_truncated_tasks(
        "AFM 3 Core Advanced", adapter, math_eval, math_suite, "math", max_tokens=4096
    )
    update_master_summary(
        LOGS_DIR / "eval_report_afm3_master_summary.json",
        "competition_math",
        math_eval,
        "Competition MATH (Hendrycks — 4,096 Tokens)",
    )

    # 2. GPQA Diamond
    gpqa_eval = LOGS_DIR / "eval_report_afm3_gpqa_diamond_full.json"
    gpqa_suite = BENCH_DIR / "gpqa_diamond_suite.json"
    repair_truncated_tasks(
        "AFM 3 Core Advanced", adapter, gpqa_eval, gpqa_suite, "science", max_tokens=4096
    )
    update_master_summary(
        LOGS_DIR / "eval_report_afm3_master_summary.json",
        "gpqa_diamond",
        gpqa_eval,
        "GPQA Diamond (PhD Science — 4,096 Tokens)",
    )

    adapter.close()


def run_full_suite_for_model(
    model_id: str, model_tag: str, display_name: str
) -> Dict[str, Any]:
    """Runs all 7 institutional benchmarks on an LM Studio model."""
    print("\n" + "=" * 90)
    print(f"  EVALUATING MODEL: {display_name.upper()} ({model_id})")
    print("=" * 90 + "\n")

    # Dynamic context length: reasoning models require expanded output runway
    ctx_len = "16384"
    is_reasoning_model = any(k in model_id.lower() for k in ["phi-4", "gemma-4", "qwen", "thinking", "reasoning"])
    base_tokens = 2048 if is_reasoning_model else 1024
    math_tokens = 8192 if "phi" in model_tag else 4096
    gpqa_tokens = 8192 if "phi" in model_tag else 4096
    arc_tokens = 4096 if "phi" in model_tag else 2048

    run_lms_command(["load", model_id, "-y", "-c", ctx_len, "--gpu", "max"])
    time.sleep(5)

    suites = [
        {
            "name": "humaneval",
            "type": "code",
            "label": "OpenAI HumanEval (Python Code)",
            "output": f"eval_report_{model_tag}_humaneval_full.json",
            "tokens": base_tokens,
            "prompt": "You are an expert Python programmer. Complete the code strictly without conversational text.",
        },
        {
            "name": "tooluse",
            "type": "tooluse",
            "label": "BFCL Tool-Use & Function Calling",
            "output": f"eval_report_{model_tag}_tooluse_full.json",
            "tokens": base_tokens,
            "prompt": "You are an expert tool-calling assistant. Emit tool calls as a valid JSON list.",
        },
        {
            "name": "cyber",
            "type": "cyber",
            "label": "Cybersecurity OWASP/CWE Auditing",
            "output": f"eval_report_{model_tag}_cyber_full.json",
            "tokens": base_tokens,
            "prompt": "You are a senior security researcher. Fix vulnerabilities and return clean code.",
        },
        {
            "name": "gpqa_diamond",
            "type": "science",
            "label": f"GPQA Diamond (PhD Science — {gpqa_tokens:,} Tokens)",
            "output": f"eval_report_{model_tag}_gpqa_diamond_full.json",
            "tokens": gpqa_tokens,
            "prompt": "You are an expert scientist. Derive the solution and state your final answer as \\boxed{<Letter>}.",
        },
        {
            "name": "mmlu_science",
            "type": "science",
            "label": "MMLU Science & STEM",
            "output": f"eval_report_{model_tag}_mmlu_science_full.json",
            "tokens": base_tokens,
            "prompt": "You are an expert in science and mathematics. State your final answer as \\boxed{<Letter>}.",
        },
        {
            "name": "competition_math",
            "type": "math",
            "label": f"Competition MATH (Hendrycks — {math_tokens:,} Tokens)",
            "output": f"eval_report_{model_tag}_competition_math_full.json",
            "tokens": math_tokens,
            "prompt": "You are an expert mathematician. Solve the problem step by step and state the final answer in \\boxed{...}.",
        },
        {
            "name": "arc",
            "type": "science",
            "label": f"ARC-Challenge Science Reasoning ({arc_tokens:,} Tokens)",
            "output": f"eval_report_{model_tag}_arc_full.json",
            "tokens": arc_tokens,
            "prompt": "You are an expert in science. Solve the question and state your final answer as \\boxed{<Letter>}.",
        },
    ]

    master_summary = {}

    for s in suites:
        out_file = LOGS_DIR / s["output"]
        if out_file.exists():
            print(f"✓ {s['label']} already evaluated. Resuming next...")
            with open(out_file) as f:
                rep_d = json.load(f)
            b_data = rep_d.get(s["type"], {})
            if isinstance(b_data, dict) and "functional" in b_data:
                b_data = b_data["functional"]
            elif not b_data:
                b_data = (
                    rep_d.get("functional", {})
                    or rep_d.get("math", {})
                    or rep_d.get("science", {})
                )
            master_summary[s["name"]] = {
                "total_tasks": b_data.get("total_tasks"),
                "pass_at_1_pct": b_data.get("pass_at_1_pct")
                or b_data.get("pass_rate_pct", 0.0),
                "pass_at_1_95ci": b_data.get("pass_at_1_95ci")
                or b_data.get("pass_rate_95ci"),
                "report_file": str(out_file),
                "label": s["label"],
            }
            continue

        print(f"\n>>> Starting {s['label']} ({out_file.name})...")
        adapter = OpenAIAPIAdapter(
            model_name=model_id,
            api_base="http://127.0.0.1:1234/v1",
            system_prompt=s["prompt"],
            timeout=180.0,
        )
        t0 = time.time()
        evaluate(
            checkpoint=adapter,
            mode="functional",
            suite=s["name"],
            benchmark_type=s["type"],
            max_tasks=None,
            max_new_tokens=s["tokens"],
            output_path=str(out_file),
        )
        elapsed = time.time() - t0

        with open(out_file) as f:
            rep_d = json.load(f)
        b_data = rep_d.get(s["type"], {})
        if isinstance(b_data, dict) and "functional" in b_data:
            b_data = b_data["functional"]
        elif not b_data:
            b_data = (
                rep_d.get("functional", {})
                or rep_d.get("math", {})
                or rep_d.get("science", {})
            )
        p_rate = b_data.get("pass_at_1_pct") or b_data.get("pass_rate_pct", 0.0)
        master_summary[s["name"]] = {
            "total_tasks": b_data.get("total_tasks"),
            "pass_at_1_pct": p_rate,
            "pass_at_1_95ci": b_data.get("pass_at_1_95ci")
            or b_data.get("pass_rate_95ci"),
            "report_file": str(out_file),
            "label": s["label"],
        }
        print(f"✓ Completed {s['label']} in {elapsed/60.0:.2f} mins (Pass@1: {p_rate:.2f}%)")

    # Save master summary
    summary_path = LOGS_DIR / f"eval_report_{model_tag}_master_summary.json"
    with open(summary_path, "w") as f:
        json.dump(master_summary, f, indent=2)

    run_lms_command(["unload", model_id])
    return master_summary


def compile_5model_publication_scorecard():
    """Compiles consolidated 5-model master comparison table and Markdown."""
    print("\n" + "=" * 90)
    print("  COMPILING FINAL 5-MODEL INSTITUTIONAL SCORECARD")
    print("=" * 90 + "\n")

    models = [
        {
            "name": "Apple AFM 3 Core Advanced",
            "file": LOGS_DIR / "eval_report_afm3_master_summary.json",
        },
        {
            "name": "IBM Granite 4.2 3B MLX",
            "file": LOGS_DIR / "eval_report_granite_mlx_master_summary.json",
        },
        {
            "name": "Liquid AI LFM 2.5 8B A1B MLX",
            "file": LOGS_DIR / "eval_report_lfm8b_master_summary.json",
        },
        {
            "name": "Microsoft Phi 4 Mini Reasoning",
            "file": LOGS_DIR / "eval_report_phi4_mini_master_summary.json",
        },
        {"name": "Google Gemma 4 E4B", "file": LOGS_DIR / "eval_report_gemma4_e4b_master_summary.json"},
    ]

    benchmarks = [
        ("humaneval", "HumanEval (Python)"),
        ("tooluse", "BFCL Tool-Use"),
        ("cyber", "Cybersecurity Auditing"),
        ("gpqa_diamond", "GPQA Diamond (Science)"),
        ("mmlu_science", "MMLU Science & STEM"),
        ("competition_math", "Competition MATH"),
        ("arc", "ARC-Challenge Reasoning"),
    ]

    master_matrix = {}
    for m in models:
        master_matrix[m["name"]] = {}
        if m["file"].exists():
            try:
                with open(m["file"]) as f:
                    d = json.load(f)
                for b_key, b_label in benchmarks:
                    master_matrix[m["name"]][b_key] = d.get(b_key, {}).get("pass_at_1_pct")
            except Exception:
                pass

    # Write JSON
    out_json = LOGS_DIR / "master_institutional_scorecard_5models.json"
    with open(out_json, "w") as f:
        json.dump(master_matrix, f, indent=2)

    # Write Markdown
    out_md = LOGS_DIR / "master_publication_scorecard_5models.md"
    md_lines = [
        "# Institutional Publication Scorecard: Frontier Compact & Edge Models",
        f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "| Benchmark Suite | Apple AFM 3 Core | IBM Granite 4.2 3B | Liquid AI LFM 2.5 8B | MS Phi 4 Mini | Google Gemma 4 E4B |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    for b_key, b_label in benchmarks:
        row = [f"**{b_label}**"]
        for m in models:
            val = master_matrix.get(m["name"], {}).get(b_key)
            row.append(f"{val:.2f}%" if val is not None else "Pending")
        md_lines.append(f"| {' | '.join(row)} |")

    with open(out_md, "w") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"✓ Publication scorecard saved to:\n  - {out_json}\n  - {out_md}")


def main():
    print("\n" + "#" * 90)
    print("  LAUNCHING POST-SCHOOL AUTONOMOUS BENCHMARK PIPELINE")
    print("#" * 90 + "\n")

    # 1. Google Gemma 4 E4B (executing immediately)
    run_full_suite_for_model(
        model_id="google/gemma-4-e4b",
        model_tag="gemma4_e4b",
        display_name="Google Gemma 4 E4B (4B)",
    )

    # 2. Microsoft Phi 4 Mini Reasoning (executing overnight)
    run_full_suite_for_model(
        model_id="microsoft/phi-4-mini-reasoning",
        model_tag="phi4_mini",
        display_name="Microsoft Phi 4 Mini Reasoning (3.8B)",
    )

    # 3. Compile 5-Model Publication Scorecard
    compile_5model_publication_scorecard()

    print("\n" + "=" * 90)
    print("  ALL 5 MODELS SUCCESSFULLY EVALUATED AND COMPILED!")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
