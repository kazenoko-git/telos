"""
Benchmark Execution Pipeline for Liquid AI's LFM 2.5 8B A1B MLX via LM Studio.
Evaluates:
1. OpenAI HumanEval (Python Code — 164 tasks, 8,096-token budget)
2. BFCL Tool-Use & Function Calling (30 tasks, 4,096-token budget)
3. Cybersecurity OWASP/CWE Auditing (50 tasks, 8,096-token budget)
4. GPQA Diamond (PhD Science — 198 tasks, 8,096-token budget)
5. MMLU Science & STEM (833 tasks, 8,096-token budget)
6. Competition MATH (Hendrycks Olympiad — 700 tasks, 8,096-token budget)
7. ARC-Challenge Science Reasoning (1,172 tasks, 4,096-token budget)
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

from telos.eval.adapters.openai_api import OpenAIAPIAdapter
from telos.eval.runner import evaluate
from telos.eval.tooluse.evaluator import _compare_arguments

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "lfm2.5-8b-a1b-mlx"
API_BASE = "http://127.0.0.1:1234/v1"

SUITES = [
    {
        "name": "humaneval",
        "label": "OpenAI HumanEval (Python Code)",
        "type": "code",
        "system_prompt": "You are an expert programming assistant. Complete the requested Python function directly. Always format your code in a ```python ... ``` code block. Do not format code using LaTeX math.",
        "output": "eval_report_lfm8b_humaneval_full.json",
        "max_new_tokens": 8096
    },
    {
        "name": "tooluse",
        "label": "BFCL Tool-Use & Function Calling",
        "type": "tooluse",
        "system_prompt": "You are an AI assistant with tool access. Call the tool using JSON format: {\"tool\": \"...\", \"arguments\": {...}}",
        "output": "eval_report_lfm8b_tooluse_full.json",
        "max_new_tokens": 4096
    },
    {
        "name": "cyber",
        "label": "Cybersecurity OWASP/CWE Auditing",
        "type": "cyber",
        "system_prompt": "You are an expert cybersecurity auditor. Identify the CWE, explain the vulnerability, and provide secure remediated code in a ```python ... ``` code block.",
        "output": "eval_report_lfm8b_cyber_full.json",
        "max_new_tokens": 8096
    },
    {
        "name": "gpqa_diamond",
        "label": "GPQA Diamond (PhD Science — 8,096 Tokens)",
        "type": "science",
        "system_prompt": "You are an expert scientist. Derive the solution step by step and conclude with the correct letter option in the format: Answer: <Letter>.",
        "output": "eval_report_lfm8b_gpqa_diamond_full.json",
        "max_new_tokens": 8096
    },
    {
        "name": "mmlu_science",
        "label": "MMLU Science & STEM",
        "type": "science",
        "system_prompt": "You are an expert in science and mathematics. Derive the solution and state the correct letter option in the format: Answer: <Letter>.",
        "output": "eval_report_lfm8b_mmlu_science_full.json",
        "max_new_tokens": 8096
    },
    {
        "name": "competition_math",
        "label": "Competition MATH (Hendrycks — 8,096 Tokens)",
        "type": "math",
        "system_prompt": "You are an expert mathematician. Solve the problem step by step and state the final answer clearly in \\boxed{...}.",
        "output": "eval_report_lfm8b_competition_math_full.json",
        "max_new_tokens": 8096
    },
    {
        "name": "arc",
        "label": "ARC-Challenge Science Reasoning",
        "type": "science",
        "system_prompt": "You are an expert in science. Solve the multiple-choice question and conclude with the letter option in the format: Answer: <Letter>.",
        "output": "eval_report_lfm8b_arc_full.json",
        "max_new_tokens": 4096
    }
]


class LFMStudioAdapter(OpenAIAPIAdapter):
    """LM Studio adapter for LFM 2.5 8B A1B MLX with stop token filtering and retries."""
    def __init__(self, model_name: str = MODEL_NAME, api_base: str = API_BASE, system_prompt: Optional[str] = None, timeout: float = 180.0):
        super().__init__(model_name=model_name, api_base=api_base, system_prompt=system_prompt, timeout=timeout)

    def generate(self, prompt: str, max_new_tokens: int = 8096, temperature: float = 0.0, stop_words: Optional[List[str]] = None, **kwargs) -> str:
        # In LM Studio, avoid premature "\n\n" termination
        if stop_words:
            stop_words = [s for s in stop_words if s != "\n\n"]

        for attempt in range(3):
            try:
                out = super().generate(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    stop_words=stop_words,
                    **kwargs
                )
                return out
            except Exception as e:
                time.sleep(1.0 + attempt * 2.0)
                if attempt == 2:
                    return ""
        return ""


def compile_master_summary(summary_results: Dict[str, Any]):
    master_file = LOGS_DIR / "eval_report_lfm8b_master_summary.json"
    with open(master_file, "w") as f:
        json.dump(summary_results, f, indent=2)

    print("\n" + "=" * 90)
    print("  MASTER BENCHMARK SCORECARD: LIQUID AI LFM 2.5 8B A1B MLX (LM STUDIO)")
    print(f"  Saved to: {master_file}")
    print("=" * 90)
    print(f"{'Benchmark Suite':<45} | {'Tasks':<6} | {'Pass@1 (%)':<10} | {'95% CI':<18} | {'AST Valid':<9}")
    print("-" * 95)
    for k, v in summary_results.items():
        ci_str = f"[{v['pass_at_1_95ci'][0]:.1f}%, {v['pass_at_1_95ci'][1]:.1f}%]" if v.get("pass_at_1_95ci") else "N/A"
        ast_str = f"{v.get('ast_validity_pct', 0.0):>8.1f}%" if v.get('ast_validity_pct') is not None else "N/A"
        print(f"{v['label']:<45} | {v['total_tasks']:>6} | {v['pass_at_1_pct']:>9.2f}% | {ci_str:<18} | {ast_str}")
    print("=" * 95 + "\n")


def main():
    print("\n" + "=" * 90)
    print(f"  LAUNCHING INSTITUTIONAL BENCHMARKS ON {MODEL_NAME.upper()} (LM STUDIO)")
    print(f"  Server Base: {API_BASE} | Reasoning Budget: 8,096 Tokens | Metal Acceleration")
    print("=" * 90 + "\n")

    summary_results = {}

    # Load any pre-existing suite reports
    for s in SUITES:
        suite_key = s["name"]
        out_file = LOGS_DIR / s["output"]
        b_type = s["type"]
        if out_file.exists():
            try:
                with open(out_file) as f:
                    cached_data = json.load(f)
                data_obj = cached_data.get(b_type, {})
                if "functional" in data_obj:
                    data_obj = data_obj["functional"]
                pass_pct = data_obj.get("pass_at_1_pct") if data_obj.get("pass_at_1_pct") is not None else data_obj.get("pass_rate_pct", 0.0)
                summary_results[suite_key] = {
                    "label": s["label"],
                    "total_tasks": data_obj.get("total_tasks"),
                    "pass_at_1_pct": pass_pct,
                    "pass_at_1_95ci": data_obj.get("pass_at_1_95ci") or data_obj.get("pass_rate_95ci"),
                    "ast_validity_pct": data_obj.get("ast_validity_pct") or data_obj.get("syntax_validity_pct"),
                    "execution_outcomes": data_obj.get("execution_outcomes"),
                    "report_file": str(out_file)
                }
            except Exception:
                pass

    for s in SUITES:
        suite_key = s["name"]
        label = s["label"]
        b_type = s["type"]
        out_file = LOGS_DIR / s["output"]
        max_tokens = s["max_new_tokens"]

        if out_file.exists():
            print(f"✓ Suite {label} already completed ({summary_results.get(suite_key, {}).get('total_tasks', '?')} tasks, Pass@1: {summary_results.get(suite_key, {}).get('pass_at_1_pct', '?')}%). Resuming next...")
            continue

        print(f"\n{'#' * 80}")
        print(f"  STARTING BENCHMARK: {label.upper()} ({suite_key})")
        print(f"  Target File: {out_file} | Reasoning Token Budget: {max_tokens}")
        print(f"{'#' * 80}\n")

        adapter = LFMStudioAdapter(
            model_name=MODEL_NAME,
            api_base=API_BASE,
            system_prompt=s["system_prompt"],
            timeout=180.0
        )

        t_start = time.time()
        report = evaluate(
            checkpoint=adapter,
            mode="functional",
            suite=suite_key,
            benchmark_type=b_type,
            max_tasks=None,
            max_new_tokens=max_tokens,
            output_path=str(out_file)
        )
        elapsed = time.time() - t_start

        # For tool-use, re-score with math equivalence and dict normalization
        if suite_key == "tooluse":
            with open(out_file) as f:
                d = json.load(f)
            t_tasks = d.get("tooluse", {}).get("tasks", [])
            p_cnt = 0
            for t in t_tasks:
                if t.get("tool_correct") and _compare_arguments(t.get("expected_args") or {}, t.get("parsed_args") or {}):
                    t["passed"] = True
                    p_cnt += 1
                else:
                    t["passed"] = False
            d["tooluse"]["pass_rate_pct"] = round((p_cnt / max(1, len(t_tasks))) * 100.0, 2)
            with open(out_file, "w") as f:
                json.dump(d, f, indent=2)
            report = d

        data_obj = report.get(b_type, {})
        if "functional" in data_obj:
            data_obj = data_obj["functional"]

        pass_at_1 = data_obj.get("pass_at_1_pct") if data_obj.get("pass_at_1_pct") is not None else data_obj.get("pass_rate_pct", 0.0)
        ci = data_obj.get("pass_at_1_95ci") or data_obj.get("pass_rate_95ci")
        ast_val = data_obj.get("ast_validity_pct") or data_obj.get("syntax_validity_pct")

        summary_results[suite_key] = {
            "label": label,
            "total_tasks": data_obj.get("total_tasks"),
            "pass_at_1_pct": pass_at_1,
            "pass_at_1_95ci": ci,
            "ast_validity_pct": ast_val,
            "execution_outcomes": data_obj.get("execution_outcomes"),
            "avg_token_length": data_obj.get("extended_metrics", {}).get("avg_token_length"),
            "elapsed_seconds": round(elapsed, 1),
            "report_file": str(out_file)
        }

        print(f"\n✓ Completed {label} in {elapsed/60.0:.2f} minutes (Pass@1: {pass_at_1}%).\n")
        compile_master_summary(summary_results)

    compile_master_summary(summary_results)
    print("\n" + "=" * 90)
    print("  ALL BENCHMARKS COMPLETED FOR LIQUID AI LFM 2.5 8B A1B MLX")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
