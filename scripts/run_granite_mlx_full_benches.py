"""
Comprehensive Institutional Benchmark Orchestrator: IBM Granite 4.2 3B MLX via LM Studio.

Evaluates Granite 4.2 3B MLX (serving at localhost:1234/v1) across all 7 institutional benchmark tracks:
1. OpenAI HumanEval (Python Code Synthesis — 164 tasks)
2. BFCL Tool-Use & Function Calling (30 tasks)
3. Cybersecurity OWASP/CWE Auditing (50 tasks)
4. GPQA Diamond (PhD Science Reasoning — 198 tasks, 1024-token budget)
5. MMLU Science & STEM (833 tasks)
6. Competition MATH (Hendrycks Olympiad Math — 700 tasks, 1024-token budget)
7. ARC-Challenge Science Reasoning (1,172 tasks)

Zero external cloud compute; pure local Metal acceleration on Apple Silicon.
Features resume capability (skips already-completed suites) and robust error handling.
"""

import json
import time
import sys
from pathlib import Path
from telos.eval.adapters.openai_api import OpenAIAPIAdapter
from telos.eval.runner import evaluate

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "granite-4.2-3b-mlx"
API_BASE = "http://localhost:1234/v1"


class GraniteLMStudioAdapter(OpenAIAPIAdapter):
    """
    Specialized OpenAI API Adapter for local Granite 4.2 MLX in LM Studio.
    Strips premature newline stops, captures deep reasoning content and boxed answers,
    and handles transient LM Studio stream iteration retries.
    """
    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.0, stop=None, **kwargs):
        # Remove '\n\n' from stop tokens if present to prevent premature truncation of multi-line blocks
        if stop and "\n\n" in stop:
            stop = [s for s in stop if s != "\n\n"]
        for attempt in range(3):
            try:
                messages = []
                if self.system_prompt:
                    messages.append({"role": "system", "content": self.system_prompt})
                messages.append({"role": "user", "content": prompt})

                payload = {
                    "model": self.model_name,
                    "messages": messages,
                    "max_tokens": max_new_tokens,
                    "temperature": temperature,
                }
                if stop:
                    payload["stop"] = stop

                resp = self._make_http_request("/chat/completions", payload)
                choices = resp.get("choices", [])
                if not choices:
                    return ""
                msg = choices[0].get("message", {})
                content = msg.get("content") or ""
                reasoning = msg.get("reasoning_content") or ""

                # Prefer boxed solution from reasoning if not present in content
                if "\\boxed{" in reasoning and "\\boxed{" not in content:
                    return f"{reasoning}\n\n{content}".strip()
                if not content.strip() and reasoning:
                    return reasoning.strip()
                return content
            except Exception as e:
                if attempt < 2:
                    time.sleep(1.5)
                    continue
                print(f"[Warning] LM Studio call failed after retries: {e}", flush=True)
                return ""


# Benchmark suite definitions
SUITES = [
    {
        "name": "humaneval",
        "label": "OpenAI HumanEval (Python Code)",
        "type": "code",
        "system_prompt": "You are an expert programming assistant. Complete the requested Python function directly.",
        "output": "eval_report_granite_mlx_humaneval_full.json",
        "max_new_tokens": 1024
    },
    {
        "name": "tooluse",
        "label": "BFCL Tool-Use & Function Calling",
        "type": "tooluse",
        "system_prompt": "You are an AI assistant with tool access. Call the tool using JSON format: {\"tool\": \"...\", \"arguments\": {...}}",
        "output": "eval_report_granite_mlx_tooluse_full.json",
        "max_new_tokens": 512
    },
    {
        "name": "cyber",
        "label": "Cybersecurity OWASP/CWE Auditing",
        "type": "cyber",
        "system_prompt": "You are an expert cybersecurity auditor. Identify the CWE, explain the vulnerability, and provide secure remediated code.",
        "output": "eval_report_granite_mlx_cyber_full.json",
        "max_new_tokens": 1024
    },
    {
        "name": "gpqa_diamond",
        "label": "GPQA Diamond (PhD Science — 1,024 Tokens)",
        "type": "science",
        "system_prompt": "You are an expert scientist. Derive the solution step by step and conclude with the correct letter option in the format: Answer: <Letter>.",
        "output": "eval_report_granite_mlx_gpqa_diamond_full.json",
        "max_new_tokens": 1024
    },
    {
        "name": "mmlu_science",
        "label": "MMLU Science & STEM",
        "type": "science",
        "system_prompt": "You are an expert in science and mathematics. Derive the solution and state the correct letter option in the format: Answer: <Letter>.",
        "output": "eval_report_granite_mlx_mmlu_science_full.json",
        "max_new_tokens": 1024
    },
    {
        "name": "competition_math",
        "label": "Competition MATH (Hendrycks — 1,024 Tokens)",
        "type": "math",
        "system_prompt": "You are an expert mathematician. Solve the problem step by step and state the final answer clearly in \\boxed{...}.",
        "output": "eval_report_granite_mlx_competition_math_full.json",
        "max_new_tokens": 1024
    },
    {
        "name": "arc",
        "label": "ARC-Challenge Science Reasoning",
        "type": "science",
        "system_prompt": "You are an expert in scientific reasoning. Answer the question by selecting the correct option.",
        "output": "eval_report_granite_mlx_arc_full.json",
        "max_new_tokens": 384
    },
]


def compile_master_summary(results: dict):
    """Compiles and writes the master summary scorecard across all completed suites."""
    master_path = LOGS_DIR / "eval_report_granite_mlx_master_summary.json"
    with open(master_path, "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "=" * 90)
    print(f"  MASTER BENCHMARK SCORECARD: IBM GRANITE 4.2 3B MLX (LM STUDIO)")
    print(f"  Saved to: {master_path}")
    print("=" * 90)
    print(f"{'Benchmark Suite':<45} | {'Tasks':<6} | {'Pass@1 (%)':<10} | {'95% CI':<18} | {'AST Valid':<9}")
    print("-" * 95)
    for k, v in results.items():
        ci = v.get("pass_at_1_95ci")
        ci_str = f"[{ci[0]:.1f}%, {ci[1]:.1f}%]" if ci else "N/A"
        ast_val = f"{v.get('ast_validity_pct', 0.0):.1f}%" if v.get("ast_validity_pct") is not None else "N/A"
        pass_pct = v.get("pass_at_1_pct", 0.0)
        total_tasks = v.get("total_tasks", "N/A")
        print(f"{v['label']:<45} | {str(total_tasks):>6} | {pass_pct:>9.2f}% | {ci_str:<18} | {ast_val:>9}")
    print("=" * 95 + "\n", flush=True)


def main():
    print("\n" + "=" * 90)
    print("  TÉLOS INSTITUTIONAL BENCHMARKS: IBM GRANITE 4.2 3B MLX (LM STUDIO)")
    print(f"  Server Endpoint: {API_BASE} | Model: {MODEL_NAME}")
    print("  Mode: Full Unconstrained Institutional Evaluation")
    print("=" * 90 + "\n", flush=True)

    summary_results = {}

    for s in SUITES:
        suite_name = s["name"]
        label = s["label"]
        b_type = s["type"]
        out_file = LOGS_DIR / s["output"]

        # Check if already completed and valid
        if out_file.exists():
            try:
                with open(out_file) as f:
                    cached_data = json.load(f)
                data_obj = cached_data.get(b_type, {})
                if "functional" in data_obj:
                    data_obj = data_obj["functional"]
                pass_score = data_obj.get("pass_at_1_pct")
                if pass_score is None:
                    pass_score = data_obj.get("pass_rate_pct", 0.0)
                if data_obj.get("total_tasks"):
                    print(f"[OK] Suite {label} already completed ({data_obj['total_tasks']} tasks, Pass@1: {pass_score:.2f}%). Resuming next...", flush=True)
                    summary_results[suite_name] = {
                        "label": label,
                        "total_tasks": data_obj.get("total_tasks"),
                        "pass_at_1_pct": pass_score,
                        "pass_at_1_95ci": data_obj.get("pass_at_1_95ci") or data_obj.get("pass_rate_95ci"),
                        "ast_validity_pct": data_obj.get("ast_validity_pct") or data_obj.get("syntax_validity_pct"),
                        "execution_outcomes": data_obj.get("execution_outcomes"),
                        "report_file": str(out_file)
                    }
                    continue
            except Exception:
                pass

        print("\n" + "#" * 90)
        print(f"  STARTING BENCHMARK: {label.upper()}")
        print(f"  Suite: {suite_name} | Type: {b_type} | Destination: {out_file}")
        print("#" * 90 + "\n", flush=True)

        adapter = GraniteLMStudioAdapter(
            model_name=MODEL_NAME,
            api_base=API_BASE,
            system_prompt=s["system_prompt"],
            timeout=120.0
        )

        t0 = time.time()
        try:
            report = evaluate(
                checkpoint=adapter,
                mode="functional",
                suite=suite_name,
                benchmark_type=b_type,
                max_tasks=None,
                max_new_tokens=s.get("max_new_tokens", 512),
                output_path=str(out_file),
            )
            elapsed = time.time() - t0

            data_obj = report.get(b_type, {})
            if "functional" in data_obj:
                data_obj = data_obj["functional"]

            pass_score = data_obj.get("pass_at_1_pct")
            if pass_score is None:
                pass_score = data_obj.get("pass_rate_pct", 0.0)

            ci = data_obj.get("pass_at_1_95ci") or data_obj.get("pass_rate_95ci")
            ast_val = data_obj.get("ast_validity_pct") or data_obj.get("syntax_validity_pct")

            summary_results[suite_name] = {
                "label": label,
                "total_tasks": data_obj.get("total_tasks"),
                "pass_at_1_pct": pass_score,
                "pass_at_1_95ci": ci,
                "ast_validity_pct": ast_val,
                "execution_outcomes": data_obj.get("execution_outcomes"),
                "avg_token_length": data_obj.get("repetition", {}).get("avg_token_length"),
                "elapsed_seconds": round(elapsed, 1),
                "report_file": str(out_file)
            }

            print(f"\n[OK] Completed {label} in {elapsed / 60:.2f} minutes (Pass@1: {pass_score:.2f}%).\n", flush=True)
            compile_master_summary(summary_results)

        except Exception as err:
            print(f"\n[ERROR] Benchmark {label} encountered error: {err}. Logging and continuing...\n", flush=True)
            summary_results[suite_name] = {
                "label": label,
                "error": str(err),
                "total_tasks": "ERROR",
                "pass_at_1_pct": 0.0,
                "elapsed_seconds": round(time.time() - t0, 1)
            }
            compile_master_summary(summary_results)

    print("\n==========================================================================================")
    print("  ALL BENCHMARKS COMPLETED FOR IBM GRANITE 4.2 3B MLX")
    print("==========================================================================================\n", flush=True)


if __name__ == "__main__":
    main()
