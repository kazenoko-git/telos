"""
Re-evaluates IBM Granite 4.2 3B MLX on GPQA Diamond (198 tasks) and MMLU Science (833 tasks)
with 1,024-token budget, reasoning content extraction, and updated MCQ letter parser.
"""
import json
import time
from pathlib import Path
from telos.eval.runner import evaluate
from scripts.run_granite_mlx_full_benches import GraniteLMStudioAdapter, MODEL_NAME, API_BASE

LOGS_DIR = Path("logs")

SUITES = [
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
    }
]


def main():
    print("\n" + "=" * 85)
    print("  RE-RUNNING SCIENCE BENCHMARKS: IBM GRANITE 4.2 3B MLX")
    print("  GPQA Diamond (198 tasks) + MMLU Science (833 tasks) | 1,024-Token Tier")
    print("=" * 85 + "\n", flush=True)

    for s in SUITES:
        label = s["label"]
        suite_name = s["name"]
        b_type = s["type"]
        out_file = LOGS_DIR / s["output"]

        print(f"\n>>> Starting {label} ({suite_name})...\n", flush=True)
        adapter = GraniteLMStudioAdapter(
            model_name=MODEL_NAME,
            api_base=API_BASE,
            system_prompt=s["system_prompt"],
            timeout=120.0
        )

        t0 = time.time()
        rep = evaluate(
            checkpoint=adapter,
            mode="functional",
            suite=suite_name,
            benchmark_type=b_type,
            max_tasks=None,
            max_new_tokens=s["max_new_tokens"],
            output_path=str(out_file)
        )
        elapsed = time.time() - t0
        data_obj = rep.get(b_type, {})
        if "functional" in data_obj:
            data_obj = data_obj["functional"]
        score = data_obj.get("pass_at_1_pct", 0.0)
        print(f"\n[OK] Completed {label} in {elapsed/60.0:.2f} mins (Pass@1: {score:.2f}%).\n", flush=True)

    print("\n>>> Recompiling master scorecard...\n", flush=True)
    from scripts.compile_final_comparative_scorecard import main as compile_main
    compile_main()
    print("[OK] All re-evaluations complete and master scorecard updated!\n", flush=True)


if __name__ == "__main__":
    main()
