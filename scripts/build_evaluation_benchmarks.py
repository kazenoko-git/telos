"""
Comprehensive Benchmark Generator & Verifier for Télos.

Generates:
1. evals/benchmarks/private_unseen_suite.json:
   512 completely unique, non-duplicated Python functional challenges across 8 categories
   (64 unique problems per category). Every task has dual BASE and HINT prompts,
   verified reference solutions, and zero modulo-cycling boilerplate.

2. evals/benchmarks/private_unseen_base.json:
   The 512 tasks formatted with BASE prompts (clean specs, no algorithmic hints).

3. evals/benchmarks/private_unseen_hint.json:
   The 512 tasks formatted with HINT prompts (augmented with algorithmic guidance).

4. evals/benchmarks/mbpp_suite.json:
   Sanitized MBPP evaluation benchmark suite (427 tasks).

5. evals/benchmarks/contextual_probes_1000.json:
   1,000 completely unique syntactic context probes across 8 categories.
"""

import json
import hashlib
from pathlib import Path

from scripts.benchmark_generators.cat1_algorithms import build_algorithms_category
from scripts.benchmark_generators.cat2_data_structures import build_data_structures_category
from scripts.benchmark_generators.cat3_string_processing import build_string_processing_category
from scripts.benchmark_generators.cat4_oop import build_oop_category
from scripts.benchmark_generators.cat5_control_flow import build_control_flow_category
from scripts.benchmark_generators.cat6_builtins_iteration import build_builtins_iteration_category
from scripts.benchmark_generators.cat7_exceptions_context import build_exceptions_context_category
from scripts.benchmark_generators.cat8_typing_signatures import build_typing_signatures_category
from scripts.benchmark_generators.mbpp_fetcher import fetch_and_build_mbpp_suite
from scripts.benchmark_generators.polyglot_fetcher import fetch_and_build_all_polyglot_suites
from scripts.benchmark_generators.react_builder import build_react_component_suite
from scripts.benchmark_generators.math_science_fetcher import fetch_and_build_all_math_science
from scripts.benchmark_generators.cybersecurity_builder import generate_cybersecurity_suite
from scripts.benchmark_generators.tooluse_builder import generate_tooluse_suite

import argparse

BENCHMARK_DIR = Path(__file__).resolve().parents[1] / "evals" / "benchmarks"
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES = [
    ("Algorithms & Numerical", build_algorithms_category),
    ("Data Structures & Collections", build_data_structures_category),
    ("String & Text Processing", build_string_processing_category),
    ("Object-Oriented Programming", build_oop_category),
    ("Control Flow & Loops", build_control_flow_category),
    ("Built-ins & Iteration", build_builtins_iteration_category),
    ("Exception Handling & Context Managers", build_exceptions_context_category),
    ("Imports, Typing & Signatures", build_typing_signatures_category),
]


def generate_verified_private_unseen_suites():
    """
    Builds and verifies 512 genuinely distinct functional tasks across all 8 categories.
    Exports master suite, BASE suite, and HINT suite.
    """
    all_tasks = []
    base_tasks = []
    hint_tasks = []

    for cat_name, generator_fn in CATEGORIES:
        print(f"Generating & verifying category: {cat_name}...")
        cat_tasks = generator_fn()
        assert len(cat_tasks) == 64, f"Category {cat_name} must have 64 tasks, got {len(cat_tasks)}"

        for t in cat_tasks:
            task_id = f"task_{len(all_tasks):04d}"
            sha256_hash = hashlib.sha256((t["prompt_base"] + t["test"]).encode()).hexdigest()

            # Master task record containing both prompt variants
            master_item = {
                "id": task_id,
                "category": cat_name,
                "name": t["name"],
                "entry_point": t["name"],
                "prompt": t["prompt_base"],
                "prompt_base": t["prompt_base"],
                "prompt_hint": t["prompt_hint"],
                "hint": t["hint"],
                "ground_truth_solution": t["solution"],
                "test_harness": t["test"],
                "sha256": sha256_hash,
            }
            all_tasks.append(master_item)

            # Dedicated BASE item (clean spec without hints)
            base_item = {
                "id": task_id,
                "category": cat_name,
                "name": t["name"],
                "entry_point": t["name"],
                "prompt": t["prompt_base"],
                "ground_truth_solution": t["solution"],
                "test_harness": t["test"],
                "sha256": sha256_hash,
            }
            base_tasks.append(base_item)

            # Dedicated HINT item (with algorithmic hint)
            hint_item = {
                "id": task_id,
                "category": cat_name,
                "name": t["name"],
                "entry_point": t["name"],
                "prompt": t["prompt_hint"],
                "hint": t["hint"],
                "ground_truth_solution": t["solution"],
                "test_harness": t["test"],
                "sha256": sha256_hash,
            }
            hint_tasks.append(hint_item)

    assert len(all_tasks) == 512, f"Total tasks must be 512, got {len(all_tasks)}"

    # Save master suite
    master_file = BENCHMARK_DIR / "private_unseen_suite.json"
    with open(master_file, "w") as f:
        json.dump(all_tasks, f, indent=2)
    print(f"✓ Generated master suite ({len(all_tasks)} tasks) -> {master_file}")

    # Save BASE suite (512 tasks)
    base_file = BENCHMARK_DIR / "private_unseen_base.json"
    with open(base_file, "w") as f:
        json.dump(base_tasks, f, indent=2)
    print(f"✓ Generated BASE suite ({len(base_tasks)} tasks) -> {base_file}")

    # Save HINT suite (512 tasks)
    hint_file = BENCHMARK_DIR / "private_unseen_hint.json"
    with open(hint_file, "w") as f:
        json.dump(hint_tasks, f, indent=2)
    print(f"✓ Generated HINT suite ({len(hint_tasks)} tasks) -> {hint_file}")


def generate_truly_unique_contextual_probes_1000():
    """Generates 1,000 completely unique contextual probes across 8 categories."""
    probes = []
    
    for cat_idx, (cat_name, _) in enumerate(CATEGORIES):
        # Generate 125 completely distinct prompts per category
        for i in range(125):
            var_name = f"var_{cat_idx}_{i}"
            fn_name = f"proc_{cat_idx}_{i}"
            prompt = f"# Category: {cat_name}\ndef {fn_name}({var_name}: int) -> int:\n    result = {var_name} +"
            target = f" {i + 1}"
            suffix = f"\n    return (result * {i + 2}) % 10007\n"
            multi_target = f" {i + 1}\n    return (result * {i + 2}) % 10007"
            
            p = {
                "id": f"probe_{len(probes):04d}",
                "category": cat_name,
                "prompt": prompt,
                "prefix": prompt,
                "target": target.strip(),
                "target_bpe": "Ġ" + target.strip(),
                "suffix": suffix,
                "multi_token_target": multi_target,
            }
            probes.append(p)

    out_file = BENCHMARK_DIR / "contextual_probes_1000.json"
    with open(out_file, "w") as f:
        json.dump(probes, f, indent=2)
    print(f"✓ Generated {len(probes)} unique contextual probes -> {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Télos Universal Benchmark Suite Generator")
    parser.add_argument("--all", action="store_true", help="Build all benchmarks across all domains")
    parser.add_argument("--python", action="store_true", help="Build Python suites (Private Unseen & MBPP)")
    parser.add_argument("--polyglot", action="store_true", help="Build Polyglot suites (C#, Java, JS, TS, Rust)")
    parser.add_argument("--react", action="store_true", help="Build React Component suite")
    parser.add_argument("--math-science", action="store_true", help="Build GSM8K, ARC-Challenge & MATH suites")
    parser.add_argument("--cyber", action="store_true", help="Build Cybersecurity suite")
    parser.add_argument("--tooluse", action="store_true", help="Build Tool-Use function calling suite")
    args = parser.parse_args()

    # Default to python & probes if no specific flag passed
    build_all = args.all or not any([args.python, args.polyglot, args.react, args.math_science, args.cyber, args.tooluse])

    print("=" * 80)
    print("Building and Verifying Télos Evaluation Suites...")
    print("=" * 80)

    if build_all or args.python:
        print("\n--- [Python Suites] ---")
        generate_verified_private_unseen_suites()
        fetch_and_build_mbpp_suite(BENCHMARK_DIR / "mbpp_suite.json")
        generate_truly_unique_contextual_probes_1000()

    if build_all or args.polyglot:
        print("\n--- [Polyglot MultiPL-E Suites (C#, Java, JS, TS, Rust)] ---")
        fetch_and_build_all_polyglot_suites(BENCHMARK_DIR)

    if build_all or args.react:
        print("\n--- [React & Frontend Component Suite] ---")
        build_react_component_suite(BENCHMARK_DIR / "react_javascript_suite.json")

    if build_all or args.math_science:
        print("\n--- [Math & Science Reasoning Suites (GSM8K, ARC, MATH)] ---")
        fetch_and_build_all_math_science(BENCHMARK_DIR)

    if build_all or args.cyber:
        print("\n--- [Cybersecurity Auditing Suite] ---")
        generate_cybersecurity_suite(BENCHMARK_DIR / "cybersecurity_suite.json", target_count=50)

    if build_all or args.tooluse:
        print("\n--- [Tool-Use & Function Calling Suite] ---")
        generate_tooluse_suite(BENCHMARK_DIR / "tooluse_suite.json", target_count=100)

    print("\n" + "=" * 80)
    print("✓ All Télos benchmarks generated, verified, and ready!")
    print("=" * 80)


if __name__ == "__main__":
    main()

