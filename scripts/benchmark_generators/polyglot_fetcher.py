"""
Polyglot Benchmark Dataset Fetcher & Sanitizer.

Fetches standardized MultiPL-E evaluation datasets for:
1. C# (.NET) -> evals/benchmarks/humaneval_csharp_suite.json
2. Java      -> evals/benchmarks/humaneval_java_suite.json
3. JavaScript -> evals/benchmarks/humaneval_js_suite.json
4. TypeScript -> evals/benchmarks/humaneval_ts_suite.json
5. Rust       -> evals/benchmarks/humaneval_rust_suite.json
"""

import urllib.request
import json
from pathlib import Path
from typing import List, Dict, Any

LANG_CONFIGS = {
    "csharp": {
        "config": "humaneval-cs",
        "output_filename": "humaneval_csharp_suite.json",
        "display_name": "C# (.NET)",
        "entry_point_key": "name",
    },
    "java": {
        "config": "humaneval-java",
        "output_filename": "humaneval_java_suite.json",
        "display_name": "Java",
        "entry_point_key": "name",
    },
    "javascript": {
        "config": "humaneval-js",
        "output_filename": "humaneval_js_suite.json",
        "display_name": "JavaScript",
        "entry_point_key": "name",
    },
    "typescript": {
        "config": "humaneval-ts",
        "output_filename": "humaneval_ts_suite.json",
        "display_name": "TypeScript",
        "entry_point_key": "name",
    },
    "rust": {
        "config": "humaneval-rs",
        "output_filename": "humaneval_rust_suite.json",
        "display_name": "Rust",
        "entry_point_key": "name",
    },
}


import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def fetch_multipl_e_language(lang_key: str, output_dir: Path) -> List[Dict[str, Any]]:
    """
    Fetches full benchmark dataset for a given language from nuprl/MultiPL-E.
    Uses direct Hugging Face Parquet endpoint for instantaneous downloads (<2s),
    falling back to paginated dataset-server API if needed.
    """
    cfg = LANG_CONFIGS[lang_key]
    out_file = output_dir / cfg["output_filename"]
    print(f"Fetching MultiPL-E benchmark for {cfg['display_name']} ({cfg['config']})...")

    all_raw_rows = []

    # Fast path: direct parquet via pandas
    if HAS_PANDAS:
        try:
            parquet_url = f"https://huggingface.co/api/datasets/nuprl/MultiPL-E/parquet/{cfg['config']}/test/0.parquet"
            df = pd.read_parquet(parquet_url)
            for _, row in df.iterrows():
                row_dict = row.to_dict()
                # Convert numpy arrays to lists for json serialization
                if isinstance(row_dict.get("stop_tokens"), (list, np.ndarray)):
                    row_dict["stop_tokens"] = [str(x) for x in row_dict["stop_tokens"]]
                all_raw_rows.append(row_dict)
        except Exception as p_err:
            print(f"  Parquet fetch failed ({p_err}), trying rows API fallback...")

    # Fallback: paginated rows API
    if not all_raw_rows:
        offset = 0
        limit = 100
        try:
            while True:
                url = f"https://datasets-server.huggingface.co/rows?dataset=nuprl%2FMultiPL-E&config={cfg['config']}&split=test&offset={offset}&limit={limit}"
                req = urllib.request.Request(url, headers={"User-Agent": "Télos-Benchmark-Fetcher/2.0"})
                with urllib.request.urlopen(req, timeout=25) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                rows = [item["row"] for item in payload.get("rows", [])]
                if not rows:
                    break
                all_raw_rows.extend(rows)
                offset += len(rows)
                if len(rows) < limit:
                    break
        except Exception as exc:
            if out_file.exists():
                print(f"  Warning: Network fetch failed ({exc}); reading existing file at {out_file}")
                with open(out_file, "r") as f:
                    return json.load(f)
            raise RuntimeError(f"Failed to fetch {cfg['display_name']} benchmark: {exc}") from exc

    # Format into standardized Télos benchmark schema
    formatted_tasks = []
    for idx, row in enumerate(all_raw_rows):
        task_name = row.get("name", f"task_{idx}")
        prompt = row.get("prompt", "")
        test_harness = row.get("tests", "")
        entry_point = row.get("entry_point", task_name)
        stop_tokens = row.get("stop_tokens", [])
        if isinstance(stop_tokens, (np.ndarray, list)):
            stop_tokens = [str(s) for s in stop_tokens]

        formatted_tasks.append({
            "id": f"{cfg['config']}/{idx}",
            "name": task_name,
            "language": lang_key,
            "category": cfg["display_name"],
            "prompt": prompt,
            "test_harness": test_harness,
            "entry_point": entry_point,
            "stop_tokens": stop_tokens,
            "doctests": row.get("doctests", ""),
        })

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(formatted_tasks, f, indent=2)

    print(f"  [OK] Saved {len(formatted_tasks)} {cfg['display_name']} tasks -> {out_file}")
    return formatted_tasks


def fetch_and_build_all_polyglot_suites(benchmark_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Fetches all 5 polyglot benchmark suites (C#, Java, JS, TS, Rust)."""
    results = {}
    for lang in LANG_CONFIGS:
        results[lang] = fetch_multipl_e_language(lang, benchmark_dir)
    return results


if __name__ == "__main__":
    b_dir = Path(__file__).resolve().parents[2] / "evals" / "benchmarks"
    fetch_and_build_all_polyglot_suites(b_dir)

