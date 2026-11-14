#!/usr/bin/env python3
"""Merge constituent comparative run JSON files into unified comparative_512_run.json."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    r_dir = root / "reports/xerj_reader_2026-11-14"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v2",
        action="store_true",
        help="Merge v2 corrected run artifacts into comparative_512_v2_run.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional custom output path",
    )
    args = parser.parse_args()

    if args.v2 or (r_dir / "comparative_512_v2_local.json").exists():
        output_path = args.output or r_dir / "comparative_512_v2_run.json"
        sources = [
            r_dir / "comparative_512_v2_xerj.json",
            r_dir / "comparative_512_v2_local.json",
            r_dir / "comparative_512_qwen08b.json",
            r_dir / "comparative_512_qwen2b.json",
        ]
    else:
        output_path = args.output or r_dir / "comparative_512_run.json"
        sources = [
            r_dir / "comparative_512_xerj.json",
            r_dir / "comparative_512_local.json",
            r_dir / "comparative_512_qwen08b.json",
            r_dir / "comparative_512_qwen2b.json",
        ]

    combined = {"benchmark_meta": {}, "model_runs": {}}

    for src in sources:
        if not src.exists():
            print(f"Notice: source file not found (skipping): {src.name}")
            continue
        try:
            data = json.loads(src.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as err:
            print(f"Warning: could not read {src.name}: {err}")
            continue

        if "benchmark_meta" in data and not combined["benchmark_meta"]:
            combined["benchmark_meta"] = data["benchmark_meta"]

        for m_id, m_run in data.get("model_runs", {}).items():
            # Exclude Gemini from final suite per user instructions
            if m_id == "gemini-3.5-flash-lite":
                continue
            combined["model_runs"][m_id] = m_run

    combined.setdefault("benchmark_meta", {})["finished_utc"] = datetime.now(timezone.utc).isoformat()
    output_path.write_text(json.dumps(combined, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Merged {len(combined['model_runs'])} models into {output_path.name}")
    for m_id, r in combined["model_runs"].items():
        c_all = r.get("summary", {}).get("combined_512", {})
        print(f"  {m_id:30s}: {c_all.get('correct_count')}/{c_all.get('total_cases')} ({c_all.get('accuracy_pct')}%)")

    try:
        import sys
        sys.path.insert(0, str(r_dir))
        from run_comparative_suite import generate_markdown_report
        generate_markdown_report(combined, output_path.with_suffix(".md"))
    except Exception as err:
        print(f"Notice: markdown report generation skipped: {err}")


if __name__ == "__main__":
    main()
