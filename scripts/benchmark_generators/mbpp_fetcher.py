"""
Fetches, sanitizes, and verifies the MBPP (Mostly Basic Python Problems) benchmark dataset.
"""

import urllib.request
import json
import re
from pathlib import Path
from typing import List, Dict, Any

MBPP_URL = "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/sanitized-mbpp.json"


def fetch_and_build_mbpp_suite(output_path: Path | None = None) -> List[Dict[str, Any]]:
    """
    Downloads sanitized MBPP dataset (427 tasks), structures it with signatures,
    and self-verifies every ground truth solution.
    """
    if output_path is None:
        output_path = Path(__file__).resolve().parents[2] / "evals" / "benchmarks" / "mbpp_suite.json"
    print("Fetching sanitized MBPP benchmark dataset...")
    try:
        req = urllib.request.urlopen(MBPP_URL, timeout=15)
        raw_data = json.loads(req.read().decode("utf-8"))
    except Exception as e:
        if output_path.exists():
            print(f"Network fetch failed ({e}); reading existing file at {output_path}")
            with open(output_path, "r", encoding="utf-8") as f:
                return json.load(f)
        raise RuntimeError(f"Failed to fetch MBPP dataset: {e}") from e

    formatted_tasks = []
    
    for item in raw_data:
        task_id = item["task_id"]
        prompt_text = item["prompt"].strip()
        code = item["code"].rstrip()
        test_imports = item.get("test_imports", [])
        test_list = item.get("test_list", [])

        # Determine entry point from test assertions
        fn_name = None
        for t in test_list:
            m = re.search(r"assert\s+([a-zA-Z0-9_]+)\s*\(", t)
            if not m:
                m = re.search(r"assert\s+not\s+([a-zA-Z0-9_]+)\s*\(", t)
            if m:
                fn_name = m.group(1)
                break

        lines = code.split("\n")
        def_idx = -1
        for idx, l in enumerate(lines):
            if l.strip().startswith(f"def {fn_name}"):
                def_idx = idx
                break
        if def_idx == -1:
            for idx, l in enumerate(lines):
                if l.strip().startswith("def "):
                    def_idx = idx
                    fn_name = l.split("(")[0].replace("def ", "").strip()
                    break

        prefix_lines = lines[:def_idx]
        def_line = lines[def_idx]
        body_lines = lines[def_idx + 1 :]

        # Detect body indentation
        indent_str = "    "
        for bl in body_lines:
            if bl.strip():
                leading = len(bl) - len(bl.lstrip())
                if leading > 0:
                    indent_str = bl[:leading]
                    break

        prompt = ""
        if prefix_lines:
            prompt += "\n".join(prefix_lines).strip() + "\n\n"
        prompt += def_line + "\n" + f'{indent_str}"""{prompt_text}"""\n'

        full_code = prompt + "\n".join(body_lines)
        test_harness = ""
        if test_imports:
            test_harness += "\n".join(test_imports) + "\n"
        test_harness += "\n".join(test_list) + "\n"

        # Verify reference code
        try:
            exec(full_code + "\n\n" + test_harness, {"__name__": "__main__"})
        except Exception as exc:
            raise RuntimeError(f"MBPP task {task_id} failed self-verification: {exc}") from exc

        task_obj = {
            "id": f"MBPP/{task_id}",
            "category": "MBPP",
            "name": fn_name or f"mbpp_task_{task_id}",
            "prompt": prompt,
            "ground_truth_solution": "\n" + "\n".join(body_lines),
            "test_harness": test_harness,
            "entry_point": fn_name,
        }
        formatted_tasks.append(task_obj)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(formatted_tasks, f, indent=2)

    print(f"[OK] Verified and wrote {len(formatted_tasks)} MBPP tasks -> {output_path}")
    return formatted_tasks
