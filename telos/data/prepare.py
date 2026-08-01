"""online data oreparation script for télos.

streams real Python source code online from HuggingFace Datasets, filters for valid AST
syntax, extracts clean function definitions with docstrings, and stops automatically once the
target token count is reached.
"""

import ast
import warnings
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset


def extract_functions_from_code(code: str) -> list[str]:
    """Single-pass function extractor: parses AST once and extracts docstring functions."""
    if "def " not in code:
        return []

    functions = []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(code)

        lines = None
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Only include functions that contain docstrings for high quality
                if ast.get_docstring(node) is not None:
                    if lines is None:
                        lines = code.splitlines()
                    start_line = node.lineno - 1
                    end_line = getattr(node, "end_lineno", start_line + 60)
                    fn_code = "\n".join(lines[start_line:end_line])
                    if 20 <= len(fn_code) <= 4000:
                        functions.append(fn_code)
    except Exception:
        pass

    return functions


def prepare_online_corpus(
    output_path: str = "data/python_corpus.txt",
    target_tokens: int = 30_000_000,
    dataset_name: str = "codeparrot/codeparrot-clean"
):
    """Streams online Python code dataset, extracts valid function blocks, and writes to file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    target_chars = target_tokens * 4

    print(f"Streaming Python code from HuggingFace dataset '{dataset_name}'...")
    print(f"Target token budget: {target_tokens:,} tokens (~{target_chars / 1e6:.1f} MB text)...")

    import os
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    ds_kwargs = {"streaming": True, "split": "train"}
    if token:
        print(f"Using HuggingFace authentication token for rate-limit bypass...")
        ds_kwargs["token"] = token

    ds = load_dataset(dataset_name, **ds_kwargs)

    written_chars = 0
    extracted_functions_count = 0
    buffer = []
    buffer_chars = 0

    pbar = tqdm(total=target_chars, unit="char", unit_scale=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for sample in ds:
            code_text = sample.get("content") or sample.get("code") or ""
            if not code_text or len(code_text) < 30:
                continue

            # Fast single-pass AST function extraction
            functions = extract_functions_from_code(code_text)
            for fn_code in functions:
                block = fn_code.strip() + "\n\n"
                buffer.append(block)
                fn_len = len(block)
                buffer_chars += fn_len
                written_chars += fn_len
                extracted_functions_count += 1
                pbar.update(fn_len)

                # Flush buffer to disk in 64KB chunks
                if buffer_chars >= 65536:
                    f.write("".join(buffer))
                    buffer.clear()
                    buffer_chars = 0

                if written_chars >= target_chars:
                    break

            if written_chars >= target_chars:
                break

        if buffer:
            f.write("".join(buffer))
            buffer.clear()

    pbar.close()
    approx_tokens = written_chars // 4
    print(f"Dataset preparation complete! Saved {extracted_functions_count:,} functions "
          f"({written_chars / 1e6:.2f} MB, ~{approx_tokens:,} tokens) to {output_path}")
