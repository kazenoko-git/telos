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


def is_valid_python(code: str) -> bool:
    """Checks if raw string parses cleanly as valid Python AST."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ast.parse(code)
        return True
    except (SyntaxError, ValueError, Exception):
        return False


def extract_functions_from_code(code: str) -> list[str]:
    """Extracts top-level and class method functions with docstrings from Python source code."""
    if not is_valid_python(code):
        return []

    functions = []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(code)

        lines = code.splitlines()

        for node in ast.walk(tree):
            # Matches top-level functions, class methods, static methods, and async defs
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if ast.get_docstring(node) is not None:
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
    """streams online Python code dataset, extracts valid function blocks, and writes to file.

    args:
        output_path: target output text file path.
        target_tokens: target total tokens (approx 4 characters per token).
        dataset_name: HuggingFace dataset name to stream from.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # convert target tokens to target character count (approx 1 token ~ 4 chars)
    target_chars = target_tokens * 4

    print(f"Streaming Python code from HuggingFace dataset '{dataset_name}'...")
    print(f"Target token budget: {target_tokens:,} tokens (~{target_chars / 1e6:.1f} MB text)...")

    # load dataset in streaming mode so no giant downloads are needed upfront
    ds = load_dataset(dataset_name, streaming=True, split="train")

    written_chars = 0
    extracted_functions_count = 0

    pbar = tqdm(total=target_chars, unit="char", unit_scale=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for sample in ds:
            # extract raw python code string from dataset column
            code_text = sample.get("content") or sample.get("code") or ""
            if not code_text or not is_valid_python(code_text):
                continue

            # extract AST-valid functions with docstrings
            functions = extract_functions_from_code(code_text)
            for fn_code in functions:
                formatted_block = fn_code.strip() + "\n\n"
                f.write(formatted_block)

                fn_len = len(formatted_block)
                written_chars += fn_len
                extracted_functions_count += 1
                pbar.update(fn_len)

                # stop automatically when target token/character budget is reached
                if written_chars >= target_chars:
                    break

            if written_chars >= target_chars:
                break

    pbar.close()
    approx_tokens = written_chars // 4
    print(f"Dataset preparation complete! Saved {extracted_functions_count:,} functions "
          f"({written_chars / 1e6:.2f} MB, ~{approx_tokens:,} tokens) to {output_path}")
