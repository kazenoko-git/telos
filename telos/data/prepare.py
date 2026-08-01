"""Data preparation script for télos.

Supports two modes:
  --raw:  Streams raw Python code directly (no AST parsing). Maximum throughput.
  (default): Extracts AST-valid functions with docstrings for high-quality data.

Supports two download strategies:
  --fast: Downloads full dataset in parallel first, then iterates locally at SSD speed.
  (default): Streams dataset over single HTTP connection (slower, lower disk usage).
"""

import ast
import os
import warnings
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset


def _resolve_hf_token() -> str | None:
    """Resolves HuggingFace token from environment variables or .env file."""
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
    )
    if not token and Path(".env").exists():
        with open(".env", "r") as env_file:
            for line in env_file:
                if line.startswith("HF_TOKEN="):
                    token = line.split("=", 1)[1].strip().strip("'\"")
                    break
    return token


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
    dataset_name: str = "codeparrot/codeparrot-clean",
    raw_mode: bool = False,
    fast_mode: bool = False,
):
    """Prepares a Python code training corpus from HuggingFace datasets.

    Args:
        output_path: Target output text file path.
        target_tokens: Target total tokens (approx 4 characters per token).
        dataset_name: HuggingFace dataset name.
        raw_mode: If True, writes raw code without AST parsing (max throughput).
        fast_mode: If True, downloads full dataset in parallel first (faster iteration).
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    target_chars = target_tokens * 4
    mode_label = "RAW (no AST)" if raw_mode else "AST-filtered (docstring functions)"
    download_label = "PARALLEL download" if fast_mode else "streaming"

    print(f"Preparing corpus from '{dataset_name}' [{mode_label}, {download_label}]")
    print(f"Target: {target_tokens:,} tokens (~{target_chars / 1e6:.1f} MB text)")

    # Resolve HuggingFace auth token
    token = _resolve_hf_token()
    ds_kwargs = {"split": "train"}
    if token:
        print("Using HuggingFace authentication token for faster downloads...")
        ds_kwargs["token"] = token

    if fast_mode:
        # Non-streaming: downloads all parquet shards in parallel, then iterates locally
        print("Downloading full dataset in parallel (this may take a few minutes)...")
        ds = load_dataset(dataset_name, **ds_kwargs)
    else:
        # Streaming: single HTTP connection, lower disk usage but slower
        ds_kwargs["streaming"] = True
        ds = load_dataset(dataset_name, **ds_kwargs)

    written_chars = 0
    item_count = 0
    buffer = []
    buffer_chars = 0

    pbar = tqdm(total=target_chars, unit="char", unit_scale=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for sample in ds:
            code_text = sample.get("content") or sample.get("code") or ""
            if not code_text or len(code_text) < 30:
                continue

            if raw_mode:
                # Raw mode: write entire file contents directly (no AST parsing)
                block = code_text.strip() + "\n\n"
                buffer.append(block)
                fn_len = len(block)
                buffer_chars += fn_len
                written_chars += fn_len
                item_count += 1
                pbar.update(fn_len)
            else:
                # AST mode: extract only docstring functions
                functions = extract_functions_from_code(code_text)
                for fn_code in functions:
                    block = fn_code.strip() + "\n\n"
                    buffer.append(block)
                    fn_len = len(block)
                    buffer_chars += fn_len
                    written_chars += fn_len
                    item_count += 1
                    pbar.update(fn_len)

                    if written_chars >= target_chars:
                        break

            # Flush buffer to disk in 256KB chunks for fast sequential writes
            if buffer_chars >= 262144:
                f.write("".join(buffer))
                buffer.clear()
                buffer_chars = 0

            if written_chars >= target_chars:
                break

        # Flush remaining buffer
        if buffer:
            f.write("".join(buffer))
            buffer.clear()

    pbar.close()
    approx_tokens = written_chars // 4
    label = "files" if raw_mode else "functions"
    print(f"Dataset preparation complete! Saved {item_count:,} {label} "
          f"({written_chars / 1e6:.2f} MB, ~{approx_tokens:,} tokens) to {output_path}")
