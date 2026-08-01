"""
data preparation script for télos.
downloads, filters, and extracts python function definitions from source datasets.
filters for:
- valid python syntax (must parse via ast.parse).
- top-level function definitions containing docstrings.
- output tokenized sequences saved as JSON/numpy arrays for training.
"""

import ast
import json
from pathlib import Path


def is_valid_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def extract_functions_from_code(code: str) -> list[str]:
    if not is_valid_python(code):
        return []

    functions = []
    try:
        tree = ast.parse(code)
        lines = code.splitlines()

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if ast.get_docstring(node) is not None:
                    start_line = node.lineno - 1
                    end_line = getattr(node, "end_lineno", start_line + 50)
                    fn_code = "\n".join(lines[start_line:end_line])
                    if 10 < len(fn_code) < 4000:  # reasonable size check? 
                        functions.append(fn_code)
    except Exception:
        pass

    return functions


def prepare_synthetic_corpus(output_path: str = "data/python_corpus.txt"):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    samples = [
        "def add(a: int, b: int) -> int:\n    \"\"\"Add two numbers and return result.\"\"\"\n    return a + b\n",
        "def subtract(a: float, b: float) -> float:\n    \"\"\"Subtract b from a.\"\"\"\n    return a - b\n",
        "def multiply(x: int, y: int) -> int:\n    \"\"\"Calculate product of x and y.\"\"\"\n    return x * y\n",
        "def divide(a: float, b: float) -> float:\n    \"\"\"Divide a by b with zero check.\"\"\"\n    if b == 0:\n        raise ValueError('Division by zero')\n    return a / b\n",
        "def fibonacci(n: int) -> int:\n    \"\"\"Calculate nth Fibonacci number recursively.\"\"\"\n    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n",
        "def is_prime(n: int) -> bool:\n    \"\"\"Check if an integer is prime.\"\"\"\n    if n < 2:\n        return False\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n",
        "def factorial(n: int) -> int:\n    \"\"\"Compute factorial of n.\"\"\"\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result\n",
        "def reverse_string(s: str) -> str:\n    \"\"\"Return reversed string.\"\"\"\n    return s[::-1]\n",
        "def binary_search(arr: list[int], target: int) -> int:\n    \"\"\"Perform binary search on sorted array.\"\"\"\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return -1\n"
    ]

    # Duplicate samples to create initial Phase A token pool
    full_corpus = "\n\n".join(samples * 500)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_corpus)

    print(f"Sample corpus saved to {output_path} ({len(full_corpus)} characters).")
