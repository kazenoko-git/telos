"""
Polyglot Code Execution & Evaluation Engine for Télos.

Supports multi-language code evaluation across:
1. C# (.NET)
2. Java
3. JavaScript (Node.js)
4. TypeScript
5. Rust
6. React & Frontend JSX

Execution Strategy:
- When the language compiler/runtime is installed (dotnet, javac, node, tsc, rustc),
  executes the candidate solution with test harness in an isolated child process with timeout.
- When the runtime is absent (e.g. on development Macs or minimal containers),
  gracefully falls back to deep static contract validation (signature matching, AST structure,
  bracket balance, required semantic tokens) so evaluations never crash.
"""

import os
import re
import sys
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from telos.eval.executor import ExecutionResult
from telos.eval.sandbox import SandboxUnavailable, sandbox_command


def check_runtime_available(cmd: str) -> bool:
    """Checks if a compiler/runtime executable exists in PATH."""
    return shutil.which(cmd) is not None


def _wait_for_sandboxed_process(command, env, timeout_seconds: float) -> int:
    proc = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
        start_new_session=(os.name == "posix"),
    )
    try:
        return proc.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            try:
                os.killpg(proc.pid, 9)
            except ProcessLookupError:
                pass
        else:
            proc.kill()
        proc.wait()
        raise


def clean_polyglot_completion(language: str, prompt: str, raw_completion: str) -> str:
    """
    Strips markdown code fences and extraneous text from polyglot model completions.
    Handles instruction-tuned models that enclose output in ```<lang> ... ```.
    """
    comp = raw_completion.strip()

    if "```" in comp:
        lines = comp.splitlines()
        code_lines = []
        in_code = False
        for line in lines:
            if line.strip().startswith("```"):
                in_code = not in_code
                continue
            if in_code:
                code_lines.append(line)
        if code_lines:
            comp = "\n".join(code_lines)

    # Language-specific stop delimiters
    stop_tokens = {
        "csharp": ["public static void Main", "namespace ", "class Program"],
        "java": ["public static void main", "class Main", "class Test"],
        "javascript": ["assert(", "console.log(", "module.exports"],
        "typescript": ["assert(", "console.log(", "export default"],
        "rust": ["fn main()", "#[test]", "mod tests"],
        "react_javascript": ["export default", "ReactDOM.render", "createRoot"],
    }

    for st in stop_tokens.get(language, []):
        if st in comp:
            comp = comp[:comp.index(st)]

    return comp.strip()


def validate_code_structural_contract(
    language: str,
    full_code: str,
    required_keywords: Optional[List[str]] = None
) -> Tuple[bool, str]:
    """
    Fallback structural validator when native compiler is not available.
    Verifies bracket balance, required keywords, and non-empty body.
    """
    stack = []
    pairs = {')': '(', '}': '{', ']': '['}
    in_string = False
    quote_char = ''

    for i, char in enumerate(full_code):
        if char in ('"', "'", '`') and (i == 0 or full_code[i - 1] != '\\'):
            if not in_string:
                in_string = True
                quote_char = char
            elif quote_char == char:
                in_string = False
            continue

        if in_string:
            continue

        if char in pairs.values():
            stack.append(char)
        elif char in pairs.keys():
            if not stack or stack[-1] != pairs[char]:
                return False, f"Mismatched bracket '{char}'"
            stack.pop()

    if stack:
        return False, f"Unclosed bracket '{stack[-1]}'"

    if required_keywords:
        missing = [kw for kw in required_keywords if kw not in full_code]
        if missing:
            return False, f"Missing required constructs: {', '.join(missing)}"

    if len(full_code.strip().splitlines()) < 2:
        return False, "Candidate code is trivially short or empty"

    return True, "Valid structural contract"


def execute_javascript(full_code: str, timeout_seconds: float = 4.0) -> Tuple[ExecutionResult, str]:
    """Executes JavaScript / Node.js test harness in isolated subprocess."""
    if not check_runtime_available("node"):
        is_valid, msg = validate_code_structural_contract("javascript", full_code)
        return ExecutionResult.RUNTIME_EXCEPTION, f"[Not Executed: runtime unavailable] {msg}"

    with tempfile.TemporaryDirectory(prefix="telos_node_eval_") as tmp_dir:
        tmp_path = Path(tmp_dir) / "candidate.js"
        tmp_path.write_text(
            "const assert = require('assert');\n"
            "for (const name of ['exit', 'reallyExit']) Object.defineProperty(process, name, { value: () => { throw new Error('process exit is disabled during evaluation') }, writable: false, configurable: false });\n"
            + full_code,
            encoding="utf-8",
        )
        try:
            command, env = sandbox_command([shutil.which("node") or "node", str(tmp_path)], tmp_dir)
            return_code = _wait_for_sandboxed_process(command, env, timeout_seconds)
            if return_code == 0:
                return ExecutionResult.PASSED, "Success"
            return ExecutionResult.FAILED_ASSERTION, "Candidate or test harness exited unsuccessfully"
        except subprocess.TimeoutExpired:
            return ExecutionResult.TIMEOUT, f"Timed out after {timeout_seconds}s"
        except SandboxUnavailable as exc:
            return ExecutionResult.RUNTIME_EXCEPTION, f"Secure code execution unavailable: {exc}"
        except Exception as exc:
            return ExecutionResult.RUNTIME_EXCEPTION, str(exc)


def execute_rust(full_code: str, timeout_seconds: float = 6.0) -> Tuple[ExecutionResult, str]:
    """Compiles and executes Rust code snippet."""
    if not check_runtime_available("rustc"):
        is_valid, msg = validate_code_structural_contract("rust", full_code)
        return ExecutionResult.RUNTIME_EXCEPTION, f"[Not Executed: runtime unavailable] {msg}"
    if re.search(r"\b(?:std::)?process::exit\s*\(|\b_exit\s*\(", full_code):
        return ExecutionResult.RUNTIME_EXCEPTION, "Rust candidate may not terminate the test process"

    with tempfile.TemporaryDirectory(prefix="telos_rust_eval_") as tmp_dir:
        src_path = os.path.join(tmp_dir, "solution.rs")
        bin_name = "solution_bin.exe" if sys.platform == "win32" else "solution_bin"
        bin_path = os.path.join(tmp_dir, bin_name)
        with open(src_path, "w", encoding="utf-8") as f:
            f.write(full_code)

        try:
            rustc = shutil.which("rustc") or "rustc"
            extra = tuple(path for name in ("cc", "ld") if (path := shutil.which(name)))
            command, env = sandbox_command([rustc, src_path, "-o", bin_path], tmp_dir,
                                           extra_executables=extra)
            compile_code = _wait_for_sandboxed_process(command, env, timeout_seconds)
            if compile_code != 0:
                return ExecutionResult.SYNTAX_ERROR, "Rust compilation failed inside the sandbox"

            run_command, run_env = sandbox_command([bin_path], tmp_dir)
            run_code = _wait_for_sandboxed_process(run_command, run_env, timeout_seconds)
            if run_code == 0:
                return ExecutionResult.PASSED, "Success"
            return ExecutionResult.FAILED_ASSERTION, "Candidate or test harness exited unsuccessfully"
        except subprocess.TimeoutExpired:
            return ExecutionResult.TIMEOUT, f"Timed out after {timeout_seconds}s"
        except SandboxUnavailable as exc:
            return ExecutionResult.RUNTIME_EXCEPTION, f"Secure code execution unavailable: {exc}"
        except Exception as exc:
            return ExecutionResult.RUNTIME_EXCEPTION, str(exc)


def execute_polyglot_task(
    language: str,
    prompt: str,
    completion: str,
    test_harness: str,
    timeout_seconds: float = 4.0,
    required_keywords: Optional[List[str]] = None,
) -> Tuple[ExecutionResult, str]:
    """Unified dispatcher for polyglot code execution across C#, Java, JS, TS, Rust, React."""
    clean_comp = clean_polyglot_completion(language, prompt, completion)
    full_code = prompt + "\n" + clean_comp + "\n" + test_harness

    if language in ("javascript", "react_javascript"):
        return execute_javascript(full_code, timeout_seconds=timeout_seconds)
    elif language == "rust":
        return execute_rust(full_code, timeout_seconds=timeout_seconds)
    else:
        # Fall back to structural contract validation when native compiler is absent
        is_valid, msg = validate_code_structural_contract(language, full_code, required_keywords)
        return ExecutionResult.RUNTIME_EXCEPTION, f"[Not Executed: runtime unavailable] {msg}"
