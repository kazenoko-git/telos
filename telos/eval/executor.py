"""OS-sandboxed execution of generated Python code for evaluation."""

from __future__ import annotations

import enum
import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from .sandbox import SandboxUnavailable, sandbox_command


class ExecutionResult(str, enum.Enum):
    """Outcomes distinguish model errors from harness and resource failures."""

    PASSED = "PASSED"
    FAILED_ASSERTION = "FAILED_ASSERTION"
    SYNTAX_ERROR = "SYNTAX_ERROR"
    TIMEOUT = "TIMEOUT"
    RUNTIME_EXCEPTION = "RUNTIME_EXCEPTION"
    MEMORY_EXCEEDED = "MEMORY_EXCEEDED"


_WORKER = r'''import json, os, resource, sys
payload = json.loads(sys.stdin.read())
limit = payload["mem_limit_bytes"]
try:
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
except (AttributeError, OSError, ValueError):
    pass
try:
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 * 1024, 16 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
except (AttributeError, OSError, ValueError):
    pass
os.chdir(payload["scratch_dir"])
code = payload["code"]
result = None
try:
    compiled = compile(code, "<eval_sandbox>", "exec")
except SyntaxError as exc:
    result = ("SYNTAX_ERROR", f"{type(exc).__name__}: {exc.msg}")
except MemoryError:
    result = ("MEMORY_EXCEEDED", "Memory limit exceeded during compilation")
if result is None:
    safe_stdout, safe_stderr = sys.stdout, sys.stderr
    with open(os.devnull, "w") as sink:
        sys.stdout = sys.stderr = sink
        try:
            exec(compiled, {"__name__": "__main__"})
            result = ("PASSED", "All assertions and statements passed")
        except AssertionError as exc:
            result = ("FAILED_ASSERTION", str(exc) or "Assertion failed")
        except MemoryError:
            result = ("MEMORY_EXCEEDED", "Process memory limit exceeded during execution")
        except BaseException as exc:
            result = ("RUNTIME_EXCEPTION", f"{type(exc).__name__}: {exc}")
        finally:
            sys.stdout, sys.stderr = safe_stdout, safe_stderr
print(json.dumps(result), flush=True)
'''


def execute_code_sandboxed(
    code: str,
    test_harness: str = "",
    timeout_seconds: float = 3.0,
    mem_limit_mb: int = 512,
    scratch_dir: Optional[str] = None,
) -> tuple[ExecutionResult, str]:
    """Run candidate code in a network-denying, scratch-only OS sandbox.

    Requires macOS ``sandbox-exec`` or Linux ``bwrap``. Evaluation fails
    closed when the platform sandbox is unavailable.
    """
    if timeout_seconds <= 0 or mem_limit_mb <= 0:
        raise ValueError("timeout_seconds and mem_limit_mb must be positive")
    full_code = f"{code}\n\n{test_harness}" if test_harness else code
    temporary = None
    if scratch_dir is None:
        temporary = tempfile.TemporaryDirectory(prefix="telos_eval_sandbox_")
        scratch = Path(temporary.name)
    else:
        scratch = Path(scratch_dir).resolve()
        scratch.mkdir(parents=True, exist_ok=True)

    try:
        command, env = sandbox_command(
            [sys.executable, "-I", "-S", "-c", _WORKER], scratch
        )
        payload = json.dumps({
            "code": full_code,
            "mem_limit_bytes": int(mem_limit_mb * 1024 * 1024),
            "scratch_dir": str(scratch),
        })
        proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            start_new_session=(os.name == "posix"),
        )
        try:
            stdout, stderr = proc.communicate(payload, timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                proc.kill()
            proc.communicate()
            return ExecutionResult.TIMEOUT, f"Execution exceeded hard timeout limit of {timeout_seconds}s"

        if proc.returncode == 0:
            try:
                result, details = json.loads(stdout)
                return ExecutionResult(result), details
            except (ValueError, TypeError):
                pass
        if proc.returncode == 0 and not stdout.strip():
            return ExecutionResult.RUNTIME_EXCEPTION, "Sandbox worker returned no result"
        return ExecutionResult.RUNTIME_EXCEPTION, (
            f"Sandbox worker failed (exit {proc.returncode}): {stderr[:300]}"
        )
    except SandboxUnavailable as exc:
        return ExecutionResult.RUNTIME_EXCEPTION, f"Secure code execution unavailable: {exc}"
    except OSError as exc:
        return ExecutionResult.RUNTIME_EXCEPTION, f"Could not start sandbox worker: {exc}"
    finally:
        if temporary is not None:
            temporary.cleanup()
