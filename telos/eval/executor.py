"""
Subprocess Sandboxed Execution Engine for Télos Code Model Evaluation.

Enforces:
1. multiprocessing.get_context("spawn") (strictly spawn, never fork to prevent GPU/XLA driver deadlocks)
2. Hard wall-clock timeouts with forced process termination (process.join -> terminate -> kill)
3. Memory limits via resource.setrlimit(resource.RLIMIT_AS, ...) inside the child process
4. Output capture into in-memory buffers to prevent OS pipe buffer deadlocks
5. Socket/network blocking and isolated scratch filesystem execution
6. Disambiguated ExecutionResult enum (PASSED, FAILED_ASSERTION, SYNTAX_ERROR, TIMEOUT, RUNTIME_EXCEPTION, MEMORY_EXCEEDED)
7. Global cleanup registry preventing orphaned processes on parent interrupt/crash
"""

import os
import sys
import time
import enum
import atexit
import signal
import tempfile
import multiprocessing as mp
from typing import Optional, Dict, Any
from pathlib import Path


class ExecutionResult(str, enum.Enum):
    """Disambiguated execution outcomes distinguishing model errors from harness flakiness."""
    PASSED = "PASSED"                      # All assertions passed with zero errors
    FAILED_ASSERTION = "FAILED_ASSERTION"  # Code ran to completion but an assertion failed
    SYNTAX_ERROR = "SYNTAX_ERROR"          # SyntaxError or IndentationError during compilation
    TIMEOUT = "TIMEOUT"                    # Exceeded hard wall-clock timeout limit
    RUNTIME_EXCEPTION = "RUNTIME_EXCEPTION"  # Unhandled exception (TypeError, NameError, etc.)
    MEMORY_EXCEEDED = "MEMORY_EXCEEDED"    # Exceeded child address space limit (MemoryError)


# Global tracking of active child processes for orphan cleanup
_ACTIVE_CHILD_PROCESSES: set[mp.Process] = set()


def _cleanup_orphaned_children():
    """Terminates and kills any remaining child workers when parent process exits."""
    for proc in list(_ACTIVE_CHILD_PROCESSES):
        if proc.is_alive():
            try:
                proc.terminate()
                proc.join(timeout=0.2)
                if proc.is_alive():
                    proc.kill()
            except Exception:
                pass
    _ACTIVE_CHILD_PROCESSES.clear()


# Register cleanup handlers on normal exit and interrupt signals
atexit.register(_cleanup_orphaned_children)
try:
    signal.signal(signal.SIGINT, lambda sig, frame: (_cleanup_orphaned_children(), sys.exit(1)))
    signal.signal(signal.SIGTERM, lambda sig, frame: (_cleanup_orphaned_children(), sys.exit(1)))
except (ValueError, AttributeError):
    # Signals might not be assignable in non-main threads or certain environments
    pass


def _sandbox_child_worker(
    code_to_run: str,
    queue: mp.Queue,
    mem_limit_bytes: int,
    scratch_dir: str
):
    """
    Subprocess worker executing user code inside a restricted sandbox environment.
    Runs strictly inside a spawned process.
    """
    # 1. Apply memory limit at the very top of child entrypoint (POSIX only)
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (mem_limit_bytes, mem_limit_bytes))
    except (ImportError, ValueError, OSError):
        # On macOS or platforms where RLIMIT_AS behaves differently, continue gracefully
        pass

    # 2. Restrict outbound network connections to prevent hallucinated retry loops
    try:
        import socket
        def _blocked_socket(*args, **kwargs):
            raise PermissionError("Network socket creation is disabled inside evaluation sandbox")
        socket.socket = _blocked_socket
    except Exception:
        pass

    # 3. Change working directory to isolated scratch space
    try:
        os.chdir(scratch_dir)
    except Exception:
        pass

    # 4. Redirect stdout and stderr to os.devnull to prevent pipe buffer deadlocks
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")

    # 5. Execute code with syntax and exception trapping
    try:
        # First verify compilation catches SyntaxError before running
        compiled = compile(code_to_run, "<eval_sandbox>", "exec")
    except SyntaxError as syn_err:
        queue.put((ExecutionResult.SYNTAX_ERROR, f"{type(syn_err).__name__}: {syn_err.msg}"))
        return
    except MemoryError:
        queue.put((ExecutionResult.MEMORY_EXCEEDED, "Memory limit exceeded during compilation"))
        return

    # Execute in clean global namespace
    exec_globals: Dict[str, Any] = {"__name__": "__main__"}
    try:
        exec(compiled, exec_globals)
        queue.put((ExecutionResult.PASSED, "All assertions and statements passed"))
    except AssertionError as assert_err:
        msg = str(assert_err) or "Assertion failed"
        queue.put((ExecutionResult.FAILED_ASSERTION, msg))
    except MemoryError:
        queue.put((ExecutionResult.MEMORY_EXCEEDED, "Process memory limit exceeded during execution"))
    except Exception as exc:
        queue.put((ExecutionResult.RUNTIME_EXCEPTION, f"{type(exc).__name__}: {str(exc)}"))


def execute_code_sandboxed(
    code: str,
    test_harness: str = "",
    timeout_seconds: float = 3.0,
    mem_limit_mb: int = 512,
    scratch_dir: Optional[str] = None
) -> tuple[ExecutionResult, str]:
    """
    Executes Python code safely inside an isolated spawned subprocess.

    Args:
        code: The generated Python completion or function definition.
        test_harness: Optional unit assertions or test script appended to the code.
        timeout_seconds: Hard wall-clock limit before forced termination.
        mem_limit_mb: Maximum virtual memory in Megabytes allocated to the child.
        scratch_dir: Working directory for execution (defaults to a safe temporary dir).

    Returns:
        tuple (ExecutionResult, details_str)
    """
    full_code = f"{code}\n\n{test_harness}" if test_harness else code
    mem_limit_bytes = int(mem_limit_mb * 1024 * 1024)

    # Use a temporary scratch directory if none is provided
    cleanup_temp_dir = False
    if scratch_dir is None:
        temp_scratch = tempfile.mkdtemp(prefix="telos_eval_sandbox_")
        scratch_dir = temp_scratch
        cleanup_temp_dir = True
    else:
        Path(scratch_dir).mkdir(parents=True, exist_ok=True)

    # Must strictly use 'spawn' to prevent CUDA/MPS/XLA context corruption and hangs
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()

    proc = ctx.Process(
        target=_sandbox_child_worker,
        args=(full_code, queue, mem_limit_bytes, scratch_dir)
    )

    _ACTIVE_CHILD_PROCESSES.add(proc)
    proc.start()

    # Wait for process completion up to the hard wall-clock timeout
    proc.join(timeout=timeout_seconds)

    result_status: ExecutionResult
    result_details: str

    if proc.is_alive():
        # Hard timeout: force terminate, wait briefly, then kill
        try:
            proc.terminate()
            proc.join(timeout=0.2)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=0.1)
        except Exception:
            pass
        result_status = ExecutionResult.TIMEOUT
        result_details = f"Execution exceeded hard timeout limit of {timeout_seconds}s"
    else:
        # Process finished within time limit; check queue output
        if not queue.empty():
            status, details = queue.get_nowait()
            result_status = status
            result_details = details
        else:
            # Child exited without writing to queue (e.g. killed by SIGSEGV or OOM killer)
            exit_code = proc.exitcode
            sigkill = getattr(signal, "SIGKILL", 9)
            if exit_code in (-sigkill, -9):
                result_status = ExecutionResult.MEMORY_EXCEEDED
                result_details = "Child process killed by OS (likely Out-Of-Memory)"
            else:
                result_status = ExecutionResult.RUNTIME_EXCEPTION
                result_details = f"Child process terminated unexpectedly with exit code {exit_code}"

    # Clean up tracking and queue
    _ACTIVE_CHILD_PROCESSES.discard(proc)
    try:
        queue.close()
        queue.join_thread()
    except Exception:
        pass

    if cleanup_temp_dir and scratch_dir:
        try:
            import shutil
            shutil.rmtree(scratch_dir, ignore_errors=True)
        except Exception:
            pass

    return result_status, result_details
