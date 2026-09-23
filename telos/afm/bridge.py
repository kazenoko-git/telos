"""
AFM Bridge
Compiles the Swift FoundationModels bridge on first use and drives it over JSON IPC.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional, Sequence

__all__ = [
    "AFMUnavailableError",
    "AFMBridge",
    "bridge_source_path",
    "ensure_bridge_binary",
    "cache_root",
]


class AFMUnavailableError(RuntimeError):
    """Raised when the AFM bridge cannot be built or used."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def cache_root() -> Path:
    """Directory holding compiled bridge binaries; honours TELOS_CACHE_DIR."""
    override = os.environ.get("TELOS_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "telos"


def _swift_source_path() -> Path:
    import importlib.resources as resources

    ref = resources.files("telos.afm").joinpath("_swift", "afm_bridge.swift")
    return Path(str(ref))


def bridge_source_path() -> Path:
    """Path to the bundled Swift source."""
    return _swift_source_path()


def _bridge_digest(source: bytes) -> str:
    # macOS version included: an OS upgrade can change the linked framework ABI.
    h = hashlib.sha256()
    h.update(source)
    h.update(b"\0")
    h.update((platform.mac_ver()[0] or "unknown").encode())
    return h.hexdigest()[:16]


def ensure_bridge_binary(*, force: bool = False) -> Path:
    """Compiles the Swift bridge if needed and returns the binary path."""
    if platform.system() != "Darwin":
        raise AFMUnavailableError(f"Apple Foundation Models are macOS-only (this is {platform.system()}).")

    src = _swift_source_path()
    if not src.is_file():
        raise AFMUnavailableError(
            f"AFM bridge source is missing from the installed package (expected {src})."
        )

    target = cache_root() / f"afm_bridge-{_bridge_digest(src.read_bytes())}"
    if target.is_file() and os.access(target, os.X_OK) and not force:
        return target

    swiftc = shutil.which("swiftc")
    if swiftc is None:
        raise AFMUnavailableError(
            "The Swift compiler ('swiftc') was not found, so the Foundation Models "
            "bridge cannot be built. Install it with `xcode-select --install`."
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.with_suffix(".lock")
    tmp_path = target.with_suffix(f".tmp.{os.getpid()}")

    import fcntl

    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            # Another process may have finished the build while we waited.
            if target.is_file() and os.access(target, os.X_OK) and not force:
                return target

            cmd = [swiftc, "-O", "-parse-as-library", str(src), "-o", str(tmp_path)]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                raise AFMUnavailableError(
                    "Failed to compile the Swift AFM bridge.\n"
                    f"Command: {' '.join(cmd)}\n"
                    f"{res.stderr.strip()}"
                )
            os.chmod(tmp_path, 0o755)
            os.replace(tmp_path, target)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            tmp_path.unlink(missing_ok=True)

    return target


def query_status(*, build: bool = True) -> dict[str, Any]:
    """Runs `afm_bridge --status` and returns its parsed JSON report."""
    binary = ensure_bridge_binary() if build else _existing_binary()
    try:
        res = subprocess.run([str(binary), "--status"], capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return {"available": False, "check": "timeout", "detail": "The status probe timed out."}

    lines = res.stdout.strip().splitlines()
    if not lines:
        detail = (res.stderr or "").strip() or "The status probe produced no output."
        return {"available": False, "check": "no_output", "detail": detail}
    try:
        return json.loads(lines[0])
    except json.JSONDecodeError:
        return {"available": False, "check": "bad_output", "detail": lines[0][:200]}


def _existing_binary() -> Path:
    src = _swift_source_path()
    if not src.is_file():
        raise AFMUnavailableError(f"AFM bridge source is missing (expected {src}).")
    target = cache_root() / f"afm_bridge-{_bridge_digest(src.read_bytes())}"
    if not (target.is_file() and os.access(target, os.X_OK)):
        raise AFMUnavailableError("The AFM bridge has not been built yet.")
    return target


class AFMBridge:
    """Persistent connection to the Swift Foundation Models bridge."""

    DEFAULT_INSTRUCTIONS = (
        "You are an expert programming and reasoning assistant. "
        "Provide direct, concise, and accurate output without conversational filler."
    )

    def __init__(self, *, binary: Optional[Path] = None, timeout: float = 600.0):
        self._binary = Path(binary) if binary is not None else None
        self._timeout = timeout
        self._proc: Optional[subprocess.Popen] = None
        self._next_id = 1

    @property
    def binary(self) -> Path:
        if self._binary is None:
            self._binary = ensure_bridge_binary()
        return self._binary

    def start(self) -> None:
        """Starts the bridge process if it is not already running."""
        if self._proc is not None and self._proc.poll() is None:
            return
        self._proc = subprocess.Popen(
            [str(self.binary)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        stop: Optional[Sequence[str]] = None,
        instructions: Optional[str] = None,
    ) -> str:
        """Generates a completion, returning the raw text."""
        self.start()
        assert self._proc is not None and self._proc.stdin and self._proc.stdout

        req_id = self._next_id
        self._next_id += 1
        payload = {
            "id": req_id,
            "prompt": prompt,
            "instructions": instructions or self.DEFAULT_INSTRUCTIONS,
            "maxTokens": max_new_tokens,
            "temperature": temperature,
            "stopTokens": list(stop) if stop else None,
        }

        try:
            self._proc.stdin.write(json.dumps(payload) + "\n")
            self._proc.stdin.flush()
            line = self._proc.stdout.readline()
        except (BrokenPipeError, OSError) as exc:
            self.close()
            raise AFMUnavailableError(f"The AFM bridge terminated unexpectedly: {exc}") from exc

        if not line:
            self.close()
            raise AFMUnavailableError(
                "The AFM bridge closed its output stream. Run `telos afm status` to diagnose."
            )

        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AFMUnavailableError(f"The AFM bridge returned malformed JSON: {line[:200]!r}") from exc

        error = data.get("error")
        if error:
            raise AFMUnavailableError(str(error))
        return data.get("content", "")

    def close(self) -> None:
        """Terminates the bridge process."""
        proc, self._proc = self._proc, None
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            proc.kill()

    def __enter__(self) -> "AFMBridge":
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
