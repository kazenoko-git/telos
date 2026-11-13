"""OS-level launch helpers for evaluating untrusted generated code."""

from __future__ import annotations

import json
import os
import shutil
import sys
import sysconfig
from pathlib import Path


class SandboxUnavailable(RuntimeError):
    """Raised when this host has no supported OS sandbox backend."""


def sandbox_command(
    command: list[str],
    scratch_dir: str | Path,
    *,
    extra_executables: tuple[str, ...] = (),
) -> tuple[list[str], dict[str, str]]:
    """Wrap a command in a network-denying, scratch-only OS sandbox.

    Supported hosts must provide macOS ``sandbox-exec`` or Linux ``bwrap``.
    The function fails closed if neither backend is available.
    """
    scratch = Path(scratch_dir).resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    executable_paths = tuple(dict.fromkeys(
        executable
        for path in (command[0], *extra_executables)
        for executable in (str(Path(path).absolute()), str(Path(path).resolve()))
    ))
    safe_env = {
        "PATH": os.pathsep.join(dict.fromkeys(str(Path(path).parent) for path in executable_paths)),
        "HOME": str(scratch),
        "TMPDIR": str(scratch),
        "TEMP": str(scratch),
        "TMP": str(scratch),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }

    if sys.platform == "darwin":
        sandbox_exec = shutil.which("sandbox-exec")
        if not sandbox_exec:
            raise SandboxUnavailable("macOS sandbox-exec is unavailable")
        read_roots = {
            "/System/Library",
            "/usr/lib",
            "/usr/share",
            "/Library/Apple/usr/lib",
            str(Path(sysconfig.get_path("stdlib")).resolve()),
            str(Path(sysconfig.get_path("platstdlib")).resolve()),
            str(Path(sysconfig.get_config_var("LIBDIR") or "/usr/lib").resolve()),
            str(Path(sys.executable).resolve().parent),
            str(scratch),
        }
        for executable in executable_paths:
            read_roots.add(str(Path(executable).parent))
            read_roots.add(str(Path(executable).parent.parent))
        rules = ["(version 1)", "(deny default)", "(deny network*)"]
        if extra_executables:
            rules.append("(allow process-fork)")
        rules.extend(f"(allow file-read* (subpath {json.dumps(root)}))" for root in sorted(read_roots))
        rules.extend([
            '(allow file-read* (literal "/dev/null"))',
            '(allow file-write* (literal "/dev/null"))',
            f"(allow file-write* (subpath {json.dumps(str(scratch))}))",
        ])
        rules.extend(
            f"(allow process-exec (literal {json.dumps(path)}))"
            for path in executable_paths
        )
        return [sandbox_exec, "-p", " ".join(rules), *command], safe_env

    if sys.platform.startswith("linux"):
        bwrap = shutil.which("bwrap")
        if not bwrap:
            raise SandboxUnavailable("Linux bwrap is unavailable")
        wrapper = [
            bwrap,
            "--die-with-parent",
            "--new-session",
            "--unshare-all",
            "--ro-bind", "/usr", "/usr",
            "--ro-bind", "/bin", "/bin",
            "--dev", "/dev",
            "--proc", "/proc",
            "--tmpfs", "/tmp",
            "--bind", str(scratch), str(scratch),
            "--chdir", str(scratch),
        ]
        for root in ("/lib", "/lib64"):
            if Path(root).exists():
                wrapper.extend(["--ro-bind", root, root])
        for executable in executable_paths:
            resolved = Path(executable)
            if not any(str(resolved).startswith(root + os.sep) for root in ("/usr", "/bin", "/lib", "/lib64")):
                runtime_root = resolved.parent.parent if resolved.parent.name in {"bin", "sbin"} else resolved.parent
                if runtime_root.exists():
                    wrapper.extend(["--ro-bind", str(runtime_root), str(runtime_root)])
        return [*wrapper, *command], safe_env

    raise SandboxUnavailable(f"No supported OS sandbox backend for {sys.platform}")
