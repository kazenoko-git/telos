"""
PyTorch-XLA device and runtime management utilities.
Caches device and topology singletons to prevent 'InitializeComputationClient() can only be called once'
fatal runtime assertions when training multiple phases or re-running in notebook/interactive environments.
Includes automated VFIO device lock reclamation for Kaggle and Google Cloud TPU VMs.
"""

import os
import sys
import time
import signal
import subprocess
from pathlib import Path

_CACHED_XLA_DEVICE = None
_CACHED_XLA_WORLD_SIZE = None
_CACHED_IS_MASTER = None


def is_tpu_environment() -> bool:
    """Non-intrusively checks if the current runtime environment is a Google Cloud TPU or TPU VM."""
    return bool(
        os.environ.get("PJRT_DEVICE", "").upper() == "TPU"
        or "TPU_NAME" in os.environ
        or "CLOUD_TPU_TASK_ID" in os.environ
        or "TPU_PROCESS_ADDRESSES" in os.environ
        or "COLAB_TPU_ADDR" in os.environ
    )


def is_xla_initialized() -> bool:
    """Returns True if the XLA device singleton has already been initialized in this process."""
    return _CACHED_XLA_DEVICE is not None


def is_interactive_notebook() -> bool:
    """Detects whether code is executing inside an interactive Jupyter/Colab/Kaggle notebook kernel."""
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is not None and "IPKernelApp" in getattr(ip, "config", {}):
            return True
    except Exception:
        pass
    return False


def reclaim_vfio_devices():
    """
    On Linux TPU VMs, hardware access is mediated through exclusive /dev/vfio character devices.
    If an earlier process crashed or was cancelled, it may linger in the background holding
    open file descriptors, which triggers 'open(/dev/vfio/X): Device or resource busy'.

    This function identifies and terminates any external stale processes holding /dev/vfio locks
    before PyTorch-XLA attempts to initialize its computation client.
    """
    vfio_dir = Path("/dev/vfio")
    if not vfio_dir.exists():
        return

    current_pid = os.getpid()
    parent_pid = os.getppid()
    stale_pids = set()

    # 1. Inspect /proc/*/fd to identify any processes holding /dev/vfio locks
    proc_dir = Path("/proc")
    if proc_dir.exists():
        for entry in proc_dir.iterdir():
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            if pid == current_pid or pid <= 1:
                continue

            fd_dir = entry / "fd"
            try:
                if not fd_dir.exists():
                    continue
                for fd_link in fd_dir.iterdir():
                    try:
                        target = os.readlink(str(fd_link))
                        if "/dev/vfio" in target:
                            if pid == parent_pid:
                                print(f"  [Hardware Warning] Parent process (PID {parent_pid}) holds /dev/vfio locks.")
                                print("  If running via shell in Kaggle/Colab, this causes contention. Restart session or use Python API.")
                            else:
                                stale_pids.add(pid)
                            break
                    except (OSError, IOError):
                        pass
            except (PermissionError, OSError):
                pass

    # 2. Terminate external stale processes holding the TPU hardware
    killed_any = False
    for pid in sorted(stale_pids):
        try:
            # Read command line for clean logging
            cmdline = ""
            cmd_file = proc_dir / str(pid) / "cmdline"
            if cmd_file.exists():
                cmdline = cmd_file.read_bytes().replace(b"\x00", b" ").decode(errors="replace").strip()
            print(f"  [Hardware] Reclaiming TPU hardware (/dev/vfio): terminating stale PID {pid} ({cmdline[:60]})...")
            os.kill(pid, signal.SIGKILL)
            killed_any = True
        except (ProcessLookupError, PermissionError):
            pass

    # 3. Fallback check via fuser if available
    try:
        res = subprocess.run(["fuser", "/dev/vfio/*"], capture_output=True, text=True, shell=True)
        out_pids = [int(p) for p in res.stdout.strip().split() if p.isdigit()]
        for p in out_pids:
            if p != current_pid and p != parent_pid and p not in stale_pids:
                try:
                    os.kill(p, signal.SIGKILL)
                    killed_any = True
                except Exception:
                    pass
    except Exception:
        pass

    # If stale processes were terminated, provide 1.5s for kernel vfio-pci driver to recycle descriptors
    if killed_any:
        time.sleep(1.5)


def get_xla_device():
    """
    Returns the XLA device singleton.
    Guarantees xm.xla_device() is called at most once per process lifetime.
    Proactively checks hardware availability to catch /dev/vfio lock contention early with clear diagnostics.
    """
    global _CACHED_XLA_DEVICE
    if _CACHED_XLA_DEVICE is not None:
        return _CACHED_XLA_DEVICE

    # Reclaim any stale device locks before initiating computation client
    reclaim_vfio_devices()

    import torch
    import torch_xla.core.xla_model as xm

    dev = xm.xla_device()

    # Eagerly probe the hardware to ensure VFIO is accessible
    # before dataset loading or model graph compilation begins.
    try:
        test_tensor = torch.zeros(1, device=dev)
        xm.mark_step()
    except RuntimeError as e:
        err_str = str(e)
        if "Device or resource busy" in err_str or "vfio" in err_str:
            diag = (
                "\n" + "=" * 76 + "\n"
                "  [CRITICAL ERROR] TPU Hardware Contention Detected (/dev/vfio/* busy)!\n"
                "  Another process or a previous notebook run is holding the TPU device lock.\n\n"
                "  To release the TPU devices in Kaggle or Google Cloud TPU VM:\n"
                "    1. Run this in a notebook cell or terminal:\n"
                "       !fuser -k -9 /dev/vfio/* 2>/dev/null || true\n"
                "    2. Or restart your Kaggle session: Session -> Restart Session\n"
                "    3. Or execute via Python in a single cell to reuse the device context:\n"
                "       from telos.train.cli import train\n"
                "       train(paradigm='corosred', phase='A', ...)\n"
                "       train(paradigm='corosred', phase='B', ...)\n"
                "=" * 76 + "\n"
            )
            sys.stderr.write(diag)
            sys.stderr.flush()

            # In CLI/standalone scripts, bypass Py_FinalizeEx to avoid C++ PrepareToExit() abort
            if not is_interactive_notebook():
                os._exit(1)
        raise

    _CACHED_XLA_DEVICE = dev
    return _CACHED_XLA_DEVICE


def get_xla_world_size() -> int:
    """
    Returns the XLA world size using the consistent xla_model client context.
    Avoids conflicting runtime client initializations.
    """
    global _CACHED_XLA_WORLD_SIZE
    if _CACHED_XLA_WORLD_SIZE is not None:
        return _CACHED_XLA_WORLD_SIZE

    try:
        import torch_xla.core.xla_model as xm
        _CACHED_XLA_WORLD_SIZE = xm.xrt_world_size()
    except Exception:
        _CACHED_XLA_WORLD_SIZE = 1
    return max(1, _CACHED_XLA_WORLD_SIZE)


def is_xla_master() -> bool:
    """Returns True if this process is the master ordinal (0)."""
    global _CACHED_IS_MASTER
    if _CACHED_IS_MASTER is not None:
        return _CACHED_IS_MASTER

    try:
        import torch_xla.core.xla_model as xm
        _CACHED_IS_MASTER = xm.is_master_ordinal()
    except Exception:
        _CACHED_IS_MASTER = True
    return _CACHED_IS_MASTER

