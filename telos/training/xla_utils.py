"""
PyTorch-XLA device and runtime management utilities.
Caches device and topology singletons to prevent 'InitializeComputationClient() can only be called once'
fatal runtime assertions when training multiple phases or re-running in notebook/interactive environments.

Includes passive pre-flight VFIO accessibility checks for Kaggle and Google Cloud TPU VMs.
IMPORTANT: This module intentionally does NOT kill external processes or eagerly probe
TPU hardware with tensor operations, because:
  - Killing a process that held /dev/vfio does NOT reset the TPU chip's internal state
    (HBM, command queues, DMA descriptors). Subsequent tensor ops SIGSEGV on corrupted state.
  - Eagerly running torch.zeros(1, device=xla) + xm.mark_step() triggers ExecuteReplicated()
    inside PJRT's Eigen thread pool. On corrupted hardware this causes a cascading SIGSEGV
    across all worker threads — far worse than a clean Python error.
The only reliable TPU hardware reset is a full session restart (Kaggle) or reboot (GCE VM).
"""

import os
import sys
import errno
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


def _is_interactive_notebook() -> bool:
    """Detects whether code is executing inside an interactive Jupyter/Colab/Kaggle notebook kernel."""
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is not None and "IPKernelApp" in getattr(ip, "config", {}):
            return True
    except Exception:
        pass
    return False


def _preflight_check_vfio():
    """
    Passive OS-level pre-flight check for TPU VFIO device accessibility.

    On Linux TPU VMs (Kaggle, GCE), each TPU chip is exposed as an exclusive
    /dev/vfio/<N> IOMMU group device. The kernel's open() syscall returns EBUSY
    if another process already holds the file descriptor.

    This function tests accessibility at the syscall level BEFORE torch_xla
    attempts PJRT client initialization. If devices are locked, it prints
    clear diagnostics and exits cleanly — avoiding both:
      1. The C++ 'InitializeComputationClient() can only be called once' abort
      2. SIGSEGV crashes from ExecuteReplicated() on corrupted TPU hardware state

    This is intentionally a READ-ONLY check. It does NOT kill processes, because
    killing a VFIO holder does not reset TPU chip state and causes worse crashes.
    """
    vfio_dir = Path("/dev/vfio")
    if not vfio_dir.exists():
        return  # Not on a TPU VM (macOS, non-TPU Linux, etc.)

    busy_devices = []
    for dev_node in sorted(vfio_dir.iterdir()):
        # Only check IOMMU group nodes (/dev/vfio/0, /dev/vfio/1, ...),
        # skip the VFIO control device (/dev/vfio/vfio)
        if not dev_node.name.isdigit():
            continue
        try:
            # Attempt open() at the OS level — same syscall PJRT uses internally.
            # O_RDWR is required for VFIO group devices; O_NONBLOCK prevents blocking.
            fd = os.open(str(dev_node), os.O_RDWR | os.O_NONBLOCK)
            os.close(fd)
        except OSError as e:
            if e.errno == errno.EBUSY:
                busy_devices.append(dev_node.name)
            # EACCES/EPERM = permissions issue, not contention — skip silently
            # and let torch_xla handle it with its own error path

    if not busy_devices:
        return

    devices_str = ", ".join(f"/dev/vfio/{d}" for d in busy_devices)
    diag = (
        "\n" + "=" * 76 + "\n"
        f"  [FATAL] TPU devices locked by another process: {devices_str}\n\n"
        "  A previous training run crashed or was interrupted without releasing\n"
        "  the TPU hardware. The chip's internal state may also be corrupted.\n\n"
        "  The ONLY reliable fix is a full hardware reset:\n\n"
        "    Kaggle:         Session -> Restart Session  (top-right menu)\n"
        "    Colab:          Runtime -> Restart Runtime\n"
        "    GCE TPU VM:     sudo reboot\n\n"
        "  WARNING: Do NOT manually kill stale processes (fuser -k, kill -9).\n"
        "  Killing the process frees the file descriptor but does NOT reset the\n"
        "  TPU chip state (HBM, command queues). Subsequent operations will\n"
        "  SIGSEGV inside PJRT's ExecuteReplicated() thread pool.\n"
        "=" * 76 + "\n"
    )
    sys.stderr.write(diag)
    sys.stderr.flush()

    if _is_interactive_notebook():
        # In notebooks, raise so the cell fails visibly without killing the kernel
        raise RuntimeError(
            f"TPU devices locked: {devices_str}. "
            "Restart your notebook session (Session -> Restart Session) to release the hardware."
        )
    else:
        # In CLI/scripts, use os._exit() to bypass Py_FinalizeEx and avoid the
        # secondary C++ 'InitializeComputationClient() can only be called once' abort
        # that torch_xla's atexit hook triggers during Python shutdown.
        os._exit(1)


def get_xla_device():
    """
    Returns the XLA device singleton.
    Guarantees xm.xla_device() is called at most once per process lifetime.
    Runs a passive VFIO pre-flight check to detect hardware lock contention
    and exit cleanly before PJRT initialization can cause unrecoverable crashes.
    """
    global _CACHED_XLA_DEVICE
    if _CACHED_XLA_DEVICE is not None:
        return _CACHED_XLA_DEVICE

    # Passive check: bail out with clear diagnostics if /dev/vfio/* is already locked,
    # BEFORE torch_xla touches the hardware at all.
    _preflight_check_vfio()

    import torch_xla.core.xla_model as xm

    # xm.xla_device() initializes the PJRT computation client exactly once.
    # If VFIO is accessible (pre-flight passed), this should succeed.
    _CACHED_XLA_DEVICE = xm.xla_device()
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


