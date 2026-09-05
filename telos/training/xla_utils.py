"""
PyTorch-XLA device and runtime management utilities.
Caches device and topology singletons to prevent 'InitializeComputationClient() can only be called once'
fatal runtime assertions when training multiple phases or re-running in notebook/interactive environments.
"""

import os

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


def get_xla_device():
    """
    Returns the XLA device singleton.
    Guarantees xm.xla_device() is called at most once per process lifetime.
    """
    global _CACHED_XLA_DEVICE
    if _CACHED_XLA_DEVICE is not None:
        return _CACHED_XLA_DEVICE

    import torch_xla.core.xla_model as xm
    # xm.xla_device() must only be called once per process to initialize the computation client
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
