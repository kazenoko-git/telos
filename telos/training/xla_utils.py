"""
PyTorch-XLA device and runtime management utilities.
Caches device and topology singletons to prevent 'InitializeComputationClient() can only be called once'
fatal runtime assertions when training multiple phases or re-running in notebook/interactive environments.
"""

import os
from pathlib import Path

_CACHED_XLA_DEVICE = None
_CACHED_XLA_WORLD_SIZE = None
_CACHED_IS_MASTER = None
_CACHED_SPMD_MESH = None


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


def normalize_single_chip_tpu_environment() -> bool:
    """Repair known v5e-1 host defaults before PJRT initialization.

    Require an explicit accelerator identifier. Do not infer chip count from
    a substring (v5e-16), process count or an arbitrary VM name.
    """
    accelerator = os.environ.get("TPU_ACCELERATOR_TYPE", "").strip().lower()
    if accelerator != "v5e-1":
        return False
    for key in ("TPU_NUM_DEVICES", "WORLD_SIZE", "PJRT_LOCAL_PROCESS_COUNT"):
        value = os.environ.get(key)
        if value is not None and value != "1":
            raise RuntimeError(f"Single-chip v5e-1 conflicts with {key}={value}; refusing topology override")
    if os.environ.get("TPU_PROCESS_BOUNDS", "1,1,1") != "1,1,1":
        raise RuntimeError("Single-chip v5e-1 conflicts with TPU_PROCESS_BOUNDS; refusing topology override")
    hosts = [item.strip() for item in os.environ.get("TPU_WORKER_HOSTNAMES", "").split(",") if item.strip()]
    if len(hosts) > 1:
        raise RuntimeError("Single-chip v5e-1 conflicts with multiple TPU workers; refusing topology override")
    # Only replace the documented default or already normalized bounds. Other
    # explicit values are an inconsistent configuration, not grounds to guess.
    for key in ("TPU_CHIPS_PER_HOST_BOUNDS", "TPU_CHIPS_PER_PROCESS_BOUNDS"):
        value = os.environ.get(key)
        if value is not None and value not in {"1,1,1", "2,2,1"}:
            raise RuntimeError(f"Unexpected {key}={value} for v5e-1; refusing topology override")
    os.environ["TPU_CHIPS_PER_HOST_BOUNDS"] = "1,1,1"
    os.environ["TPU_CHIPS_PER_PROCESS_BOUNDS"] = "1,1,1"
    return True


def clean_tpu_environment():
    """
    Cleans conflicting Kaggle/GCP environment variables that trigger fatal
    SliceBuilder port lookup crashes in PyTorch-XLA PJRT multiprocessing,
    and configures LibTPU runtime flags to eliminate host CPU busy-polling.
    """
    normalize_single_chip_tpu_environment()
    if os.environ.get("TPU_PROCESS_ADDRESSES", "").strip().lower() == "local":
        os.environ.pop("TPU_PROCESS_ADDRESSES", None)
    if "CLOUD_TPU_TASK_ID" in os.environ and "TPU_NAME" not in os.environ:
        os.environ.pop("CLOUD_TPU_TASK_ID", None)

    # Purge any unrecognized spin-wait flags from LIBTPU_INIT_ARGS that abort gflags startup:
    if "LIBTPU_INIT_ARGS" in os.environ:
        cleaned_args = " ".join(arg for arg in os.environ["LIBTPU_INIT_ARGS"].split() if "spin_wait" not in arg)
        if cleaned_args:
            os.environ["LIBTPU_INIT_ARGS"] = cleaned_args
        else:
            os.environ.pop("LIBTPU_INIT_ARGS", None)

    # OpenMP / BLAS thread explosion and glibc arena bloat prevention across spawned processes:
    os.environ.setdefault("MALLOC_ARENA_MAX", "2")
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def get_xla_device():
    """
    Returns the XLA device singleton.
    Guarantees xm.xla_device() is called at most once per process lifetime,
    preventing 'InitializeComputationClient() can only be called once' fatal assertions.
    """
    global _CACHED_XLA_DEVICE
    if _CACHED_XLA_DEVICE is not None:
        return _CACHED_XLA_DEVICE

    clean_tpu_environment()
    import torch
    try:
        import torch_xla
        if not hasattr(torch, "xla"):
            torch.xla = torch_xla
    except ImportError:
        pass
    import torch_xla.core.xla_model as xm
    _CACHED_XLA_DEVICE = xm.xla_device()
    return _CACHED_XLA_DEVICE



def get_xla_spmd_mesh():
    """Returns the active SPMD Mesh singleton across addressable TPU chips, or None."""
    global _CACHED_SPMD_MESH
    if _CACHED_SPMD_MESH is not None:
        return _CACHED_SPMD_MESH

    try:
        import torch_xla.runtime as xr
        if xr.is_spmd():
            import torch_xla.distributed.spmd as xs
            import numpy as np
            n_dev = xr.global_runtime_device_count()
            _CACHED_SPMD_MESH = xs.Mesh(np.arange(n_dev), (n_dev,), ("data",))
            return _CACHED_SPMD_MESH
    except Exception:
        pass
    return None


def get_xla_world_size() -> int:
    """Returns the XLA world size (SPMD chip count or standard XRT world size)."""
    global _CACHED_XLA_WORLD_SIZE
    if _CACHED_XLA_WORLD_SIZE is not None:
        return _CACHED_XLA_WORLD_SIZE

    # If XLA is not yet initialized in this process, read environment variables first
    # to avoid premature ComputationClient initialization before the trainer starts.
    if not is_xla_initialized():
        if normalize_single_chip_tpu_environment():
            return 1
        for env_k in ("TPU_NUM_DEVICES", "WORLD_SIZE", "PJRT_LOCAL_PROCESS_COUNT"):
            if env_k in os.environ:
                try:
                    return max(1, int(os.environ[env_k]))
                except ValueError:
                    pass
        if is_tpu_environment():
            try:
                vfio_chips = list(Path("/dev/vfio").glob("[0-9]*"))
                if len(vfio_chips) >= 4:
                    return 8
                elif len(vfio_chips) > 0:
                    return len(vfio_chips) * 2
            except Exception:
                pass
            return 8

    try:
        import torch_xla.runtime as xr
        if hasattr(xr, "world_size"):
            ws = xr.world_size()
            if ws > 1:
                _CACHED_XLA_WORLD_SIZE = ws
                return _CACHED_XLA_WORLD_SIZE
        if xr.is_spmd():
            _CACHED_XLA_WORLD_SIZE = xr.global_runtime_device_count()
            return max(1, _CACHED_XLA_WORLD_SIZE)
    except Exception:
        pass

    try:
        import torch_xla.core.xla_model as xm
        if hasattr(xm, "xrt_world_size"):
            ws = xm.xrt_world_size()
            if ws > 1:
                _CACHED_XLA_WORLD_SIZE = ws
                return _CACHED_XLA_WORLD_SIZE
    except Exception:
        pass

    _CACHED_XLA_WORLD_SIZE = 1
    return _CACHED_XLA_WORLD_SIZE


def is_xla_master() -> bool:
    """Returns True only for the verified XLA master; never guess on a TPU."""
    global _CACHED_IS_MASTER
    if _CACHED_IS_MASTER is not None:
        return _CACHED_IS_MASTER

    errors = []
    try:
        import torch_xla.runtime as xr
        if hasattr(xr, "process_index"):
            _CACHED_IS_MASTER = (xr.process_index() == 0)
            return _CACHED_IS_MASTER
        errors.append("torch_xla.runtime.process_index is unavailable")
    except Exception as exc:
        errors.append(f"torch_xla.runtime.process_index failed: {exc}")

    try:
        import torch_xla.core.xla_model as xm
        if hasattr(xm, "is_master_ordinal"):
            _CACHED_IS_MASTER = xm.is_master_ordinal()
            return _CACHED_IS_MASTER
        errors.append("torch_xla.core.xla_model.is_master_ordinal is unavailable")
    except Exception as exc:
        errors.append(f"torch_xla.core.xla_model.is_master_ordinal failed: {exc}")

    if is_tpu_environment() or is_xla_initialized():
        raise RuntimeError("Unable to determine the PyTorch/XLA master rank: " + "; ".join(errors))

    _CACHED_IS_MASTER = True
    return _CACHED_IS_MASTER


