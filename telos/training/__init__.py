from .core import build_special_token_lut, get_sys_mem_str
from .dataloader import (
    get_global_targets_contiguous,
    get_global_targets_contiguous_pytorch,
)
from .hardware import (
    HardwareProfile,
    detect_apple_silicon_profile,
    detect_cuda_profile,
    detect_tpu_profile,
)

_TORCH_EXPORTS = {
    "UnifiedPyTorchTrainer": ".trainer_pytorch",
    "WarmupCosineLR": ".lr_schedule",
}

def __getattr__(name: str):
    if name == "UnifiedMLXTrainer":
        try:
            from .trainer_mlx import UnifiedMLXTrainer
            return UnifiedMLXTrainer
        except ImportError as err:
            raise ImportError(
                "UnifiedMLXTrainer requires 'mlx', which is not available in this environment. "
                "Install it on Apple Silicon via `pip install 'telos[mlx]'`."
            ) from err
    if name in _TORCH_EXPORTS:
        try:
            import importlib
            mod = importlib.import_module(_TORCH_EXPORTS[name], __package__)
            return getattr(mod, name)
        except ImportError as err:
            raise ImportError(
                f"{name} requires 'torch', which is not available in this environment. "
                "Install it via `pip install torch`."
            ) from err
    if name in ("clip_grad_norm_mlx", "execute_mlx_training_step", "cast_optimizer_moments_bf16"):
        from . import core
        return getattr(core, name)
    if name == "get_global_targets_contiguous_mlx":
        from . import dataloader
        return getattr(dataloader, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "clip_grad_norm_mlx",
    "build_special_token_lut",
    "get_sys_mem_str",
    "execute_mlx_training_step",
    "cast_optimizer_moments_bf16",
    "get_global_targets_contiguous",
    "get_global_targets_contiguous_mlx",
    "get_global_targets_contiguous_pytorch",
    "WarmupCosineLR",
    "UnifiedMLXTrainer",
    "UnifiedPyTorchTrainer",
    "HardwareProfile",
    "detect_apple_silicon_profile",
    "detect_cuda_profile",
    "detect_tpu_profile",
]
