_MLX_FUNCTIONS = {
    "ar_loss_fn_mlx": ".ar",
    "apply_masking_mlx": ".mdlm",
    "mdlm_loss_mlx": ".mdlm",
    "apply_uniform_noise_mlx": ".undlm",
    "undlm_loss_mlx": ".undlm",
    "crsr_phase_a_loss_fn_mlx": ".corosred",
    "crsr_phase_b_loss_fn_mlx": ".corosred",
    "MLXMDLMSampler": ".sampler",
}

_TORCH_FUNCTIONS = {
    "ar_loss_fn_pytorch": ".ar",
    "apply_masking_pytorch": ".mdlm",
    "mdlm_loss_pytorch": ".mdlm",
    "sample_beta_timesteps": ".mdlm",
    "sample_uniform_timesteps": ".mdlm",
    "apply_uniform_noise_pytorch": ".undlm",
    "undlm_loss_pytorch": ".undlm",
    "crsr_phase_a_loss_fn_pytorch": ".corosred",
    "crsr_phase_b_loss_fn_pytorch": ".corosred",
    "MDLMSampler": ".sampler",
}

def __getattr__(name: str):
    if name in _MLX_FUNCTIONS:
        try:
            import mlx.core
        except ImportError as err:
            raise ImportError(
                f"{name} requires 'mlx', which is not available in this environment. "
                "Install it on Apple Silicon via `pip install 'telos[mlx]'`."
            ) from err
        import importlib
        mod = importlib.import_module(_MLX_FUNCTIONS[name], __package__)
        return getattr(mod, name)
    
    if name in _TORCH_FUNCTIONS:
        try:
            import torch
        except ImportError as err:
            raise ImportError(
                f"{name} requires 'torch', which is not available in this environment. "
                "Install it via `pip install torch`."
            ) from err
        import importlib
        mod = importlib.import_module(_TORCH_FUNCTIONS[name], __package__)
        return getattr(mod, name)
        
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ar_loss_fn_pytorch",
    "apply_masking_pytorch",
    "mdlm_loss_pytorch",
    "sample_beta_timesteps",
    "sample_uniform_timesteps",
    "apply_uniform_noise_pytorch",
    "undlm_loss_pytorch",
    "crsr_phase_a_loss_fn_pytorch",
    "crsr_phase_b_loss_fn_pytorch",
    "MDLMSampler",
    "MLXMDLMSampler",
    "ar_loss_fn_mlx",
    "apply_masking_mlx",
    "mdlm_loss_mlx",
    "apply_uniform_noise_mlx",
    "undlm_loss_mlx",
    "crsr_phase_a_loss_fn_mlx",
    "crsr_phase_b_loss_fn_mlx",
]
