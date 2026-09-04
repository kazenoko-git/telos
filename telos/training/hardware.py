"""
Modular Hardware Profiles and Runtime Auto-Detection for Telos.
Handles Apple Silicon unified memory tiers, CUDA multi-GPU scaling, and TPU topologies.
"""

import os
import subprocess
from dataclasses import dataclass


@dataclass
class HardwareProfile:
    backend: str
    device_name: str
    total_memory_gb: float
    device_count: int
    eval_policy: str  # "eager", "step", "lazy"
    precision: str
    is_distributed: bool


def detect_apple_silicon_profile(user_policy: str = "auto") -> HardwareProfile:
    """Detects Apple Silicon unified RAM and chooses the optimal MLX graph evaluation policy."""
    total_mem_gb = 16.0
    try:
        res = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
        if res.returncode == 0:
            total_mem_gb = int(res.stdout.strip()) / (1024 ** 3)
    except Exception:
        pass

    chip_name = "Apple Silicon"
    try:
        res = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
        if res.returncode == 0:
            chip_name = res.stdout.strip()
    except Exception:
        pass

    if user_policy in ["eager", "step", "lazy"]:
        eval_policy = user_policy
    else:
        # Low RAM (<24GB): M1/M2 8-16GB, M5 Pro 24GB base -> eager microbatch eval
        # Balanced (24GB-48GB): M5 Pro 24GB, M5 Max 36GB -> step-boundary eval
        # High RAM (>48GB): M5 Max 64GB+, M2-M5 Ultra 96-192GB -> lazy pipelining
        if total_mem_gb < 24.0:
            eval_policy = "eager"
        elif total_mem_gb <= 48.0:
            eval_policy = "step"
        else:
            eval_policy = "lazy"

    return HardwareProfile(
        backend="mlx",
        device_name=chip_name,
        total_memory_gb=total_mem_gb,
        device_count=1,
        eval_policy=eval_policy,
        precision="bfloat16",
        is_distributed=False
    )


def detect_cuda_profile() -> HardwareProfile:
    """Detects CUDA multi-GPU topology and sets optimal precision (bf16 vs fp16)."""
    import torch
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        return HardwareProfile("pytorch", "CPU", 0.0, 0, "eager", "fp32", False)

    gpu_name = torch.cuda.get_device_name(0)
    total_mem_gb = sum(torch.cuda.get_device_properties(i).total_memory for i in range(n_gpus)) / (1024 ** 3)
    
    # Check native BF16 support (Ampere/Ada/Hopper support bf16, Turing T4 uses fp16 + GradScaler)
    has_bf16 = torch.cuda.is_bf16_supported()
    precision = "bf16" if has_bf16 else "fp16"

    return HardwareProfile(
        backend="pytorch",
        device_name=f"{n_gpus}x {gpu_name}",
        total_memory_gb=total_mem_gb,
        device_count=n_gpus,
        eval_policy="eager",
        precision=precision,
        is_distributed=(n_gpus > 1)
    )


def detect_tpu_profile() -> HardwareProfile:
    """Detects PyTorch-XLA TPU slice configuration (v6e-1, v3e-8, v5e-8, v6e-16)."""
    try:
        import torch_xla.core.xla_model as xm
        world_size = xm.xrt_world_size()
        return HardwareProfile(
            backend="pytorch",
            device_name=f"TPU Slice ({world_size} Cores)",
            total_memory_gb=world_size * 16.0,
            device_count=world_size,
            eval_policy="step",
            precision="bfloat16",
            is_distributed=(world_size > 1)
        )
    except Exception:
        return HardwareProfile("pytorch", "XLA Device", 16.0, 1, "step", "bfloat16", False)
