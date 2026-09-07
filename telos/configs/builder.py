"""
Automatic Configuration Builder for Télos.
Translates the 6 fundamental user dimensions (parameters, tokens, batching,
tokenizer, hardware, hardware count) into an executable training configuration.
Supports optional YAML config file bypasses.
"""

import os
import math
from pathlib import Path
import yaml

from .solver import solve_transformer_geometry, parse_human_number


def auto_detect_hardware() -> tuple[str, str, int]:
    """
    Detects optimal hardware backend, device type, and available device count.
    Returns:
        (backend: 'mlx' | 'pytorch', device: 'cuda' | 'xla' | 'mps' | 'cpu', device_count: int)
    """
    import platform
    is_mac = platform.system().lower() == "darwin"
    
    # 1. Check Apple Silicon Metal (MLX)
    if is_mac:
        try:
            import mlx.core as mx
            return "mlx", "gpu", 1
        except ImportError:
            pass

    # 2. Check PyTorch-XLA (Google Cloud / TPU Pods)
    # Check non-intrusively via environment inspection or cached device singleton
    # to avoid premature ComputationClient initialization before the trainer starts.
    import os
    from telos.training.xla_utils import is_tpu_environment, is_xla_initialized, get_xla_world_size
    if is_tpu_environment() or is_xla_initialized():
        count = get_xla_world_size()
        return "pytorch", "xla", max(1, count)

    # 3. Check NVIDIA CUDA GPUs
    try:
        import torch
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            return "pytorch", "cuda", max(1, count)
    except ImportError:
        pass

    # 4. Fallback to CPU
    return "pytorch", "cpu", 1


def build_config(
    paradigm: str,
    phase: str = "A",
    params: str | int | None = "12M",
    tokens: str | int | None = None,
    effective_batch: int | str | None = None,
    batch_size: int | None = None,
    grad_accum: int | None = None,
    seq_len: int = 512,
    tokenizer: str | None = None,
    vocab_size: int | None = None,
    hardware: str | None = "auto",
    devices: int | str | None = "auto",
    max_steps: int | None = None,
    max_lr: float | None = None,
    min_lr: float | None = None,
    warmup_steps: int | None = None,
    weight_decay: float | None = None,
    checkpoint_dir: str | None = None,
    save_every: int | None = None,
    config_path: str | Path | None = None,
    data_path: str | Path | None = None,
    synthetic: bool = False,
    **kwargs
) -> dict:
    """
    Constructs a complete, validated configuration dictionary directly from CLI dimensions.
    """
    paradigm = str(paradigm).lower()
    phase = str(phase).upper()

    # 1. Base Configuration (from YAML bypass if supplied, else empty)
    if config_path and Path(config_path).exists():
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {"model": {}, "training": {}, "checkpoint": {}, "data": {}}

    m_cfg = cfg.setdefault("model", {})
    t_cfg = cfg.setdefault("training", {})
    c_cfg = cfg.setdefault("checkpoint", {})
    d_cfg = cfg.setdefault("data", {})

    # 2. Tokenizer & Vocab Size
    if tokenizer:
        d_cfg["tokenizer"] = str(tokenizer)
    actual_vocab = vocab_size or m_cfg.get("vocab_size", 8192)
    m_cfg["vocab_size"] = actual_vocab
    m_cfg["seq_len"] = seq_len
    m_cfg["max_seq_len"] = seq_len

    # 3. Model Architecture Resolution from Parameters
    if params is not None and not (m_cfg.get("d_model") and m_cfg.get("n_layers") and config_path):
        geometry = solve_transformer_geometry(params, vocab_size=actual_vocab)
        m_cfg["d_model"] = geometry["d_model"]
        m_cfg["n_layers"] = geometry["n_layers"]
        m_cfg["n_heads"] = geometry["n_heads"]
        m_cfg["n_kv_heads"] = geometry["n_kv_heads"]
        m_cfg["tied_embeddings"] = geometry["tied_embeddings"]
        cfg["_resolved_params"] = geometry["actual_params"]
    else:
        m_cfg.setdefault("d_model", 512)
        m_cfg.setdefault("n_layers", 12)
        m_cfg.setdefault("n_heads", 16)
        m_cfg.setdefault("n_kv_heads", m_cfg["n_heads"])
        m_cfg.setdefault("tied_embeddings", True)

    # 4. Hardware & Distributed Setup Resolution
    hw = str(hardware).lower() if hardware else "auto"
    if hw in ["auto", "none"] or devices in ["auto", None]:
        det_backend, det_device, det_count = auto_detect_hardware()
    else:
        det_backend, det_device, det_count = ("mlx", "gpu", 1) if hw == "mlx" else ("pytorch", hw, 1)
    
    if hw in ["auto", "none"]:
        final_backend = det_backend
        final_device = det_device
    elif hw in ["mlx"]:
        final_backend = "mlx"
        final_device = "gpu"
    elif hw in ["cuda", "mps", "xla", "cpu"]:
        final_backend = "pytorch"
        final_device = hw
    elif hw in ["pytorch"]:
        final_backend = "pytorch"
        final_device = det_device if det_device != "gpu" else "cpu"
    else:
        final_backend = "pytorch"
        final_device = hw

    cfg["_backend"] = final_backend
    cfg["_device"] = final_device
    
    dev_count = det_count if devices in ["auto", None] else int(devices)
    cfg["_device_count"] = dev_count

    # 5. Profile Inheritance Resolution (tpu:, mac:, gpu:, lightning:)
    profile_candidate = None
    if final_device == "xla" or hw in ["tpu", "xla"]:
        profile_candidate = "tpu"
    elif final_backend == "mlx" or final_device in ["mps", "mac"]:
        profile_candidate = "mac"
    elif final_device == "cuda" or hw in ["cuda", "gpu"]:
        is_lightning = bool(os.environ.get("LIGHTNING_CLUSTER") or os.environ.get("LIGHTNING_ENVIRONMENT"))
        if is_lightning and "lightning" in t_cfg:
            profile_candidate = "lightning"
        else:
            profile_candidate = "gpu"

    if profile_candidate and profile_candidate in t_cfg and isinstance(t_cfg[profile_candidate], dict):
        prof = t_cfg[profile_candidate]
        if batch_size is None and "batch_size" in prof:
            t_cfg["batch_size"] = prof["batch_size"]
        if grad_accum is None and "gradient_accumulation" in prof:
            t_cfg["gradient_accumulation"] = prof["gradient_accumulation"]
        if devices in ["auto", None] and "num_devices" in prof:
            dev_count = prof["num_devices"]
            cfg["_device_count"] = dev_count
        if "compile" in prof and t_cfg.get("compile") is None:
            t_cfg["compile"] = bool(prof["compile"])

    if kwargs.get("compile") is not None:
        t_cfg["compile"] = bool(kwargs["compile"])

    # 6. Batch Size & Gradient Accumulation Resolution (Hardware Tier Aware)
    d_model = m_cfg["d_model"]
    
    if final_device == "xla":
        # TPU v3 (16GB HBM) per-core microbatch sizing: 32-48 sequences is the sweet spot
        # to saturate MXU systolic arrays while keeping attention matrices well within 16GB HBM.
        auto_microbatch = 48 if d_model <= 768 else 16
    elif final_backend == "mlx":
        # Apple Silicon memory-tier scaling: scale microbatch based on unified memory capacity
        try:
            from telos.training.hardware import detect_apple_silicon_profile
            hw_prof = detect_apple_silicon_profile()
            tot_gb = hw_prof.total_memory_gb
        except Exception:
            tot_gb = 16.0

        if tot_gb >= 64.0:
            auto_microbatch = 64 if d_model <= 256 else (32 if d_model <= 512 else 16)
        elif tot_gb >= 32.0:
            auto_microbatch = 32 if d_model <= 256 else (16 if d_model <= 512 else 8)
        elif tot_gb >= 24.0:
            auto_microbatch = 16 if d_model <= 512 else 8
        else:
            auto_microbatch = 16 if d_model <= 256 else (8 if d_model <= 512 else 4)
    else:
        # CUDA / CPU default heuristic
        auto_microbatch = 16 if d_model <= 512 else (8 if d_model <= 1024 else 4)

    actual_bs = batch_size if batch_size is not None else t_cfg.get("batch_size", auto_microbatch)
    t_cfg["batch_size"] = actual_bs

    if effective_batch is not None:
        # Check if user specified effective batch with explicit token suffix (e.g. 32k, 1M, tokens)
        is_token_suffix = False
        if isinstance(effective_batch, str):
            clean_str = effective_batch.strip().lower()
            if clean_str.endswith(("k", "m", "b", "tokens")):
                is_token_suffix = True
        eff_b = parse_human_number(effective_batch)
        if is_token_suffix:
            eff_b = max(1, eff_b // seq_len)
        
        # In multi-device setups (e.g. 8 TPU cores), effective_batch is the total cluster target
        eff_per_device = max(1, eff_b // dev_count) if dev_count > 1 else eff_b
        if batch_size is None:
            actual_bs = min(auto_microbatch, eff_per_device)
            t_cfg["batch_size"] = actual_bs
        actual_accum = max(1, math.ceil(eff_per_device / actual_bs))
        t_cfg["gradient_accumulation"] = actual_accum
    elif grad_accum is not None:
        t_cfg["gradient_accumulation"] = grad_accum
    else:
        t_cfg.setdefault("gradient_accumulation", 1)

    # Hardware Safeguard: On Cloud TPU v3 (16 GB HBM), a per-core microbatch > 48 in a single forward pass
    # produces large [B*H, T, T] attention score matrices that exceed physical 16GB HBM capacity.
    # Auto-split into a hardware-safe per-core microbatch (<= 48) and scale gradient_accumulation.
    if final_device == "xla" and t_cfg["batch_size"] > 48:
        raw_bs = t_cfg["batch_size"]
        safe_microbatch = 48 if raw_bs % 48 == 0 else (32 if raw_bs % 32 == 0 else (16 if raw_bs % 16 == 0 else 24))
        accum_multiplier = max(1, math.ceil(raw_bs / safe_microbatch))
        t_cfg["batch_size"] = safe_microbatch
        t_cfg["gradient_accumulation"] = t_cfg.get("gradient_accumulation", 1) * accum_multiplier


    cluster_multiplier = dev_count if dev_count > 1 else 1
    effective_seqs = t_cfg["batch_size"] * t_cfg["gradient_accumulation"] * cluster_multiplier
    tokens_per_step = effective_seqs * seq_len

    # 6. Training Duration (Tokens -> Steps)
    # If max_steps is explicitly provided (e.g. pre-resolved by cluster coordinator),
    # prioritize it over recalculating from tokens on a single worker's local microbatch.
    if max_steps is not None:
        t_cfg["max_steps"] = int(max_steps)
    elif tokens is not None:
        total_tokens = parse_human_number(tokens)
        resolved_steps = max(1, math.ceil(total_tokens / tokens_per_step))
        t_cfg["max_steps"] = resolved_steps
    else:
        t_cfg.setdefault("max_steps", 5000)


    # 7. Automatic Learning Rate, Warmup & Regularization Scaling
    # Width-adjusted scaling law: base lr = 6e-4 * sqrt(256 / d_model)
    auto_max_lr = 6.0e-4 * math.sqrt(256.0 / max(64, d_model))
    auto_min_lr = 0.1 * auto_max_lr
    auto_warmup = max(50, min(2000, int(0.02 * t_cfg["max_steps"])))

    t_cfg["max_lr"] = float(max_lr if max_lr is not None else t_cfg.get("max_lr", auto_max_lr))
    t_cfg["min_lr"] = float(min_lr if min_lr is not None else t_cfg.get("min_lr", auto_min_lr))
    t_cfg["warmup_steps"] = int(warmup_steps if warmup_steps is not None else t_cfg.get("warmup_steps", auto_warmup))
    t_cfg["weight_decay"] = float(weight_decay if weight_decay is not None else t_cfg.get("weight_decay", 0.1))
    t_cfg.setdefault("grad_clip", 1.0)
    t_cfg.setdefault("precision", "bf16")

    # 8. Paradigm-Specific Properties
    if paradigm == "corosred":
        cfg["corosred"] = {
            "phase": phase,
            "mask_prob": float(kwargs.get("mask_prob", 0.15)),
            "k_amb": int(kwargs.get("k_amb", 5)),
            "self_condition": bool(kwargs.get("self_condition", True)),
            "self_cond_prob": float(kwargs.get("self_cond_prob", 0.5)),
        }
        m_cfg["use_reliability_head"] = True
        m_cfg["mask_token_id"] = 1
    else:
        m_cfg["use_reliability_head"] = False
        m_cfg["mask_token_id"] = 1

    cfg["paradigm"] = paradigm
    m_cfg["paradigm"] = paradigm
    m_cfg["is_causal"] = paradigm in ("ar", "corosred")

    # 9. Checkpoint Storage & Cadence
    ckpt_default = f"checkpoints/{paradigm}"
    if paradigm == "corosred":
        ckpt_default += f"/phase_{phase.lower()}"
        
    if checkpoint_dir is not None:
        c_cfg["checkpoint_dir"] = str(checkpoint_dir)
    else:
        c_cfg.setdefault("checkpoint_dir", ckpt_default)
    
    # Auto-save cadence: 10% of total steps, clamped between 500 and 2500
    auto_save_cadence = max(500, min(2500, int(0.1 * t_cfg["max_steps"])))
    if save_every is not None:
        c_cfg["save_every_steps"] = int(save_every)
    else:
        c_cfg.setdefault("save_every_steps", auto_save_cadence)

    # 10. Data & Synthetic Flag
    if data_path:
        d_cfg["train_path"] = str(data_path)
    if synthetic:
        d_cfg["synthetic"] = True

    return cfg
