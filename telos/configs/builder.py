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
    phase: str | None = None,
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
        # TPU (v5e / v6e / v4 / v3) per-core microbatch sizing:
        # Microbatch 32 (for d_model <= 512) or 16 (for d_model >= 768)
        # Strictly maintains a solid medium effective batch size of 256 sequences across 8 TPU cores:
        # - d_model <= 512 (15M, 25M, 50M): batch_size = 32, grad_accum = 1 -> 32 * 1 * 8 = 256 sequences (131k tok/step).
        # - d_model >= 768 (100M+): batch_size = 16, grad_accum = 2 -> 16 * 2 * 8 = 256 sequences (131k tok/step).
        # Cuts peak activation memory in half for 100M+ to eliminate OOM on 16GB HBM TPU v5e
        # while keeping the effective batch size strictly at 256 (never <= 128).
        auto_microbatch = 32 if d_model <= 512 else 16
        auto_accum = 1 if d_model <= 512 else 2
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
        # CUDA / CPU default heuristic: detect VRAM capacity on CUDA
        cuda_gb = 0.0
        if final_device == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    cuda_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            except Exception:
                cuda_gb = 0.0

        if cuda_gb >= 70.0:
            # Flagship Datacenter GPUs (H100 80GB SXM5/PCIe, A100 80GB):
            # Massive 80GB HBM3 bandwidth (3.35 TB/s) permits large microbatches (up to 384+)
            # Saturates all 132 SMs with FlashAttention-2 and zero activation spilling
            auto_microbatch = 384 if d_model <= 384 else (192 if d_model <= 512 else (128 if d_model <= 768 else 64))
            auto_accum = 1 if d_model <= 384 else (2 if d_model <= 512 else (3 if d_model <= 768 else 4))
        elif cuda_gb >= 40.0:
            # Datacenter GPUs (A100 40GB, L40S 48GB, RTX 6000 Ada 48GB):
            # Microbatch 64 maximally saturates Tensor Cores with FlashAttention-2
            # Defaults to medium effective batch of 256 sequences
            auto_microbatch = 64 if d_model <= 512 else (32 if d_model <= 768 else 16)
            auto_accum = 4 if d_model <= 512 else (8 if d_model <= 768 else 16)
        elif cuda_gb >= 24.0:
            # High-end Consumer GPUs (RTX 3090/4090 24GB):
            auto_microbatch = 32 if d_model <= 512 else (16 if d_model <= 768 else 8)
            auto_accum = 8 if d_model <= 512 else 16
        elif cuda_gb >= 16.0:
            # Mid-tier GPUs (T4 / V100 16GB):
            auto_microbatch = 16 if d_model <= 512 else (8 if d_model <= 768 else 4)
            auto_accum = 16 if d_model <= 512 else 32
        else:
            # Low VRAM / CPU fallback
            auto_microbatch = 16 if d_model <= 512 else (8 if d_model <= 1024 else 4)
            auto_accum = 1

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
        
        # In multi-device setups (e.g. 8 TPU cores or 8 GPUs), effective_batch is the total cluster target
        eff_per_device = max(1, eff_b // dev_count) if dev_count > 1 else eff_b
        if batch_size is None:
            if final_device == "xla" and eff_per_device <= 48 and d_model <= 512:
                actual_bs = eff_per_device
            else:
                max_cand = min(auto_microbatch, eff_per_device)
                divisors = [d for d in range(1, max_cand + 1) if eff_per_device % d == 0]
                actual_bs = divisors[-1] if divisors else max_cand
            t_cfg["batch_size"] = actual_bs
        actual_accum = max(1, math.ceil(eff_per_device / actual_bs))
        t_cfg["gradient_accumulation"] = actual_accum
    elif grad_accum is not None:
        t_cfg["gradient_accumulation"] = grad_accum
    else:
        if final_device == "xla" and batch_size is None:
            t_cfg["gradient_accumulation"] = auto_accum
        elif final_device == "cuda" and batch_size is None:
            # Scale target effective batch: 384 sequences (196k tok/step) on >=70GB H100/A100, 256 sequences on others
            target_eff = 384 if cuda_gb >= 70.0 else 256
            cluster_multiplier = dev_count if dev_count > 1 else 1
            eff_per_dev = max(1, target_eff // cluster_multiplier)
            t_cfg["gradient_accumulation"] = max(1, math.ceil(eff_per_dev / actual_bs))
        else:
            t_cfg.setdefault("gradient_accumulation", 1)

    # Hardware Safeguard: On TPU v5e (16 GB HBM), for d_model >= 768 (100M+),
    # microbatch > 16 without gradient accumulation can cause HBM exhaustion during full bidirectional infilling.
    if final_device == "xla" and t_cfg["batch_size"] > 16 and grad_accum != 1 and d_model >= 768:
        raw_bs = t_cfg["batch_size"]
        safe_microbatch = 16
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
        raw_phase = str(phase).upper() if phase is not None else None
        is_legacy = (raw_phase in ["A", "B", "C"]) and bool(kwargs.get("legacy_phases", False))
        is_unified = not is_legacy

        if is_unified:
            resolved_phase = "UNIFIED"
            cfg["corosred"] = {
                "unified": True,
                "phase": "unified",
                "alpha_max": float(kwargs.get("alpha_max", 0.85)),
                "alpha_min": float(kwargs.get("alpha_min", 0.20)),
                "beta_min": float(kwargs.get("beta_min", 0.15)),
                "beta_max": float(kwargs.get("beta_max", 0.70)),
                "gamma_max": float(kwargs.get("gamma_max", 0.10)),
                "hold_fraction": float(kwargs.get("hold_fraction", 0.20)),
                "decay_power": float(kwargs.get("decay_power", 2.5)),
                "gamma_gate_auc": float(kwargs.get("gamma_gate_auc", 0.55)),
                "acc_gate_threshold": float(kwargs.get("acc_gate_threshold", 0.65)),
                "causal_ratio": float(kwargs.get("causal_ratio", 0.75)),
                "routing_cache_steps": int(kwargs.get("routing_cache_steps", 50)),
                "adaptive_rebalance": bool(kwargs.get("adaptive_rebalance", False)),
                "mask_prob": float(kwargs.get("mask_prob", 0.15)),
                "k_amb": int(kwargs.get("k_amb", 5)),
            }
        else:
            resolved_phase = raw_phase if raw_phase else "A"
            is_phase_c = (resolved_phase == "C")
            default_self_cond = is_phase_c
            default_sc_prob = 0.5 if is_phase_c else 0.0
            cfg["corosred"] = {
                "unified": False,
                "phase": resolved_phase,
                "mask_prob": float(kwargs.get("mask_prob", 0.15)),
                "k_amb": int(kwargs.get("k_amb", 5)),
                "self_condition": bool(kwargs.get("self_condition", default_self_cond)),
                "self_cond_prob": float(kwargs.get("self_cond_prob", default_sc_prob)),
            }
        m_cfg["use_reliability_head"] = True
        m_cfg["mask_token_id"] = 1
    else:
        is_unified = False
        resolved_phase = "A"
        m_cfg["use_reliability_head"] = False
        m_cfg["mask_token_id"] = 1

    cfg["paradigm"] = paradigm
    m_cfg["paradigm"] = paradigm
    m_cfg["is_causal"] = paradigm in ("ar", "corosred")

    # 9. Checkpoint Storage & Cadence
    ckpt_default = f"checkpoints/{paradigm}"
    if paradigm == "corosred":
        if is_unified:
            ckpt_default += "/unified"
        else:
            ckpt_default += f"/phase_{resolved_phase.lower()}"
        
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
