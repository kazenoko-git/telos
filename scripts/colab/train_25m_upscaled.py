"""
Google Colab / Kaggle TPU & Cloud GPU Training Script for 25M Upscaled Suite.

Supports:
- Direct Single-Process TPU execution (matching benchmark setup)
- Standard PyTorch DataLoader with hardware sync via torch_xla.sync()
- Zero-shock upscaling with invariant RMSNorm scaling
"""

import os
import shutil
if "TPU_PROCESS_ADDRESSES" in os.environ:
    os.environ.pop("TPU_PROCESS_ADDRESSES")
if "CLOUD_TPU_TASK_ID" in os.environ:
    os.environ.pop("CLOUD_TPU_TASK_ID")

import sys
import time
import math
import gc
import yaml
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from safetensors.torch import load_file, save_file
from huggingface_hub import snapshot_download, HfApi

# Ensure repo root is on PATH
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from mdiff.model.transformer import TelosTransformer, TelosConfig


# In PyTorch 2.x+, LRScheduler is the public API; fall back to _LRScheduler on older versions
_LRSchedulerBase = getattr(torch.optim.lr_scheduler, "LRScheduler", getattr(torch.optim.lr_scheduler, "_LRScheduler", object))


class WarmupCosineLR(_LRSchedulerBase):
    """Cosine learning rate schedule with linear warmup."""
    def __init__(self, optimizer, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float = 1e-5, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            lr = self.max_lr * (step + 1) / max(1, self.warmup_steps)
        elif step > self.max_steps:
            lr = self.min_lr
        else:
            decay_ratio = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            lr = self.min_lr + coeff * (self.max_lr - self.min_lr)
        # Return Python float: standard PyTorch LR update without creating on-device XLA graph constants
        return [float(lr) for _ in self.base_lrs]


def mdlm_loss_pytorch(model, clean_targets, mask_token_id=1, vocab_size=8192):
    """Absorbing discrete diffusion ELBO loss for PyTorch & TPU."""
    B, T = clean_targets.shape
    device = clean_targets.device
    
    t_values = torch.rand((B, 1), device=device)
    mask_matrix = torch.rand((B, T), device=device) < t_values
    mask_token_tensor = torch.full((B, T), mask_token_id, dtype=torch.long, device=device)
    masked_ids = torch.where(mask_matrix, mask_token_tensor, clean_targets)
    
    logits = model(masked_ids)
    
    ce_per_token = nn.functional.cross_entropy(
        logits.view(-1, vocab_size).float(),
        clean_targets.view(-1),
        reduction="none"
    ).view(B, T)
    
    masked_ce = ce_per_token * mask_matrix.float()
    masked_counts = torch.clamp(torch.sum(mask_matrix.float(), dim=1), min=1.0, max=float(T))
    per_example_ce = torch.sum(masked_ce, dim=1) / masked_counts
    
    t_weights = 1.0 / torch.clamp(t_values.squeeze(-1), min=1e-3, max=1.0)
    return torch.mean(per_example_ce * t_weights)


def undlm_loss_pytorch(model, clean_targets, vocab_size=8192):
    """Uniform noise discrete diffusion ELBO loss for PyTorch & TPU."""
    B, T = clean_targets.shape
    device = clean_targets.device
    
    t_values = torch.rand((B, 1), device=device)
    rand_noise = torch.randint(0, vocab_size, size=(B, T), device=device)
    corrupt_mask = torch.rand((B, T), device=device) < t_values
    noisy_ids = torch.where(corrupt_mask, rand_noise, clean_targets)
    
    logits = model(noisy_ids)
    
    ce_per_token = nn.functional.cross_entropy(
        logits.view(-1, vocab_size).float(),
        clean_targets.view(-1),
        reduction="none"
    ).view(B, T)
    
    per_example_ce = torch.mean(ce_per_token, dim=1)
    t_weights = 1.0 / torch.clamp(t_values.squeeze(-1), min=1e-3, max=1.0)
    return torch.mean(per_example_ce * t_weights)


def detect_device_type(device_str: str | None = None) -> str:
    """Detects hardware type WITHOUT creating any device (safe for SPMD init ordering)."""
    if device_str:
        d = device_str.lower()
        if d in ("tpu", "xla"): return "tpu"
        if "cuda" in d: return "cuda"
        if "mps" in d: return "mps"
        return "cpu"
    if os.environ.get("PJRT_DEVICE") == "TPU": return "tpu"
    if os.path.exists("/dev/accel0"): return "tpu"
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


def create_device(device_type: str):
    """Creates a torch device after SPMD has been initialized (if TPU)."""
    if device_type == "tpu":
        import torch_xla.core.xla_model as xm
        return xm.xla_device()
    elif device_type == "cuda":
        return torch.device("cuda")
    elif device_type == "mps":
        return torch.device("mps")
    return torch.device("cpu")


def resolve_training_params(cfg: dict, device_type: str = "tpu") -> dict:
    """
    Dynamically computes batch size, gradient accumulation, max steps, and warmup steps
    from device-independent total_tokens and hardware execution profiles.
    """
    t_cfg = cfg["training"]
    m_cfg = cfg["model"]
    seq_len = m_cfg.get("seq_len", 512)

    dev_key = "tpu" if device_type in ("tpu", "xla") else ("mac" if device_type in ("mac", "mps", "metal", "mlx") else ("gpu" if "cuda" in str(device_type).lower() else "mac"))

    # Dynamically detect available CUDA devices on GPU systems
    if dev_key == "gpu" and torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
    elif dev_key in t_cfg and isinstance(t_cfg[dev_key], dict):
        num_devices = int(t_cfg[dev_key].get("num_devices", 1))
    else:
        num_devices = int(t_cfg.get("num_devices", 1))

    if dev_key in t_cfg and isinstance(t_cfg[dev_key], dict):
        dev_profile = t_cfg[dev_key]
        batch_size = int(dev_profile.get("batch_size", 32))
        grad_accum = int(dev_profile.get("gradient_accumulation", 1))
    else:
        batch_size = int(t_cfg.get("batch_size", 32))
        grad_accum = int(t_cfg.get("gradient_accumulation", 1))

    if "total_tokens" in t_cfg:
        total_tokens = int(t_cfg["total_tokens"])
        tokens_per_step = batch_size * grad_accum * num_devices * seq_len
        max_steps = max(1, math.ceil(total_tokens / tokens_per_step))
        warmup_ratio = float(t_cfg.get("warmup_ratio", 0.05))
        warmup_steps = max(1, int(max_steps * warmup_ratio))
    else:
        max_steps = int(t_cfg.get("max_steps", 100))
        warmup_steps = int(t_cfg.get("warmup_steps", 10))

    resolved = dict(t_cfg)
    resolved["batch_size"] = batch_size
    resolved["gradient_accumulation"] = grad_accum
    resolved["num_devices"] = num_devices
    resolved["max_steps"] = max_steps
    resolved["warmup_steps"] = warmup_steps
    resolved["max_lr"] = float(t_cfg.get("max_lr", 1e-4))
    resolved["min_lr"] = float(t_cfg.get("min_lr", 1e-5))
    resolved["weight_decay"] = float(t_cfg.get("weight_decay", 0.1))
    resolved["precision"] = t_cfg.get("precision", "bf16")
    return resolved


def get_upscaled_layer_mapping(src_layers: int, tgt_layers: int) -> list[int]:
    ratio = src_layers / tgt_layers
    return [min(int(i * ratio), src_layers - 1) for i in range(tgt_layers)]


def load_upscaled_weights_pytorch(tgt_model: nn.Module, tgt_cfg: dict, src_ckpt_path: str, src_cfg_path: str):
    """
    Initializes PyTorch 25M model weights by mapping depths and zero-padding dimensions from 12.5M weights.
    Applies exact variance-preserving RMSNorm scaling sqrt(d_old / d_new).
    """
    with open(src_cfg_path) as f:
        src_cfg = yaml.safe_load(f)["model"]
        
    src_layers = src_cfg["n_layers"]
    tgt_layers = tgt_cfg["n_layers"]
    layer_map = get_upscaled_layer_mapping(src_layers, tgt_layers)
    
    src_state = load_file(src_ckpt_path) if src_ckpt_path.endswith(".safetensors") else torch.load(src_ckpt_path, map_location="cpu")
    if "model_state_dict" in src_state:
        src_state = src_state["model_state_dict"]
        
    tgt_state = tgt_model.state_dict()
    for k, tgt_tensor in tgt_state.items():
        if "layers." in k:
            parts = k.split(".")
            tgt_idx = int(parts[1])
            src_idx = layer_map[tgt_idx]
            src_key = ".".join([parts[0], str(src_idx)] + parts[2:])
        else:
            src_key = k
            
        if src_key in src_state:
            src_tensor = src_state[src_key]
            if tgt_tensor.shape == src_tensor.shape:
                tgt_tensor.copy_(src_tensor)
            else:
                if "norm" in src_key and len(src_tensor.shape) == 1:
                    scale_factor = math.sqrt(src_tensor.shape[0] / tgt_tensor.shape[0])
                    padded = torch.ones_like(tgt_tensor)
                    padded[:src_tensor.shape[0]] = src_tensor * scale_factor
                    tgt_tensor.copy_(padded)
                else:
                    padded = torch.zeros_like(tgt_tensor)
                    slices = tuple(slice(0, min(s, t)) for s, t in zip(src_tensor.shape, tgt_tensor.shape))
                    padded[slices] = src_tensor[slices]
                    tgt_tensor.copy_(padded)
                
    tgt_model.load_state_dict(tgt_state)


def _train_worker(index: int, paradigm: str, config_path: str, src_tier: str = "12m", device=None, device_type: str = "tpu", mesh=None):
    """Worker function for single-process training. Supports SPMD data-parallel across TPU chips."""
    if device is None:
        device = create_device(device_type)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
        
    stem = Path(config_path).stem
    tier = "25m" if "25m" in stem else "12m"
    model_cfg = cfg["model"]
    train_cfg = resolve_training_params(cfg, device_type)
    
    # SPMD mesh means data-parallel across all chips; otherwise detect CUDA multi-GPU
    num_devices = train_cfg["num_devices"] if (mesh is not None or (device_type == "cuda" and torch.cuda.device_count() > 1)) else 1
    is_master = True
    
    # Global batch size: scales with TPU SPMD mesh or multi-GPU CUDA count
    dl_batch_size = train_cfg["batch_size"] * num_devices if (mesh is not None or (device_type == "cuda" and num_devices > 1)) else train_cfg["batch_size"]
    
    if is_master:
        print("\n" + "=" * 80, flush=True)
        spmd_tag = f" SPMD {num_devices}-chip" if mesh is not None else (f" DataParallel ({num_devices}x GPUs)" if device_type == "cuda" and num_devices > 1 else "")
        print(f"STARTING {paradigm.upper()} TRAINING FOR {stem} (Device: {device} [{device_type.upper()}{spmd_tag}])", flush=True)
        print(f"Steps: {train_cfg['max_steps']} | Global Batch: {dl_batch_size} (={train_cfg['batch_size']}x{num_devices}) | Grad Accum: {train_cfg['gradient_accumulation']} | LR: {train_cfg['max_lr']} | Warmup: {train_cfg['warmup_steps']}", flush=True)
        print("=" * 80, flush=True)
    
    is_causal = (paradigm.lower() == "ar")
    telos_cfg = TelosConfig(
        vocab_size=model_cfg["vocab_size"],
        d_model=model_cfg["d_model"],
        n_layers=model_cfg["n_layers"],
        n_heads=model_cfg["n_heads"],
        n_kv_heads=model_cfg["n_kv_heads"],
        seq_len=model_cfg["seq_len"],
        is_causal=is_causal
    )
    model = TelosTransformer(telos_cfg).to(device)
    
    if src_tier:
        src_stem = stem.replace(tier, src_tier)
        p_dir = "masked" if paradigm == "mdlm" else ("uniform" if paradigm == "undlm" else "ar")
        src_ckpt = f"checkpoints/{p_dir}/{src_tier}/{src_stem}/model.safetensors"
        src_cfg_path = f"configs/unified/{src_tier}/{src_stem}.yaml"
        
        # If exact matching source checkpoint is missing, search for fallback
        if not Path(src_ckpt).exists():
            for fallback_r in ["r35", "r30", "r25", "r20", "r15", "r10", "r1"]:
                fallback_stem = f"telos_{src_tier}_{fallback_r}"
                fallback_ckpt = f"checkpoints/{p_dir}/{src_tier}/{fallback_stem}/model.safetensors"
                fallback_cfg = f"configs/unified/{src_tier}/{fallback_stem}.yaml"
                if Path(fallback_ckpt).exists() and Path(fallback_cfg).exists():
                    if is_master:
                        print(f"  [PyTorch Upscaling] {src_stem} not found; using highest available source: {fallback_stem}", flush=True)
                    src_ckpt = fallback_ckpt
                    src_cfg_path = fallback_cfg
                    break

        if Path(src_ckpt).exists() and Path(src_cfg_path).exists():
            load_upscaled_weights_pytorch(model, model_cfg, src_ckpt, src_cfg_path)
            if is_master:
                print(f"  [PyTorch Upscaling] Success: Initialized from {src_ckpt} with zero-shock RMSNorm parity.", flush=True)
        else:
            if is_master:
                print(f"  [Training From Scratch] Source checkpoint not found; initializing {stem} with cold random weights.", flush=True)

    # Multi-GPU DataParallel for 2x T4 or cloud multi-GPU
    if device_type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    # NOTE: torch.compile(backend='openxla') is intentionally NOT used.
    # XLA lazy tensors already perform full graph tracing and fusion via xm.mark_step().
    # Layering torch.compile on top creates redundant compiled graph buffers that
    # consume ~2-4 GB extra HBM per chip, causing OOM on v5e (16GB HBM).
        
    decay_params = []
    nodecay_params = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            if p.dim() >= 2:
                decay_params.append(p)
            else:
                nodecay_params.append(p)
                
    optim_groups = [
        {"params": decay_params, "weight_decay": float(train_cfg.get("weight_decay", 0.1))},
        {"params": nodecay_params, "weight_decay": 0.0}
    ]
    
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=float(train_cfg["max_lr"]),
        betas=(0.9, 0.95)
    )
    scheduler = WarmupCosineLR(
        optimizer,
        warmup_steps=int(train_cfg.get("warmup_steps", 10)),
        max_steps=int(train_cfg["max_steps"]),
        max_lr=float(train_cfg["max_lr"]),
        min_lr=float(train_cfg.get("min_lr", 1e-5))
    )
    
    dataset_path = Path("data/python_corpus_2.5b.bin")
    if not dataset_path.exists():
        dataset_path = Path("data/python_corpus_1.7b.bin")
    if not dataset_path.exists():
        dataset_path = Path("data/python_corpus_mac.bin")
    if not dataset_path.exists():
        dataset_path = list(Path("data").glob("*.bin"))[0] if list(Path("data").glob("*.bin")) else Path("data/python_corpus_2.5b.bin")

    seq_len = model_cfg["seq_len"]
    if dataset_path.exists():
        file_bytes = dataset_path.stat().st_size
        if file_bytes % (seq_len * 4) == 0 and "2.5b" not in str(dataset_path):
            bytes_per_elem = 4
            np_dtype = np.uint32
        else:
            bytes_per_elem = 2
            np_dtype = np.uint16
            
        num_samples = file_bytes // (seq_len * bytes_per_elem)
        if is_master:
            print(f"  [Dataset] Loading {dataset_path} ({np_dtype.__name__}, {num_samples:,} sequences of len {seq_len})...", flush=True)
        dataset_np = np.fromfile(dataset_path, dtype=np_dtype).reshape(num_samples, seq_len)
    else:
        if is_master:
            print("  [Dataset] Warning: Binary dataset not found; using random samples.", flush=True)
        dataset_np = np.random.randint(0, model_cfg["vocab_size"], size=(1000, seq_len), dtype=np.uint32)

    num_samples = len(dataset_np)
    
    # On TPU, stage the dataset tensor directly into HBM to eliminate CPU host indexing & PCIe communication bottleneck
    # Uses int16 (vocab_size=8192 fits in 16 bits) to halve HBM footprint: ~3.4 GB vs ~6.8 GB at int32
    use_hbm_dataset = (device_type == "tpu")
    dataset_hbm = None
    if use_hbm_dataset:
        try:
            hbm_gb = num_samples * seq_len * 2 / (1024**3)  # 2 bytes per token (int16)
            if is_master:
                print(f"  [TPU HBM Staging] Moving dataset to TPU HBM as int16 ({hbm_gb:.2f} GB) to eliminate CPU host bottleneck...", flush=True)
            dataset_hbm = torch.from_numpy(dataset_np.astype(np.int16)).to(device)
            if mesh is not None:
                import torch_xla.distributed.spmd as xs
                xs.mark_sharding(dataset_hbm, mesh, (None, None))  # Fully replicated across all TPU chips
        except Exception as e:
            if is_master:
                print(f"  [TPU HBM Staging] Warning: Could not stage full dataset to HBM ({e}). Falling back to CPU RAM gather.", flush=True)
            use_hbm_dataset = False
            dataset_hbm = None
    
    # Configure automatic mixed precision (AMP) & GradScaler (supports native BF16 or FP16 on T4)
    amp_device = "xla" if device_type == "tpu" else ("cuda" if device_type == "cuda" else "cpu")
    supports_bf16 = (torch.cuda.is_bf16_supported() if hasattr(torch.cuda, "is_bf16_supported") and device_type == "cuda" else (device_type == "tpu"))
    use_bf16 = (train_cfg.get("precision", "bf16") == "bf16") and supports_bf16
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    use_amp = device_type in ("tpu", "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=(device_type == "cuda" and amp_dtype == torch.float16))

    model.train()
    grad_accum = int(train_cfg["gradient_accumulation"])
    max_steps = int(train_cfg["max_steps"])
    p_dir = "masked" if paradigm == "mdlm" else ("uniform" if paradigm == "undlm" else "ar")
    save_dir = Path(f"checkpoints/{p_dir}/{tier}/{stem}")
    if is_master:
        save_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    for step in range(1, max_steps + 1):
        optimizer.zero_grad()
        
        for mb in range(grad_accum):
            if use_hbm_dataset and dataset_hbm is not None:
                # 100% On-Chip TPU Gather: 0 CPU overhead, 0 PCIe transfer
                batch_indices = torch.randint(0, num_samples, (dl_batch_size,), device=device)
                x = dataset_hbm[batch_indices].long()
            else:
                # Vectorized batch indexing in RAM (instantaneous C-level slicing, 0ms overhead)
                batch_indices = np.random.randint(0, num_samples, size=dl_batch_size)
                raw_batch = dataset_np[batch_indices]
                x = torch.from_numpy(raw_batch).long().to(device)
            
            # SPMD: shard batch dimension across TPU chips for data parallelism
            if mesh is not None:
                import torch_xla.distributed.spmd as xs
                xs.mark_sharding(x, mesh, ('data', None))
            
            # Forward pass wrapped in hardware autocast to utilize TPU Matrix Multiply Units (MXUs) or CUDA Tensor Cores
            with torch.autocast(device_type=amp_device, dtype=amp_dtype, enabled=use_amp):
                if paradigm == "ar":
                    logits = model(x)
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = x[:, 1:].contiguous()
                    loss = nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                elif paradigm == "mdlm":
                    loss = mdlm_loss_pytorch(model, x, mask_token_id=1, vocab_size=model_cfg["vocab_size"])
                else:
                    loss = undlm_loss_pytorch(model, x, vocab_size=model_cfg["vocab_size"])
                    
                loss = loss / grad_accum
                
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
                
            # Bound graph compilation per microbatch on TPU
            if device_type == "tpu" and grad_accum > 1:
                import torch_xla.core.xla_model as xm
                xm.mark_step()
                
        # Perform gradient clipping and optimizer step across all device backends
        if device_type == "tpu":
            import torch_xla.core.xla_model as xm
            # Reduce gradients across SPMD mesh / TPU replicas prior to norm clipping
            xm.reduce_gradients(optimizer)
            # Clip gradient norm to 1.0 to prevent training divergence & NaN weight explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            xm.optimizer_step(optimizer)
            xm.mark_step()
        elif scaler.is_enabled():
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        scheduler.step()
        
        # Asynchronous non-blocking loss logging (zero pipeline stalls on TPU)
        if is_master and (step % 5 == 0 or step == max_steps or step <= 3):
            lr_curr = scheduler.get_last_lr()[0]
            toks_done = step * dl_batch_size * grad_accum * model_cfg["seq_len"]
            
            if device_type == "tpu":
                import torch_xla.core.xla_model as xm
                def _log_closure(step_idx, max_s, loss_val, lr, toks):
                    print(f"Step {step_idx:>5}/{max_s} | Loss: {loss_val.item():.4f} | LR: {lr:.2e} | Tokens: {toks/1e6:.1f}M", flush=True)
                loss_scaled = (loss * grad_accum).detach()
                xm.add_step_closure(_log_closure, (step, max_steps, loss_scaled, lr_curr, toks_done))
            else:
                step_loss = loss.item() * grad_accum
                print(f"Step {step:>5}/{max_steps} | Loss: {step_loss:.4f} | LR: {lr_curr:.2e} | Tokens: {toks_done/1e6:.1f}M", flush=True)
            
        if is_master and (step % int(cfg.get("checkpoint", {}).get("save_every_steps", 25)) == 0 or step == max_steps):
            ckpt_file = save_dir / f"checkpoint_step_{step}.safetensors"
            raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
            cpu_state = {k: v.detach().cpu().clone().contiguous() for k, v in raw_model.state_dict().items()}
            save_file(cpu_state, str(ckpt_file))
            
    # Save final model (unwrapped clean weights)
    if is_master:
        raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        cpu_state = {k: v.detach().cpu().clone().contiguous() for k, v in raw_model.state_dict().items()}
        save_file(cpu_state, str(save_dir / "model.safetensors"))
        with open(save_dir / "config.json", "w") as f:
            yaml.dump(model_cfg, f)
            
        print(f"FINISHED {paradigm.upper()} {stem} IN {(time.time() - start_time)/60.0:.2f} MINUTES!", flush=True)
        
    del model, optimizer, scheduler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_paradigm_pytorch(paradigm: str, config_path: str, src_tier: str = "12m", device=None, device_type: str = "tpu", mesh=None):
    _train_worker(0, paradigm, config_path, src_tier, device=device, device_type=device_type, mesh=mesh)


def run_upscaled_suite(ratios: list[str], tier: str = "25m", src_tier: str | None = None, hf_repo: str = "Kazenowoko/telos", device: str | None = None):
    """
    Downloads prerequisites from HuggingFace, trains upscaled models (25M or 50M), and uploads checkpoints.
    Uses SPMD data-parallel sharding on TPU for full 8-chip utilization.
    """
    if src_tier is None:
        src_tier = "25m" if tier == "50m" else "12m"

    # Step 1: Detect device type WITHOUT creating any device (critical for SPMD ordering)
    dev_type = detect_device_type(device)
    
    # Step 2: Initialize SPMD BEFORE creating any XLA device
    mesh = None
    if dev_type == "tpu":
        import torch_xla.runtime as xr
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.spmd as xs
        from torch_xla.distributed.spmd import Mesh
        
        xr.use_spmd()
        dev_obj = xm.xla_device()
        num_chips = xr.global_runtime_device_count()
        mesh = Mesh(np.arange(num_chips), (num_chips,), ('data',))
        print(f"[TPU SPMD] {num_chips}-chip data-parallel mesh initialized on {dev_obj}", flush=True)
    else:
        dev_obj = create_device(dev_type)
    
    print(f"Hardware initialization: Device = {dev_obj} ({dev_type.upper()})", flush=True)

    print("=" * 85, flush=True)
    print(f"SYNCING {src_tier.upper()} SOURCE WEIGHTS & DATASET FROM HUGGINGFACE ({hf_repo})...", flush=True)
    print("=" * 85, flush=True)
    snapshot_download(
        repo_id=hf_repo,
        local_dir="./",
        allow_patterns=[
            f"checkpoints/*/{src_tier}/*",
            f"checkpoints/*/{tier}/*",
            "data/python_corpus_*.bin",
            "data/python_corpus_1.7b.bin",
            "tokenizer*"
        ]
    )
    
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    
    for r in ratios:
        cfg_path = f"configs/unified/{tier}/telos_{tier}_{r}.yaml"
        print(f"\n>>>> EXECUTING UNIFIED {tier.upper()} RUN FOR RATIO: {r} (Upscaling from {src_tier}) <<<<", flush=True)
        for paradigm in ["ar", "mdlm", "undlm"]:
            train_paradigm_pytorch(paradigm=paradigm, config_path=cfg_path, src_tier=src_tier, device=dev_obj, device_type=dev_type, mesh=mesh)
            
            # Instantly upload individual model to Hugging Face
            p_dir = "masked" if paradigm == "mdlm" else ("uniform" if paradigm == "undlm" else "ar")
            model_dir = Path(f"checkpoints/{p_dir}/{tier}/telos_{tier}_{r}")
            if model_dir.exists() and os.environ.get("HF_TOKEN"):
                print(f"\n[Instant HF Export] Uploading {model_dir} to {hf_repo}...", flush=True)
                try:
                    api.upload_folder(
                        folder_path=str(model_dir),
                        path_in_repo=f"checkpoints/{p_dir}/{tier}/telos_{tier}_{r}",
                        repo_id=hf_repo,
                        repo_type="model",
                        allow_patterns=["*.safetensors", "*.json"]
                    )
                    print(f"[Instant HF Export] Success: {p_dir} telos_{tier}_{r} is now live on HuggingFace!", flush=True)
                    # Purge local checkpoint directory to free TPU disk storage for subsequent runs
                    shutil.rmtree(str(model_dir), ignore_errors=True)
                    print(f"[Storage Cleanup] Deleted {model_dir} to reclaim disk space.", flush=True)
                except Exception as e:
                    print(f"[Instant HF Export] Upload warning: {e}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Universal Upscaling Suite Runner on TPU / Cloud GPU (25M / 50M)")
    parser.add_argument("--tier", type=str, default="25m", choices=["12m", "25m", "50m"], help="Model parameter tier to train ('25m' or '50m')")
    parser.add_argument("--src-tier", type=str, default=None, help="Source tier to upscale from (defaults to '25m' for 50m, '12m' for 25m)")
    parser.add_argument("--ratios", nargs="+", default=None, help="Ratios to train (defaults to 'r1 r35 r40 r45 r50' for 50m, 'r1 r10 r15 r20 r25 r30 r35' for 25m)")
    parser.add_argument("--hf-repo", type=str, default="Kazenowoko/telos", help="Hugging Face Model Repository")
    parser.add_argument("--device", type=str, default="tpu", help="Device to use ('tpu', 'cuda', 'cpu')")
    args = parser.parse_args()
    
    tier = args.tier
    src_tier = args.src_tier
    if src_tier is None:
        src_tier = "25m" if tier == "50m" else ("12m" if tier == "25m" else None)

    ratios = args.ratios
    if ratios is None:
        ratios = ["r1", "r35", "r40", "r45", "r50"] if tier == "50m" else ["r1", "r10", "r15", "r20", "r25", "r30", "r35"]

    run_upscaled_suite(ratios=ratios, tier=tier, src_tier=src_tier, hf_repo=args.hf_repo, device=args.device)


if __name__ == "__main__":
    main()
