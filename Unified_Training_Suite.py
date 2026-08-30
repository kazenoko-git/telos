import os
import sys
import shutil
import time
import math
import gc
import yaml
from pathlib import Path
import numpy as np

# Ensure project root is on PATH
project_root = Path.cwd()
while not (project_root / "mdiff").exists() and project_root.parent != project_root:
    project_root = project_root.parent
os.chdir(project_root)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# -----------------------------------------------------------------------------
# 1. Hardware Detection & Dependency Sync
# -----------------------------------------------------------------------------
def detect_backend(device_override=None):
    """Detects backend WITHOUT creating any XLA device (safe for SPMD init ordering)."""
    if device_override:
        d = device_override.lower()
        if d == "mlx": return "mlx", "metal"
        if d in ("tpu", "xla"): return "pytorch", "tpu"
        if "cuda" in d: return "pytorch", "cuda"
        if "mps" in d: return "pytorch", "mps"
        return "pytorch", "cpu"
    # Check TPU by environment/device files without creating an XLA device prematurely
    if os.environ.get("PJRT_DEVICE") == "TPU" or os.path.exists("/dev/accel0"):
        return "pytorch", "tpu"
    try:
        import torch
        if torch.cuda.is_available(): return "pytorch", "cuda"
    except Exception:
        pass
    if sys.platform == "darwin":
        try:
            import mlx.core as mx
            return "mlx", "metal"
        except Exception:
            pass
    return "pytorch", "cpu"

def resolve_training_params(cfg, device_type="mac"):
    """
    Dynamically computes batch size, gradient accumulation, max steps, and warmup steps
    from device-independent total_tokens budget and hardware-specific execution profiles.
    """
    t_cfg = cfg["training"]
    m_cfg = cfg["model"]
    seq_len = m_cfg.get("seq_len", 512)

    # Normalize device key
    if device_type in ("tpu", "xla"):
        dev_key = "tpu"
    elif device_type in ("mac", "metal", "mlx", "mps"):
        dev_key = "mac"
    elif "cuda" in str(device_type).lower():
        dev_key = "gpu"
    else:
        dev_key = "mac"

    if dev_key in t_cfg and isinstance(t_cfg[dev_key], dict):
        dev_profile = t_cfg[dev_key]
        batch_size = int(dev_profile.get("batch_size", 32))
        grad_accum = int(dev_profile.get("gradient_accumulation", 1))
        num_devices = int(dev_profile.get("num_devices", 1))
    else:
        batch_size = int(t_cfg.get("batch_size", 32))
        grad_accum = int(t_cfg.get("gradient_accumulation", 1))
        num_devices = int(t_cfg.get("num_devices", 1))

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

def sync_hf_assets(hf_repo="Kazenowoko/telos"):
    data_bins = list(Path("data").glob("*.bin")) if Path("data").exists() else []
    if not data_bins or not Path("checkpoints/masked/12m").exists():
        print(f"Syncing prerequisites from Hugging Face ({hf_repo})...")
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=hf_repo, local_dir="./", allow_patterns=["checkpoints/ar/12m/*/model.safetensors", "checkpoints/ar/12m/*/config.json", "checkpoints/masked/12m/*/model.safetensors", "checkpoints/masked/12m/*/config.json", "checkpoints/uniform/12m/*/model.safetensors", "checkpoints/uniform/12m/*/config.json", "data/python_corpus_1.7b.bin", "tokenizer*"])
            print("HF sync complete!")
        except Exception as e:
            print(f"HF sync notice: {e}")

def upload_to_hf(model_dir, path_in_repo, hf_repo="Kazenowoko/telos"):
    token = os.environ.get("HF_TOKEN")
    if token and Path(model_dir).exists():
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=token)
            api.upload_folder(folder_path=str(model_dir), path_in_repo=path_in_repo, repo_id=hf_repo, repo_type="model", allow_patterns=["*.safetensors", "*.json", "*.yaml"])
            print(f"[HF Upload Success] {path_in_repo}")
            # Purge local checkpoint directory to free TPU disk storage for subsequent runs
            shutil.rmtree(str(model_dir), ignore_errors=True)
            print(f"[Storage Cleanup] Deleted {model_dir} to reclaim disk space.")
        except Exception as e:
            print(f"[HF Upload Warning] {e}")

# -----------------------------------------------------------------------------
# 2. PyTorch Training Engine (TPU SPMD / CUDA / MPS)
# -----------------------------------------------------------------------------
class WarmupCosineLR:
    def __init__(self, optimizer, warmup_steps, max_steps, max_lr, min_lr=1e-5):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.current_step = 0
    def step(self):
        self.current_step += 1
        s = self.current_step
        if s < self.warmup_steps:
            lr = self.max_lr * s / max(1, self.warmup_steps)
        elif s > self.max_steps:
            lr = self.min_lr
        else:
            decay = (s - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
            lr = self.min_lr + 0.5 * (1.0 + math.cos(math.pi * decay)) * (self.max_lr - self.min_lr)
        device = self.optimizer.param_groups[0]["params"][0].device
        tensor_lr = torch.tensor(lr, dtype=torch.float32, device=device)
        for g in self.optimizer.param_groups:
            g["lr"] = tensor_lr
    def get_last_lr(self):
        return [self.optimizer.param_groups[0]["lr"]]

def mdlm_loss_pytorch(model, clean_targets, mask_token_id=1, vocab_size=8192):
    import torch
    import torch.nn as nn
    B, T = clean_targets.shape
    device = clean_targets.device
    t_values = torch.rand((B, 1), device=device)
    mask_matrix = torch.rand((B, T), device=device) < t_values
    masked_ids = torch.where(mask_matrix, torch.full((B, T), mask_token_id, dtype=torch.long, device=device), clean_targets)
    logits = model(masked_ids)
    ce = nn.functional.cross_entropy(logits.view(-1, vocab_size).float(), clean_targets.view(-1), reduction="none").view(B, T)
    masked_ce = ce * mask_matrix.float()
    per_example_ce = torch.sum(masked_ce, dim=1) / torch.clamp(torch.sum(mask_matrix.float(), dim=1), min=1.0, max=float(T))
    return torch.mean(per_example_ce * (1.0 / torch.clamp(t_values.squeeze(-1), min=1e-3, max=1.0)))

def undlm_loss_pytorch(model, clean_targets, vocab_size=8192):
    import torch
    import torch.nn as nn
    B, T = clean_targets.shape
    device = clean_targets.device
    t_values = torch.rand((B, 1), device=device)
    noisy_ids = torch.where(torch.rand((B, T), device=device) < t_values, torch.randint(0, vocab_size, size=(B, T), device=device), clean_targets)
    logits = model(noisy_ids)
    ce = nn.functional.cross_entropy(logits.view(-1, vocab_size).float(), clean_targets.view(-1), reduction="none").view(B, T)
    return torch.mean(torch.mean(ce, dim=1) * (1.0 / torch.clamp(t_values.squeeze(-1), min=1e-3, max=1.0)))

def load_upscaled_pytorch(tgt_model, tgt_cfg, src_ckpt_path, src_cfg_path):
    import torch
    from safetensors.torch import load_file
    with open(src_cfg_path) as f: src_cfg = yaml.safe_load(f)["model"]
    ratio = src_cfg["n_layers"] / tgt_cfg["n_layers"]
    layer_map = [min(int(i * ratio), src_cfg["n_layers"] - 1) for i in range(tgt_cfg["n_layers"])]
    src_state = load_file(src_ckpt_path) if src_ckpt_path.endswith(".safetensors") else torch.load(src_ckpt_path, map_location="cpu")
    if "model_state_dict" in src_state: src_state = src_state["model_state_dict"]
    tgt_state = tgt_model.state_dict()
    for k, tgt_tensor in tgt_state.items():
        if "layers." in k:
            parts = k.split(".")
            src_key = ".".join([parts[0], str(layer_map[int(parts[1])])] + parts[2:])
        else: src_key = k
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
    print(f"  [PyTorch Upscaling] Loaded {src_ckpt_path} with zero-shock RMSNorm parity!")

def train_pytorch(paradigm, config_path, upscale_from_tier=None, device_type="cuda"):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from safetensors.torch import save_file
    from mdiff.model.transformer import TelosTransformer, TelosConfig
    
    # Use pre-initialized SPMD device/mesh for TPU; standard device for others
    mesh = None
    num_devices_actual = 1
    if device_type == "tpu":
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.spmd as xs
        device = _tpu_device
        mesh = _tpu_mesh
        num_devices_actual = _num_tpu_devices
    elif device_type == "cuda": device = torch.device("cuda")
    elif device_type == "mps": device = torch.device("mps")
    else: device = torch.device("cpu")

    with open(config_path) as f: cfg = yaml.safe_load(f)
    stem = Path(config_path).stem
    tier = "25m" if "25m" in stem else ("50m" if "50m" in stem else "12m")
    m_cfg = cfg["model"]
    t_cfg = resolve_training_params(cfg, device_type)
    p_dir = "masked" if paradigm.lower() == "mdlm" else ("uniform" if paradigm.lower() == "undlm" else "ar")

    # SPMD: DataLoader produces global batch (per_device * num_devices)
    dl_batch_size = t_cfg["batch_size"] * t_cfg["num_devices"] if mesh is not None else t_cfg["batch_size"]

    spmd_tag = f" SPMD {num_devices_actual}-chip" if mesh is not None else ""
    print(f"\n>>> {paradigm.upper()} Training ({stem}) on {device_type.upper()}{spmd_tag} <<<")
    print(f"  [Config Resolved] Steps: {t_cfg['max_steps']} | Global Batch: {dl_batch_size} (={t_cfg['batch_size']}x{t_cfg['num_devices']}) | Accum: {t_cfg['gradient_accumulation']} | Warmup: {t_cfg['warmup_steps']}")
    telos_cfg = TelosConfig(vocab_size=m_cfg["vocab_size"], d_model=m_cfg["d_model"], n_layers=m_cfg["n_layers"], n_heads=m_cfg["n_heads"], n_kv_heads=m_cfg["n_kv_heads"], seq_len=m_cfg["seq_len"], is_causal=(paradigm.lower() == "ar"))
    model = TelosTransformer(telos_cfg).to(device)

    if upscale_from_tier:
        src_stem = stem.replace(tier, upscale_from_tier)
        src_ckpt = f"checkpoints/{p_dir}/{upscale_from_tier}/{src_stem}/model.safetensors"
        src_cfg = f"configs/unified/{upscale_from_tier}/{src_stem}.yaml"
        if not Path(src_ckpt).exists():
            fallback_stem = f"telos_{upscale_from_tier}_r25"
            fallback_ckpt = f"checkpoints/{p_dir}/{upscale_from_tier}/{fallback_stem}/model.safetensors"
            fallback_cfg = f"configs/unified/{upscale_from_tier}/{fallback_stem}.yaml"
            if Path(fallback_ckpt).exists() and Path(fallback_cfg).exists():
                print(f"  [Upscaling] {src_stem} not found; using highest available source: {fallback_stem}")
                src_ckpt = fallback_ckpt
                src_cfg = fallback_cfg
        if Path(src_ckpt).exists():
            load_upscaled_pytorch(model, m_cfg, src_ckpt, src_cfg)

    decay_params = []
    nodecay_params = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            if p.dim() >= 2:
                decay_params.append(p)
            else:
                nodecay_params.append(p)
    optim_groups = [
        {"params": decay_params, "weight_decay": float(t_cfg.get("weight_decay", 0.01))},
        {"params": nodecay_params, "weight_decay": 0.0}
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=float(t_cfg["max_lr"]), betas=(0.9, 0.95))
    scheduler = WarmupCosineLR(optimizer, warmup_steps=int(t_cfg["warmup_steps"]), max_steps=int(t_cfg["max_steps"]), max_lr=float(t_cfg["max_lr"]), min_lr=float(t_cfg.get("min_lr", 1e-5)))

    data_path = Path("data/python_corpus_1.7b.bin")
    if not data_path.exists(): data_path = Path("data/python_corpus_mac.bin")
    if not data_path.exists(): data_path = list(Path("data").glob("*.bin"))[0] if list(Path("data").glob("*.bin")) else Path("data/python_corpus_1.7b.bin")
    
    seq_len = m_cfg["seq_len"]
    if data_path.exists():
        dataset = np.memmap(data_path, dtype=np.uint32, mode='r').reshape(data_path.stat().st_size // (seq_len * 4), seq_len) # In-memory to prevent IO thrashing
    else:
        dataset = np.random.randint(0, m_cfg["vocab_size"], size=(1000, seq_len), dtype=np.uint32)
    num_samples = len(dataset)
    amp_device = "xla" if device_type == "tpu" else ("cuda" if device_type == "cuda" else "cpu")
    use_amp = (t_cfg.get("precision", "bf16") == "bf16") and (device_type in ("tpu", "cuda"))
    amp_dtype = torch.bfloat16 if t_cfg.get("precision", "bf16") == "bf16" else torch.float16

    model.train()
    grad_accum, max_steps = int(t_cfg["gradient_accumulation"]), int(t_cfg["max_steps"])
    save_dir = Path(f"checkpoints/{p_dir}/{tier}/{stem}")
    save_dir.mkdir(parents=True, exist_ok=True)

    for step in range(1, max_steps + 1):
        optimizer.zero_grad()
        for mb in range(grad_accum):
            # Vectorized batch indexing directly from RAM (0ms overhead)
            batch_indices = np.random.randint(0, num_samples, size=dl_batch_size)
            raw_batch = dataset[batch_indices]
            x = torch.from_numpy(raw_batch).long().to(device)
            # SPMD: shard batch dimension across TPU chips for data parallelism
            if mesh is not None:
                xs.mark_sharding(x, mesh, ('data', None))
            with torch.autocast(device_type=amp_device, dtype=amp_dtype, enabled=use_amp):
                if paradigm.lower() == "ar":
                    logits = model(x)
                    loss = nn.functional.cross_entropy(logits[:, :-1, :].contiguous().view(-1, logits.size(-1)), x[:, 1:].contiguous().view(-1))
                elif paradigm.lower() == "mdlm":
                    loss = mdlm_loss_pytorch(model, x, mask_token_id=1, vocab_size=m_cfg["vocab_size"])
                else:
                    loss = undlm_loss_pytorch(model, x, vocab_size=m_cfg["vocab_size"])
                loss = loss / grad_accum
            loss.backward()
        if device_type == "tpu":
            # Reduce across TPU replicas and clip gradient norm to prevent explosion
            xm.reduce_gradients(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            xm.optimizer_step(optimizer)
            xm.mark_step()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()
        if step % 20 == 0 or step == max_steps or step <= 5:
            toks_done = step * dl_batch_size * grad_accum * seq_len
            step_loss = loss.item() * grad_accum
            print(f"  [Step {step:>4}/{max_steps}] Loss: {step_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e} | Tokens: {toks_done/1e6:.1f}M", flush=True)

    cpu_state = {k: v.detach().cpu().clone().contiguous() for k, v in model.state_dict().items()}
    save_file(cpu_state, str(save_dir / "model.safetensors"))
    with open(save_dir / "config.json", "w") as f: json.dump(m_cfg, f, indent=2)
    del model, optimizer, scheduler
    gc.collect()

# -----------------------------------------------------------------------------
# 3. Apple Silicon MLX Training Engine (Metal)
# -----------------------------------------------------------------------------
def train_mlx(paradigm, config_path, upscale_from_tier=None):
    from pathlib import Path
    import mlx.core as mx
    import gc
    import yaml
    from mdiff.model.mlx_components import MLXTelosTransformer, load_upscaled_weights
    from mdiff.training.trainer import TelosMLXTrainer as MDLM_Trainer
    from undiff.training.trainer import TelosMLXUNDLMTrainer as UNDLM_Trainer
    from ar.model.mlx_components import MLXCausalTransformer
    from ar.training.trainer import TelosMLXARTrainer as AR_Trainer
    
    with open(config_path) as f: cfg = yaml.safe_load(f)
    stem = Path(config_path).stem
    tier = "25m" if "25m" in stem else ("50m" if "50m" in stem else "12m")
    p_dir = "masked" if paradigm.lower() == "mdlm" else ("uniform" if paradigm.lower() == "undlm" else "ar")
    p_cfg = yaml.safe_load(yaml.dump(cfg))
    p_cfg["training"] = resolve_training_params(cfg, "mac")
    p_cfg["checkpoint"] = {"dir": f"checkpoints/{p_dir}/{tier}/{stem}", "save_every_steps": cfg.get("checkpoint", {}).get("save_every_steps", 100)}

    print(f"\n>>> MLX {paradigm.upper()} Training ({stem}) on Apple Silicon Metal <<<")
    print(f"  [Config Resolved] Steps: {p_cfg['training']['max_steps']} | Batch: {p_cfg['training']['batch_size']} | Accum: {p_cfg['training']['gradient_accumulation']} | Warmup: {p_cfg['training']['warmup_steps']}")
    
    if paradigm.lower() == "ar":
        model = MLXCausalTransformer(**p_cfg["model"])
        TrainerClass = AR_Trainer
    elif paradigm.lower() == "mdlm":
        model = MLXTelosTransformer(**p_cfg["model"])
        TrainerClass = MDLM_Trainer
    else:
        model = MLXTelosTransformer(**p_cfg["model"])
        TrainerClass = UNDLM_Trainer

    if upscale_from_tier:
        src_stem = stem.replace(tier, upscale_from_tier)
        src_ckpt = f"checkpoints/{p_dir}/{upscale_from_tier}/{src_stem}/model.safetensors"
        src_cfg = f"configs/unified/{upscale_from_tier}/{src_stem}.yaml"
        if not Path(src_ckpt).exists():
            fallback_stem = f"telos_{upscale_from_tier}_r25"
            fallback_ckpt = f"checkpoints/{p_dir}/{upscale_from_tier}/{fallback_stem}/model.safetensors"
            fallback_cfg = f"configs/unified/{upscale_from_tier}/{fallback_stem}.yaml"
            if Path(fallback_ckpt).exists() and Path(fallback_cfg).exists():
                print(f"  [Upscaling] {src_stem} not found; using highest available source: {fallback_stem}")
                src_ckpt = fallback_ckpt
                src_cfg = fallback_cfg
        if Path(src_ckpt).exists():
            load_upscaled_weights(model, p_cfg["model"], src_ckpt, src_cfg)

    model.set_dtype(mx.bfloat16)
    trainer = TrainerClass(model, p_cfg)
    trainer.train()
    
    del model, trainer
    gc.collect()
    mx.clear_cache()

# -----------------------------------------------------------------------------
# 4. Unified Execution Entrypoints
# -----------------------------------------------------------------------------
def run_unified_training_suite(config_path, upscale_from_tier=None, device=None, hf_repo="Kazenowoko/telos"):
    backend, device_type = detect_backend(device)
    print("=" * 85)
    print(f"TÉLOS UNIFIED 3-PARADIGM SUITE | Backend: {backend.upper()} | Device: {device_type.upper()}")
    print("=" * 85)
    if device_type in ("tpu", "cuda"):
        sync_hf_assets(hf_repo=hf_repo)
    stem = Path(config_path).stem
    tier = "25m" if "25m" in stem else ("50m" if "50m" in stem else "12m")
    for p in ["ar", "mdlm", "undlm"]:
        if backend == "mlx": train_mlx(p, config_path, upscale_from_tier=upscale_from_tier)
        else: train_pytorch(p, config_path, upscale_from_tier=upscale_from_tier, device_type=device_type)
        p_dir = "masked" if p == "mdlm" else ("uniform" if p == "undlm" else "ar")
        upload_to_hf(Path(f"checkpoints/{p_dir}/{tier}/{stem}"), f"checkpoints/{p_dir}/{tier}/{stem}", hf_repo=hf_repo)

def :
    for r in ratios:
        cfg_path = f"configs/unified/25m/telos_25m_{r}.yaml"
        print(f"\n>>>> RUNNING ZERO-SHOCK 25M {r.upper()} <<<<")
        run_unified_training_suite(cfg_path, upscale_from_tier="12m", device=device, hf_repo=hf_repo)

# -----------------------------------------------------------------------------
# 5. Environment Initialization (SPMD must happen BEFORE any xm.xla_device())
# -----------------------------------------------------------------------------
backend, device_type = detect_backend()
_tpu_device = None
_tpu_mesh = None
_num_tpu_devices = 1

if device_type == "tpu":
    import torch_xla.runtime as xr
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.spmd as xs
    from torch_xla.distributed.spmd import Mesh
    
    xr.use_spmd()
    _tpu_device = xm.xla_device()
    _num_tpu_devices = xr.global_runtime_device_count()
    _tpu_mesh = Mesh(np.arange(_num_tpu_devices), (_num_tpu_devices,), ('data',))
    print(f"[TPU SPMD] {_num_tpu_devices}-chip data-parallel mesh initialized on {_tpu_device}")

print(f"[Environment Initialized] Active Backend: {backend.upper()} ({device_type.upper()})")


# 12.5M 1:1 Ratio (~12.5M tokens, 96 steps)

# 12.5M 1:5 Ratio (~62.5M tokens, 480 steps)

# 12.5M 1:10 Ratio (~125M tokens, 960 steps)

# 12.5M 1:15 Ratio (~188M tokens, 1440 steps)

# 12.5M 1:20 Ratio (~250M tokens, 1920 steps)

# 12.5M 1:25 Ratio (~314M tokens, 2400 steps)
# 
# 12.5M 1:30 Ratio (~376M tokens, 2880 steps)
# 

# [1/7] 25M 1:1 Ratio (~26.2M tokens, 100 steps) — Upscaled from 12.5M 1:1


# [2/7] 25M 1:10 Ratio (~261.6M tokens, 998 steps) — Upscaled from 12.5M 1:10


# [3/7] 25M 1:15 Ratio (~392.2M tokens, 1496 steps) — Upscaled from 12.5M 1:15


# [4/7] 25M 1:20 Ratio (~523.0M tokens, 1995 steps) — Upscaled from 12.5M 1:20


# [5/7] 25M 1:25 Ratio (~653.8M tokens, 2494 steps) — Upscaled from 12.5M 1:25


# [6/7] 25M 1:30 Ratio (~784.3M tokens, 2992 steps) — Upscaled from 12.5M 1:30


# [7/7] 25M 1:35 Ratio (~915.1M tokens, 3491 steps) — Upscaled from 12.5M 1:30


# Execute full 25M suite: 1:1, 1:10, 1:15, 1:20, 1:25, 1:30, 1:35

