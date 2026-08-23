"""
Google Colab / Kaggle TPU & Cloud GPU Training Script for 25M Upscaled Suite.

Supports:
- Full 8-Core TPU v5e-8 multiprocessing via torch_xla.distributed.xla_multiprocessing (xmp.spawn)
- High-throughput ParallelLoader / MpDeviceLoader across all 8 cores
- Hardware gradient All-Reduce via xm.optimizer_step
- Zero-shock upscaling with invariant RMSNorm scaling
"""

import os
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


class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
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
        return [lr for _ in self.base_lrs]


def mdlm_loss_pytorch(model, clean_targets, mask_token_id=4, vocab_size=8192):
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


def resolve_device(device_str: str | None = None):
    """Detects device type in parent process without initializing XLA runtime."""
    if device_str:
        if device_str.lower() in ("tpu", "xla"):
            return "tpu"
        elif "cuda" in device_str.lower():
            return "cuda"
        elif "mps" in device_str.lower():
            return "mps"
        else:
            return "cpu"
            
    if os.environ.get("PJRT_DEVICE") == "TPU":
        return "tpu"
    if "torch_xla" in sys.modules or os.path.exists("/dev/accel0"):
        return "tpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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


def _train_worker(index: int, paradigm: str, config_path: str, src_tier: str = "12m", device_type: str = "tpu"):
    """Worker function executed inside each core process."""
    if device_type == "tpu":
        import torch_xla.core.xla_model as xm
        import torch_xla.runtime as xr
        import torch_xla
        import torch_xla.distributed.parallel_loader as pl
        device = xm.xla_device()
        rank = xr.global_ordinal()
        world_size = xr.world_size()
    elif device_type == "cuda":
        device = torch.device("cuda")
        rank = 0
        world_size = 1
    else:
        device = torch.device("cpu")
        rank = 0
        world_size = 1

    is_master = (rank == 0)
    
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
        
    stem = Path(config_path).stem
    tier = "25m" if "25m" in stem else "12m"
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    
    if is_master:
        print("\n" + "=" * 80, flush=True)
        print(f"STARTING {paradigm.upper()} TRAINING FOR {stem} (Device: {device} [{device_type.upper()} x {world_size} cores])", flush=True)
        print(f"Steps: {train_cfg['max_steps']} | Batch/Core: {train_cfg['batch_size']} | Grad Accum: {train_cfg['gradient_accumulation']} | LR: {train_cfg['max_lr']}", flush=True)
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
        
        if not Path(src_ckpt).exists():
            fallback_stem = f"telos_{src_tier}_r25"
            fallback_ckpt = f"checkpoints/{p_dir}/{src_tier}/{fallback_stem}/model.safetensors"
            fallback_cfg = f"configs/unified/{src_tier}/{fallback_stem}.yaml"
            if Path(fallback_ckpt).exists() and Path(fallback_cfg).exists():
                if is_master:
                    print(f"  [Upscaling] {src_stem} not found; using highest available source: {fallback_stem}", flush=True)
                src_ckpt = fallback_ckpt
                src_cfg_path = fallback_cfg

        if Path(src_ckpt).exists() and Path(src_cfg_path).exists():
            load_upscaled_weights_pytorch(model, model_cfg, src_ckpt, src_cfg_path)
            if is_master:
                print("  [PyTorch Upscaling] Success: Model initialized with zero-shock RMSNorm parity.", flush=True)
        else:
            if is_master:
                print(f"  [Upscaling] Initializing {stem} from cold random weights.", flush=True)
        
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["max_lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.1)),
        betas=(0.9, 0.95)
    )
    scheduler = WarmupCosineLR(
        optimizer,
        warmup_steps=int(train_cfg.get("warmup_steps", 10)),
        max_steps=int(train_cfg["max_steps"]),
        max_lr=float(train_cfg["max_lr"]),
        min_lr=float(train_cfg.get("min_lr", 1e-5))
    )
    
    dataset_path = Path("data/python_corpus_1.7b.bin")
    if not dataset_path.exists():
        dataset_path = Path("data/python_corpus_mac.bin")
    if not dataset_path.exists():
        dataset_path = list(Path("data").glob("*.bin"))[0] if list(Path("data").glob("*.bin")) else Path("data/python_corpus_1.7b.bin")

    seq_len = model_cfg["seq_len"]
    if dataset_path.exists():
        num_samples = dataset_path.stat().st_size // (seq_len * 4)
        if is_master:
            print(f"  [Dataset] Memory-mapping {dataset_path} ({num_samples:,} samples)...", flush=True)
        dataset = np.memmap(dataset_path, dtype=np.uint32, mode="r", shape=(num_samples, seq_len))
    else:
        if is_master:
            print("  [Dataset] Warning: Binary dataset not found; using random samples.", flush=True)
        dataset = np.random.randint(0, model_cfg["vocab_size"], size=(1000, seq_len), dtype=np.uint32)

    if device_type == "tpu" and world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True
        )
        base_loader = DataLoader(dataset, batch_size=train_cfg["batch_size"], sampler=sampler, num_workers=0, drop_last=True)
        train_loader = pl.MpDeviceLoader(base_loader, device)
    else:
        train_loader = DataLoader(dataset, batch_size=train_cfg["batch_size"], shuffle=True, drop_last=True)
        
    loader_iter = iter(train_loader)
    
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
            try:
                raw_batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(train_loader)
                raw_batch = next(loader_iter)
                
            x = raw_batch.long().to(device) if isinstance(raw_batch, torch.Tensor) else torch.from_numpy(np.array(raw_batch, copy=True)).long().to(device)
            
            if paradigm == "ar":
                logits = model(x)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = x[:, 1:].contiguous()
                loss = nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            elif paradigm == "mdlm":
                loss = mdlm_loss_pytorch(model, x, mask_token_id=4, vocab_size=model_cfg["vocab_size"])
            else:
                loss = undlm_loss_pytorch(model, x, vocab_size=model_cfg["vocab_size"])
                
            loss = loss / grad_accum
            loss.backward()
            
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if device_type == "tpu":
            import torch_xla.core.xla_model as xm
            import torch_xla
            xm.optimizer_step(optimizer)
            torch_xla.sync()
        else:
            optimizer.step()
            
        scheduler.step()
        
        if is_master and (step % 5 == 0 or step == max_steps or step <= 3):
            lr_curr = scheduler.get_last_lr()[0]
            toks_done = step * train_cfg["batch_size"] * grad_accum * model_cfg["seq_len"] * world_size
            step_loss = loss.item() * grad_accum
            print(f"Step {step:>5}/{max_steps} | Loss: {step_loss:.4f} | LR: {lr_curr:.2e} | Tokens: {toks_done/1e6:.1f}M", flush=True)
            
        if is_master and (step % int(cfg.get("checkpoint", {}).get("save_every_steps", 25)) == 0 or step == max_steps):
            ckpt_file = save_dir / f"checkpoint_step_{step}.safetensors"
            cpu_state = {k: v.detach().cpu().clone().contiguous() for k, v in model.state_dict().items()}
            save_file(cpu_state, str(ckpt_file))
            
    # Save final model
    if is_master:
        cpu_state = {k: v.detach().cpu().clone().contiguous() for k, v in model.state_dict().items()}
        save_file(cpu_state, str(save_dir / "model.safetensors"))
        with open(save_dir / "config.json", "w") as f:
            yaml.dump(model_cfg, f)
            
        print(f"FINISHED {paradigm.upper()} {stem} IN {(time.time() - start_time)/60.0:.2f} MINUTES!", flush=True)
        
    del model, optimizer, scheduler
    gc.collect()


def train_paradigm_pytorch(paradigm: str, config_path: str, src_tier: str = "12m", device_arg: str | None = None):
    device_type = resolve_device(device_arg)
    if device_type == "tpu":
        import torch_xla.distributed.xla_multiprocessing as xmp
        xmp.spawn(
            _train_worker,
            args=(paradigm, config_path, src_tier, device_type),
            start_method="fork"
        )
    else:
        _train_worker(0, paradigm, config_path, src_tier, device_type)


def run_full_25m_suite(ratios: list[str], hf_repo: str = "Kazenowoko/telos", device: str | None = None):
    """
    Downloads prerequisites from HuggingFace, trains 25M upscaled models, and uploads checkpoints.
    """
    print("=" * 85, flush=True)
    print(f"SYNCING 12.5M SOURCE WEIGHTS & DATASET FROM HUGGINGFACE ({hf_repo})...", flush=True)
    print("=" * 85, flush=True)
    snapshot_download(
        repo_id=hf_repo,
        local_dir="./",
        allow_patterns=[
            "checkpoints/ar/12m/*",
            "checkpoints/masked/12m/*",
            "checkpoints/uniform/12m/*",
            "data/python_corpus_1.7b.bin",
            "tokenizer*"
        ]
    )
    
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    
    for r in ratios:
        cfg_path = f"configs/unified/25m/telos_25m_{r}.yaml"
        print(f"\n>>>> EXECUTING UNIFIED 25M RUN FOR RATIO: {r} <<<<", flush=True)
        for paradigm in ["ar", "mdlm", "undlm"]:
            train_paradigm_pytorch(paradigm=paradigm, config_path=cfg_path, src_tier="12m", device_arg=device)
            
            # Instantly upload individual model to Hugging Face
            p_dir = "masked" if paradigm == "mdlm" else ("uniform" if paradigm == "undlm" else "ar")
            model_dir = Path(f"checkpoints/{p_dir}/25m/telos_25m_{r}")
            if model_dir.exists() and os.environ.get("HF_TOKEN"):
                print(f"\n[Instant HF Export] Uploading {model_dir} to {hf_repo}...", flush=True)
                try:
                    api.upload_folder(
                        folder_path=str(model_dir),
                        path_in_repo=f"checkpoints/{p_dir}/25m/telos_25m_{r}",
                        repo_id=hf_repo,
                        repo_type="model",
                        allow_patterns=["*.safetensors", "*.json"]
                    )
                    print(f"[Instant HF Export] Success: {p_dir} telos_25m_{r} is now live on HuggingFace!", flush=True)
                except Exception as e:
                    print(f"[Instant HF Export] Upload warning: {e}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="25M Upscaling Suite Runner on TPU / Cloud GPU")
    parser.add_argument("--ratios", nargs="+", default=["r1", "r10", "r15", "r20", "r25", "r30", "r35"], help="Ratios to train (e.g. r1 r10 r15 r20 r25 r30 r35)")
    parser.add_argument("--hf-repo", type=str, default="Kazenowoko/telos", help="Hugging Face Model Repository")
    parser.add_argument("--device", type=str, default="tpu", help="Device to use ('tpu', 'cuda', 'cpu')")
    args = parser.parse_args()
    
    run_full_25m_suite(ratios=args.ratios, hf_repo=args.hf_repo, device=args.device)


if __name__ == "__main__":
    main()
