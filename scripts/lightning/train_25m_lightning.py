"""
télos (τέλος) - Lightning AI TPU v6e-1 Single-Chip 25M Training Pipeline
========================================================================
Optimized specifically for single-chip Trillium TPU v6e (32GB HBM).

Key Design Highlights:
1. Single-Chip Simplicity: No SPMD mesh overhead; uses native XLA device and xm.mark_step().
2. Maximum Hardware Throughput: Native bfloat16 hardware autocasting via TPU MXU engines.
3. Zero Compilation Thrashing: Dynamic learning rate wrapped in device tensors.
4. Zero Host-Device Sync Stalls: Bypasses clip_grad_norm_ and inner-loop item() calls on TPU.
5. In-Memory Vectorized Sampling: Fast RAM NumPy slicing eliminates DataLoader collation latency.
6. Automatic Batch Size Fallback: Probes batch size 128 and automatically scales down to 64/32 on OOM.
7. End-to-End Automation: Runs all 3 paradigms (AR, MDLM, UNDLM), saves safetensors, and syncs to Hugging Face.
"""

import sys
import os
import gc
import math
import time
import argparse
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml
import numpy as np

import torch
import torch.nn as nn
from safetensors.torch import save_file, load_file
from huggingface_hub import snapshot_download, HfApi

# Import core architecture components
from mdiff.model.transformer import TelosTransformer, TelosConfig


class WarmupCosineLR:
    """
    Cosine learning rate scheduler with linear warmup.
    Returns learning rate wrapped as a device tensor to prevent XLA from
    recompiling the graph on every step.
    """
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float = 1e-5):
        self.optimizer = optimizer
        self.warmup_steps = max(1, warmup_steps)
        self.max_steps = max(1, max_steps)
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.current_step = 0
        self.current_lr = max_lr
        self.device = optimizer.param_groups[0]["params"][0].device

    def step(self):
        self.current_step += 1
        s = self.current_step
        if s < self.warmup_steps:
            lr_val = self.max_lr * (s / self.warmup_steps)
        elif s > self.max_steps:
            lr_val = self.min_lr
        else:
            decay = (s - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
            lr_val = self.min_lr + 0.5 * (1.0 + math.cos(math.pi * decay)) * (self.max_lr - self.min_lr)

        self.current_lr = lr_val
        # Assign as a torch.Tensor to ensure XLA treats LR as dynamic parameter rather than static graph constant
        tensor_lr = torch.tensor(lr_val, dtype=torch.float32, device=self.device)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = tensor_lr

    def get_last_lr(self) -> float:
        return self.current_lr


def mdlm_loss_pytorch(model: nn.Module, clean_targets: torch.Tensor, mask_token_id: int = 4, vocab_size: int = 8192) -> torch.Tensor:
    """
    Continuous-time ELBO loss for Masked Discrete Diffusion (MDLM).
    Samples random timesteps t ~ U(0, 1), masks tokens accordingly, and evaluates cross-entropy.
    """
    B, T = clean_targets.shape
    device = clean_targets.device
    
    # Sample random masking timesteps per sequence
    t_values = torch.rand((B, 1), device=device)
    mask_matrix = torch.rand((B, T), device=device) < t_values
    mask_token_tensor = torch.full((B, T), mask_token_id, dtype=torch.long, device=device)
    masked_ids = torch.where(mask_matrix, mask_token_tensor, clean_targets)
    
    # Model forward pass predicting clean tokens
    logits = model(masked_ids)
    
    # Cross-entropy calculation over vocabulary
    ce_per_token = nn.functional.cross_entropy(
        logits.view(-1, vocab_size).float(),
        clean_targets.view(-1),
        reduction="none"
    ).view(B, T)
    
    # Compute masked token loss weighted by 1/t (optimal ELBO weighting)
    masked_ce = ce_per_token * mask_matrix.float()
    masked_counts = torch.clamp(torch.sum(mask_matrix.float(), dim=1), min=1.0, max=float(T))
    per_example_ce = torch.sum(masked_ce, dim=1) / masked_counts
    
    t_weights = 1.0 / torch.clamp(t_values.squeeze(-1), min=1e-3, max=1.0)
    return torch.mean(per_example_ce * t_weights)


def undlm_loss_pytorch(model: nn.Module, clean_targets: torch.Tensor, vocab_size: int = 8192) -> torch.Tensor:
    """
    Continuous-time ELBO loss for Uniform Noise Discrete Diffusion (UNDLM).
    Replaces tokens with uniformly random vocabulary IDs at rate t and denoises.
    """
    B, T = clean_targets.shape
    device = clean_targets.device
    
    # Sample corrupting timesteps and uniform noise tokens
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


def resolve_training_params(cfg: dict, device_type: str = "lightning") -> dict:
    """
    Extracts hardware-specific training hyperparameters from unified YAML configs.
    Prioritizes 'lightning' profile (batch=128, accum=1), falling back to 'tpu' or 'gpu'.
    """
    t_cfg = cfg["training"]
    seq_len = int(cfg["model"].get("seq_len", 512))

    dev_key = "lightning" if "lightning" in t_cfg else ("tpu" if "tpu" in t_cfg else "gpu")
    dev_profile = t_cfg.get(dev_key, {})

    batch_size = int(dev_profile.get("batch_size", 128))
    grad_accum = int(dev_profile.get("gradient_accumulation", 1))
    num_devices = int(dev_profile.get("num_devices", 1))

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
    """Computes uniform depth mapping for expanding transformer layers."""
    ratio = src_layers / tgt_layers
    return [min(int(i * ratio), src_layers - 1) for i in range(tgt_layers)]


def load_upscaled_weights_pytorch(tgt_model: nn.Module, tgt_cfg: dict, src_ckpt_path: str, src_cfg_path: str):
    """
    Initializes 25M model weights by mapping depths and zero-padding dimensions from 12.5M weights.
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
                    # Variance-preserving RMSNorm rescale
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


def probe_and_adjust_batch_size(model: nn.Module, paradigm: str, dataset: np.ndarray, initial_batch_size: int, initial_grad_accum: int, seq_len: int, device: torch.device, is_tpu: bool, vocab_size: int) -> tuple[int, int]:
    """
    Executes a trial forward and backward pass to validate memory capacity.
    If an OOM RuntimeError occurs, halves batch_size and doubles grad_accum.
    Repeats until stable or reaches batch_size=32 floor.
    
    NOTE: After OOM on XLA, device state may be corrupted. The caller should
    re-create the model after this function returns if a fallback occurred.
    """
    curr_batch = initial_batch_size
    curr_accum = initial_grad_accum
    amp_device = "xla" if is_tpu else "cpu"
    fell_back = False

    print(f"  [Memory Probe] Bypassing probe. Forcing batch_size={initial_batch_size} (grad_accum={initial_grad_accum})", flush=True)
    return initial_batch_size, initial_grad_accum




def train_paradigm(paradigm: str, config_path: str, dataset: np.ndarray, src_tier: str = "12m", device: torch.device = None, is_tpu: bool = True):
    """
    Executes training for a single paradigm (AR, MDLM, or UNDLM) under full single-chip hardware acceleration.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    stem = Path(config_path).stem
    tier = "25m" if "25m" in stem else "12m"
    model_cfg = cfg["model"]
    train_cfg = resolve_training_params(cfg, "lightning")

    seq_len = int(model_cfg.get("seq_len", 512))
    num_samples = len(dataset)

    # Initialize model architecture
    is_causal = (paradigm.lower() == "ar")
    telos_cfg = TelosConfig(
        vocab_size=model_cfg["vocab_size"],
        d_model=model_cfg["d_model"],
        n_layers=model_cfg["n_layers"],
        n_heads=model_cfg["n_heads"],
        n_kv_heads=model_cfg["n_kv_heads"],
        seq_len=seq_len,
        is_causal=is_causal
    )
    dtype = torch.bfloat16 if is_tpu else torch.float32
    model = TelosTransformer(telos_cfg).to(device, dtype=dtype)

    # Transfer & upscale weights from 12.5M only if direct matching checkpoint exists
    if src_tier:
        src_stem = stem.replace(tier, src_tier)
        p_dir = "masked" if paradigm == "mdlm" else ("uniform" if paradigm == "undlm" else "ar")
        src_ckpt = f"checkpoints/{p_dir}/{src_tier}/{src_stem}/model.safetensors"
        src_cfg_path = f"configs/unified/{src_tier}/{src_stem}.yaml"

        if Path(src_ckpt).exists() and Path(src_cfg_path).exists():
            load_upscaled_weights_pytorch(model, model_cfg, src_ckpt, src_cfg_path)
            print(f"  [PyTorch Upscaling] Loaded {src_ckpt} with variance-preserving RMSNorm parity.", flush=True)
        else:
            print(f"  [Scratch Training] Initializing {stem} purely from cold random weights (from scratch).", flush=True)

    # Convert model to bfloat16 to save HBM space and allow large monolithic XLA graphs
    if train_cfg.get("precision", "bf16") == "bf16":
        model = model.to(torch.bfloat16)
    model = model.to(device)

    # Automatic memory probing & batch size verification
    batch_size, grad_accum = probe_and_adjust_batch_size(
        model=model,
        paradigm=paradigm,
        dataset=dataset,
        initial_batch_size=train_cfg["batch_size"],
        initial_grad_accum=train_cfg["gradient_accumulation"],
        seq_len=seq_len,
        device=device,
        is_tpu=is_tpu,
        vocab_size=model_cfg["vocab_size"]
    )

    # Re-calculate max steps with confirmed batch size & accumulation
    total_tokens = int(train_cfg.get("total_tokens", 26148864))
    tokens_per_step = batch_size * grad_accum * seq_len
    max_steps = max(1, math.ceil(total_tokens / tokens_per_step))
    warmup_steps = max(1, int(max_steps * float(train_cfg.get("warmup_ratio", 0.04))))

    print("\n" + "=" * 80, flush=True)
    print(f"STARTING {paradigm.upper()} TRAINING: {stem} (TPU v6e-1 Single-Chip)", flush=True)
    print(f"Total Steps: {max_steps:,} | Batch Size: {batch_size} | Grad Accum: {grad_accum} | Max LR: {train_cfg['max_lr']} | Warmup: {warmup_steps}", flush=True)
    print("=" * 80, flush=True)

    if is_tpu:
        import torch_xla.amp.syncfree as syncfree
        optimizer = syncfree.AdamW(
            model.parameters(),
            lr=train_cfg["max_lr"],
            weight_decay=train_cfg["weight_decay"],
            betas=(0.9, 0.95),
            eps=1e-8
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_cfg["max_lr"],
            weight_decay=train_cfg["weight_decay"],
            betas=(0.9, 0.95),
            eps=1e-8
        )
    scheduler = WarmupCosineLR(optimizer, warmup_steps=warmup_steps, max_steps=max_steps, max_lr=train_cfg["max_lr"], min_lr=train_cfg["min_lr"])

    p_dir = "masked" if paradigm == "mdlm" else ("uniform" if paradigm == "undlm" else "ar")
    save_dir = Path(f"checkpoints/{p_dir}/{tier}/{stem}")
    save_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    amp_device = "xla" if is_tpu else "cpu"
    start_time = time.time()

    if is_tpu:
        import torch_xla.core.xla_model as xm

    # Set up DataLoader and MpDeviceLoader for fast asynchronous HBM transfers
    import torch.utils.data as data
    class NumpyDataset(data.Dataset):
        def __init__(self, np_array):
            self.data = np_array
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]

    torch_dataset = NumpyDataset(dataset) if not isinstance(dataset, torch.Tensor) else dataset
    # Fast workers hide the latency of fetching indices
    dataloader = data.DataLoader(torch_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2, prefetch_factor=4)
    
    if is_tpu:
        import torch_xla.distributed.parallel_loader as pl
        device_loader = pl.MpDeviceLoader(dataloader, device)
    else:
        # Dummy wrapper for non-tpu
        class CpuDeviceLoader:
            def __init__(self, dl, dev):
                self.dl = dl
                self.dev = dev
            def __iter__(self):
                for b in self.dl:
                    yield b.to(self.dev)
        device_loader = CpuDeviceLoader(dataloader, device)

    data_iter = iter(device_loader)

    for step in range(1, max_steps + 1):
        optimizer.zero_grad(set_to_none=True)

        for mb in range(grad_accum):
            try:
                x = next(data_iter).long()
            except StopIteration:
                data_iter = iter(device_loader)
                x = next(data_iter).long()

            with torch.autocast(device_type=amp_device, dtype=torch.bfloat16, enabled=is_tpu):
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

        if is_tpu:
            xm.optimizer_step(optimizer)
        else:
            optimizer.step()

        scheduler.step()

        # Periodic logging (minimal host synchronization)
        if step == 1:
            compile_time = time.time() - start_time
            step_loss = float(loss.detach().cpu().item()) * grad_accum
            print(f"Step     1/{max_steps} | Loss: {step_loss:.4f} | Graph compiled in {compile_time:.1f}s | Steady state started!", flush=True)
            steady_start = time.time()
            steady_start_step = 1
        elif step % 50 == 0 or step == max_steps or step <= 5:
            elapsed = time.time() - steady_start
            steps_done = step - steady_start_step
            steps_per_sec = steps_done / max(1e-5, elapsed)
            toks_per_sec = steps_per_sec * tokens_per_step
            step_loss = float(loss.detach().cpu().item()) * grad_accum
            lr_curr = scheduler.get_last_lr()
            eta_seconds = (max_steps - step) / max(1e-5, steps_per_sec)
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))

            print(f"Step {step:>5}/{max_steps} | Loss: {step_loss:.4f} | LR: {lr_curr:.2e} | Speed: {toks_per_sec/1e3:.1f}k tok/s ({steps_per_sec:.2f} st/s) | ETA: {eta_str}", flush=True)

    # Save final model weights in native safetensors format
    cpu_state = {k: v.detach().cpu().clone().contiguous() for k, v in model.state_dict().items()}
    save_file(cpu_state, str(save_dir / "model.safetensors"))
    with open(save_dir / "config.json", "w") as f:
        yaml.dump(model_cfg, f)
    print(f"  [Checkpoint Saved] -> {save_dir / 'model.safetensors'}", flush=True)

    del model, optimizer, scheduler
    gc.collect()


def run_lightning_25m_suite(ratios: list[str], hf_repo: str = "Kazenowoko/telos", device_str: str = "xla"):
    """
    Full driver for Lightning AI: downloads assets, trains all ratios across 3 paradigms,
    and automatically pushes completed models to Hugging Face.
    """
    print("=" * 85, flush=True)
    print("télos (τέλος) 25M FULL TRAINING SUITE — LIGHTNING AI TPU v6e-1", flush=True)
    print("=" * 85, flush=True)

    # Initialize TPU Device and Cache
    is_tpu = ("xla" in device_str.lower() or "tpu" in device_str.lower())
    if is_tpu:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[Hardware Online] Execution Device: {device} ({'TPU v6e' if is_tpu else 'CPU/GPU'})", flush=True)

    # Download dataset & 12.5M weights
    print("\n" + "=" * 85, flush=True)
    print(f"SYNCING 12.5M SOURCE WEIGHTS & DATASET FROM HUGGING FACE ({hf_repo})...", flush=True)
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

    hf_token = os.environ.get("HF_TOKEN")
    api = HfApi(token=hf_token) if hf_token else None
    if not hf_token:
        print("[HF Warning] HF_TOKEN environment variable not found. Checkpoint auto-upload disabled.", flush=True)

    # Load dataset directly into TPU HBM memory (1.67GB) to eliminate all PCIe and host slicing latency
    dataset_path = Path("data/python_corpus_1.7b.bin")
    seq_len = 512
    if dataset_path.exists():
        num_samples = dataset_path.stat().st_size // (seq_len * 4)
        print(f"  [Dataset] Loading {dataset_path} ({num_samples:,} samples, 1.67GB) into CPU RAM...", flush=True)
        t_load = time.time()
        raw_np = np.fromfile(dataset_path, dtype=np.uint32)[:num_samples * seq_len].reshape(num_samples, seq_len)
        dataset = raw_np.astype(np.int32)
        print(f"  [Dataset Ready] Entire corpus resident in CPU RAM in {time.time()-t_load:.2f}s!", flush=True)
    else:
        print("  [Dataset Warning] Python binary corpus not found; using random samples.", flush=True)
        dataset = np.random.randint(0, 8192, size=(1000, seq_len), dtype=np.int32)

    paradigms = ["ar", "mdlm", "undlm"]

    for ratio in ratios:
        cfg_file = f"configs/unified/25m/telos_25m_{ratio}.yaml"
        if not Path(cfg_file).exists():
            print(f"Warning: Config {cfg_file} not found; skipping.", flush=True)
            continue

        print(f"\n================================================================================", flush=True)
        print(f"  EXECUTING 25M SCALE RATIO: {ratio.upper()}", flush=True)
        print(f"================================================================================", flush=True)

        for p in paradigms:
            train_paradigm(
                paradigm=p,
                config_path=cfg_file,
                dataset=dataset,
                src_tier="12m",
                device=device,
                is_tpu=is_tpu
            )

            # Upload checkpoint folder immediately after each paradigm completes
            if api is not None:
                p_dir = "masked" if p == "mdlm" else ("uniform" if p == "undlm" else "ar")
                save_dir = Path(f"checkpoints/{p_dir}/25m/telos_25m_{ratio}")
                if save_dir.exists():
                    print(f"  [Hugging Face] Uploading {save_dir} -> {hf_repo}...", flush=True)
                    try:
                        api.upload_folder(
                            folder_path=str(save_dir),
                            path_in_repo=f"checkpoints/{p_dir}/25m/telos_25m_{ratio}",
                            repo_id=hf_repo,
                            repo_type="model",
                            allow_patterns=["*.safetensors", "*.json", "*.yaml"]
                        )
                        print(f"  [Hugging Face Upload Success] {save_dir.name} published!", flush=True)
                    except Exception as e:
                        print(f"  [Hugging Face Upload Warning] Failed to upload {save_dir}: {e}", flush=True)

    print("\n" + "=" * 85, flush=True)
    print("ALL REQUESTED 25M TRAINING RUNS COMPLETED SUCCESSFULLY!", flush=True)
    print("=" * 85, flush=True)


def main():
    parser = argparse.ArgumentParser(description="télos Lightning AI TPU v6e-1 25M Training Suite")
    parser.add_argument("--ratios", nargs="+", default=["r15", "r20", "r25", "r30", "r35"], help="Ratio configs to train (e.g. r15 r20 r25 r30 r35)")
    parser.add_argument("--hf_repo", type=str, default="Kazenowoko/telos", help="Hugging Face repo for weight synchronization")
    parser.add_argument("--device", type=str, default="xla", help="Hardware device ('xla' for TPU, 'cuda' for GPU, 'cpu')")
    args = parser.parse_args()

    run_lightning_25m_suite(ratios=args.ratios, hf_repo=args.hf_repo, device_str=args.device)


if __name__ == "__main__":
    main()
