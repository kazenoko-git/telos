"""
PyTorch trainer for télos MDLM.

includes:
- AdamW optimizer (weight decay = .1, beta1 = .9, beta2 = .95)
- mixed precision training via torch.amp (bf16 on MPS/CUDA)
- gradient clipping (max_norm=1.0)
- periodic checkpointing (by minutes or by step count)

training configuration:
- max_steps: 5000
- max_lr: 3e-4
- min_lr: 3e-5
- warmup_steps: 100
- weight_decay: 0.1
- grad_clip: 1.0
- precision: bf16 (fp16 on CPU)
- checkpoint_dir: checkpoints
- save_every_steps: 500
- save_every_minutes: 10
"""

import numpy as np
import time
import math
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mdiff.model.transformer import TelosTransformer, TelosConfig
from mdiff.diffusion.loss import mdlm_loss
from mdiff.diffusion.sampler import MDLMSampler
from mdiff.training.lr_schedule import WarmupCosineLR


class TelosTrainer:
    """trainer orchestrator for MDLM model training and checkpointing."""

    def __init__(
        self,
        model: TelosTransformer,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        config: dict | None = None,
        device: str | torch.device = "cpu"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or {}
        # Resolve device (including TPU / PyTorch XLA)
        if str(device).lower() in ["tpu", "xla"]:
            try:
                # pyrefly: ignore
                import torch_xla.core.xla_model as xm
                self.device = xm.xla_device()
                self.is_tpu = True
            except ImportError:
                print("Warning: torch_xla not installed. Falling back to CPU.")
                self.device = torch.device("cpu")
                self.is_tpu = False
        else:
            self.device = torch.device(device)
            self.is_tpu = False

        self.model.to(self.device)

        # Training hyperparameters
        train_cfg = self.config.get("training", {})
        self.max_steps = int(train_cfg.get("max_steps", 5000))
        self.max_lr = float(train_cfg.get("max_lr", 3e-4))
        self.min_lr = float(train_cfg.get("min_lr", 3e-5))
        self.warmup_steps = int(train_cfg.get("warmup_steps", 100))
        self.weight_decay = float(train_cfg.get("weight_decay", 0.1))
        self.grad_clip = float(train_cfg.get("grad_clip", 1.0))
        self.precision = train_cfg.get("precision", "bf16")

        # checkpoint parameters
        ckpt_cfg = self.config.get("checkpoint", {})
        self.save_every_steps = ckpt_cfg.get("save_every_steps", 500)
        self.save_every_minutes = ckpt_cfg.get("save_every_minutes", 10)
        self.checkpoint_dir = Path(ckpt_cfg.get("dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # setup AdamW optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.max_lr,
            betas=(0.9, 0.95),
            weight_decay=self.weight_decay,
            eps=1e-8
        )

        # setup LR scheduler
        self.scheduler = WarmupCosineLR(
            self.optimizer,
            warmup_steps=self.warmup_steps,
            max_steps=self.max_steps,
            min_lr=self.min_lr
        )

        # mixed precision GradScaler if using fp16/bf16 on CUDA/MPS/XLA
        use_amp = (self.precision in ["fp16", "bf16"]) and (self.device.type in ["cuda", "mps", "xla"])
        self.amp_dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
        self.use_amp = use_amp

        self.global_step = 0
        self.last_saved_time = time.time()

    def save_checkpoint(self, path: str | Path):
        """saves complete checkpoint including model, optimizer, scheduler, and RNG states."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "config": self.config,
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path} (Step {self.global_step})")

    def load_checkpoint(self, path: str | Path):
        """loads checkpoint and restores all states to continue training smoothly."""
        path = Path(path)
        assert path.exists(), f"Checkpoint not found at {path}"

        checkpoint = torch.load(path, map_location="cpu")
        self.global_step = checkpoint["global_step"]
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        
        if checkpoint.get("cuda_rng_state") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])

        print(f"Checkpoint restored from {path} at Step {self.global_step}")

    def train(self):
        """executes full training loop."""
        self.model.train()
        start_time = time.time()

        if self.is_tpu:
            # pyrefly: ignore
            import torch_xla.distributed.parallel_loader as pl
            para_loader = pl.ParallelLoader(self.train_loader, [self.device])
            train_iterator = iter(para_loader.per_device_loader(self.device))
        else:
            train_iterator = iter(self.train_loader)

        print(f"Starting training: total_steps={self.max_steps}, device={self.device}, amp={self.use_amp}")

        grad_accum = self.config.get("training", {}).get("gradient_accumulation", 1)
        self.optimizer.zero_grad()

        while self.global_step < self.max_steps:
            # metrics accumulate as on-device tensors — NO .item() in the hot loop
            last_metrics = None

            for micro_step in range(grad_accum):
                try:
                    masked_input_ids, targets, mask_positions, t_values = next(train_iterator)
                except StopIteration:
                    train_iterator = iter(self.train_loader)
                    masked_input_ids, targets, mask_positions, t_values = next(train_iterator)

                masked_input_ids = masked_input_ids.to(self.device)
                targets = targets.to(self.device)
                mask_positions = mask_positions.to(self.device)
                t_values = t_values.to(self.device)

                if self.use_amp:
                    with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                        logits = self.model(masked_input_ids)
                        loss, metrics = mdlm_loss(logits, targets, mask_positions, t_values)
                else:
                    logits = self.model(masked_input_ids)
                    loss, metrics = mdlm_loss(logits, targets, mask_positions, t_values)

                loss = loss / grad_accum
                loss.backward()

                # keep last micro-step metrics (tensor, no sync)
                last_metrics = metrics

            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            if self.is_tpu:
                self.optimizer.step()
                # pyrefly: ignore
                import torch_xla
                torch_xla.sync()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad()
            self.scheduler.step()
            self.global_step += 1

            # logging step — ONLY sync device→host here via .item()
            if self.global_step % 50 == 0 or self.global_step == 1:
                lr = self.scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time
                print(f"Step {self.global_step}/{self.max_steps} | Loss: {last_metrics['loss'].item():.4f} | "
                      f"Unweighted CE: {last_metrics['unweighted_ce'].item():.4f} | LR: {lr:.2e} | Elapsed: {elapsed:.1f}s", flush=True)

            # Checkpoint by step count, time interval, or explicit ratio milestones
            current_time = time.time()
            time_since_last_save = (current_time - self.last_saved_time) / 60.0

            ratio_milestones = {
                162: "checkpoint_ratio_1_1_step_162.pt",
                486: "checkpoint_ratio_1_3_step_486.pt",
                811: "checkpoint_ratio_1_5_step_811.pt",
                1621: "checkpoint_ratio_1_10_step_1621.pt",
                2741: "checkpoint_ratio_1_17_step_2741.pt",
            }

            if self.global_step in ratio_milestones:
                milestone_path = self.checkpoint_dir / ratio_milestones[self.global_step]
                self.save_checkpoint(milestone_path)

            if (self.global_step % self.save_every_steps == 0) or (time_since_last_save >= self.save_every_minutes):
                ckpt_path = self.checkpoint_dir / f"checkpoint_step_{self.global_step}.pt"
                self.save_checkpoint(ckpt_path)
                self.last_saved_time = current_time

        # save final checkpoint
        final_path = self.checkpoint_dir / "checkpoint_final.pt"
        self.save_checkpoint(final_path)
        print("Training complete!")


# =========================================================================
# MLX NATIVE TRAINER
# =========================================================================

try:
    import mlx.core as mx
    import mlx.nn as mx_nn
    import mlx.optimizers as mx_optim
    from mlx.utils import tree_map
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


def get_sys_mem_str() -> str:
    try:
        active_gb = mx.get_active_memory() / 1e9
        peak_gb = mx.get_peak_memory() / 1e9
        import subprocess
        swap_res = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True)
        swap_parts = swap_res.stdout.strip().split()
        used_swap = swap_parts[6] if len(swap_parts) >= 7 else "0M"
        return f"Metal Unified GPU: {active_gb:.2f}GB (Peak: {peak_gb:.2f}GB) | Swap: {used_swap}"
    except Exception:
        return ""

def sample_cosine_timesteps_mlx(batch_size: int, eps: float = 1e-5):
    """Cosine-transformed timestep sampler (equivalent to Beta(0.5, 0.5) / arcsine distribution).
    Oversamples endpoints t near 0 and 1. Legacy sampler — kept for reproducibility."""
    u = mx.random.uniform(0.0, 1.0, (batch_size, 1))
    t = 0.5 - 0.5 * mx.cos(math.pi * u)
    return mx.clip(t, eps, 1.0)

def sample_uniform_timesteps_mlx(batch_size: int, eps: float = 1e-5):
    """Uniform timestep sampler for standard MDLM baseline training."""
    u = mx.random.uniform(0.0, 1.0, (batch_size, 1))
    return mx.clip(u, eps, 1.0)

def sample_beta_timesteps_mlx(batch_size: int, alpha: float = 1.5, beta: float = 1.5, eps: float = 1e-5):
    """Beta(α, β) timestep sampler matching MDLM paper specification.
    Default Beta(1.5, 1.5) concentrates mass toward moderate masking rates (t ≈ 0.5).
    Uses numpy for sampling (negligible cost vs forward/backward pass)."""
    t = np.random.beta(alpha, beta, size=(batch_size, 1)).astype(np.float32)
    return mx.clip(mx.array(t), eps, 1.0)

def build_special_token_lut(vocab_size: int, special_tokens=(0, 1, 2, 3)):
    """Precomputes 1D boolean array for constant-time special token lookup."""
    lut = [False] * vocab_size
    for token_id in special_tokens:
        if token_id < vocab_size:
            lut[token_id] = True
    return mx.array(lut, dtype=mx.bool_)

def apply_masking_mlx(input_ids, mask_token_id=1, special_token_lut=None, strategy="beta"):
    B, T = input_ids.shape
    if strategy == "beta":
        t_values = sample_beta_timesteps_mlx(B)
    elif strategy == "cosine":
        t_values = sample_cosine_timesteps_mlx(B)
    else:
        t_values = sample_uniform_timesteps_mlx(B)

    rand_matrix = mx.random.uniform(0.0, 1.0, (B, T))
    raw_mask = rand_matrix < t_values

    if special_token_lut is not None:
        is_special = special_token_lut[input_ids]
    else:
        is_special = (input_ids == 0) | (input_ids == 1) | (input_ids == 2) | (input_ids == 3)

    mask_positions = raw_mask & (~is_special)
    masked_input_ids = mx.where(mask_positions, mask_token_id, input_ids)
    return masked_input_ids, mask_positions, t_values

def loss_fn_mlx(model, masked_input_ids, targets, mask_positions, t_values, vocab_size):
    logits = model(masked_input_ids)
    B, T, V = logits.shape
    logits_flat = logits.reshape(-1, V)
    targets_flat = targets.reshape(-1)

    ce_per_token = mx_nn.losses.cross_entropy(logits_flat, targets_flat, reduction="none").reshape(B, T)
    masked_ce = ce_per_token * mask_positions.astype(mx.float32)

    masked_count = mx.clip(mx.sum(mask_positions.astype(mx.float32), axis=1), 1.0, float(T))
    per_example_ce = mx.sum(masked_ce, axis=1) / masked_count
    unweighted_ce = mx.mean(per_example_ce)

    t_weights = 1.0 / mx.clip(mx.squeeze(t_values, -1), 1e-3, 1.0)
    reweighted_loss = mx.mean(per_example_ce * t_weights)
    return reweighted_loss, unweighted_ce

def get_global_targets_contiguous(dataset_matrix, idx_ptr, total_batch, seq_len):
    n_rows = dataset_matrix.shape[0]
    end_ptr = idx_ptr + total_batch
    if end_ptr <= n_rows:
        batch = dataset_matrix[idx_ptr:end_ptr, :seq_len]
        next_ptr = end_ptr % n_rows
    else:
        first = dataset_matrix[idx_ptr:n_rows, :seq_len]
        remainder = end_ptr - n_rows
        second = dataset_matrix[:remainder, :seq_len]
        batch = np.concatenate((first, second), axis=0)
        next_ptr = remainder
    return mx.array(batch, dtype=mx.int32), next_ptr

def cast_optimizer_moments_bf16(state_dict: dict) -> dict:
    """Casts AdamW moment tensors m and v to bfloat16 to reduce memory footprint by 50%."""
    new_state = {}
    for k, v in state_dict.items():
        if isinstance(v, dict):
            new_state[k] = cast_optimizer_moments_bf16(v)
        elif isinstance(v, mx.array) and k in ("m", "v") and v.dtype == mx.float32:
            new_state[k] = v.astype(mx.bfloat16)
        else:
            new_state[k] = v
    return new_state

class TelosMLXTrainer:
    def __init__(self, model, cfg):
        if not MLX_AVAILABLE:
            raise ImportError("MLX is not installed. Cannot use TelosMLXTrainer.")
        self.model = model
        self.cfg = cfg
        self.m_cfg = cfg["model"]
        self.t_cfg = cfg["training"]
        self.c_cfg = cfg.get("checkpoint", {})
        self.special_lut = build_special_token_lut(self.m_cfg["vocab_size"])
        
        # Enable gradient checkpointing on model if requested in config
        if self.t_cfg.get("gradient_checkpointing", False) or self.m_cfg.get("use_grad_checkpoint", False):
            self.model.use_grad_checkpoint = True
            print("  [Memory] Gradient Checkpointing Enabled.")

    def train(self, resume_step: int = 0):
        import numpy as np
        
        train_bin = Path("data/python_corpus_mac.bin")
        if train_bin.exists():
            print(f"  Loading pre-tokenized dataset from {train_bin}...")
            raw_data = np.memmap(train_bin, dtype=np.int32, mode="r")
            n_seqs = len(raw_data) // self.m_cfg["seq_len"]
            dataset_matrix = raw_data[:n_seqs * self.m_cfg["seq_len"]].reshape(n_seqs, self.m_cfg["seq_len"])
        else:
            print("  Notice: Pre-tokenized dataset file not found. Generating synthetic stream for throughput run...")
            dataset_matrix = np.random.randint(0, self.m_cfg["vocab_size"], (10000, self.m_cfg["seq_len"]), dtype=np.uint16)

        max_steps = int(self.t_cfg["max_steps"])
        warmup_steps = int(self.t_cfg["warmup_steps"])
        max_lr = float(self.t_cfg["max_lr"])
        min_lr = float(self.t_cfg["min_lr"])
        weight_decay = float(self.t_cfg.get("weight_decay", 0.1))

        bs = self.t_cfg["batch_size"]
        grad_accum = self.t_cfg["gradient_accumulation"]
        idx_ptr = 0
        
        if resume_step > 0:
            print(f"  Resuming from step {resume_step}. Fast-forwarding dataset...")
            # Each step consumes bs * grad_accum sequences
            seqs_consumed = resume_step * (bs * grad_accum)
            idx_ptr = seqs_consumed % len(dataset_matrix)

        def get_lr(step):
            if step < warmup_steps:
                return max_lr * (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
            return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

        optimizer = mx_optim.AdamW(learning_rate=max_lr, weight_decay=weight_decay)

        loss_and_grad_fn = mx_nn.value_and_grad(self.model, loss_fn_mlx)
        special_lut = self.special_lut
        
        def microbatch_step_uncompiled(batch_seqs):
            masked_ids, mask_pos, t_vals = apply_masking_mlx(batch_seqs, mask_token_id=1, special_token_lut=special_lut)
            (loss, ce), grads = loss_and_grad_fn(self.model, masked_ids, batch_seqs, mask_pos, t_vals, self.m_cfg["vocab_size"])
            return loss, ce, grads
            
        # Run graph trace once WITHOUT calling optimizer.update() to avoid synthetic AdamW moment pollution!
        dummy_seqs = mx.random.randint(0, self.m_cfg["vocab_size"], (self.t_cfg["batch_size"], self.m_cfg["seq_len"]))
        dummy_loss, dummy_ce, dummy_grads = microbatch_step_uncompiled(dummy_seqs)
        mx.eval(dummy_loss, dummy_ce, dummy_grads)
        del dummy_loss, dummy_ce, dummy_grads

        state = [self.model.state]
        microbatch_step = mx.compile(microbatch_step_uncompiled, inputs=state, outputs=state)

        base_dir_str = self.c_cfg.get("checkpoint_dir", self.c_cfg.get("dir", "checkpoints/uniform/25m/kappa_25m_1to35_mlx"))
        ckpt_dir = Path(base_dir_str)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Checkpoint Directory: {ckpt_dir} (Canonical)")

        start_time = time.time()

        for step in range(resume_step + 1, max_steps + 1):
            lr = get_lr(step)
            optimizer.learning_rate = lr

            global_targets, idx_ptr = get_global_targets_contiguous(dataset_matrix, idx_ptr, bs * grad_accum, self.m_cfg["seq_len"])
            
            accum_grads = None
            accum_loss = mx.array(0.0, dtype=mx.float32)
            accum_ce = mx.array(0.0, dtype=mx.float32)

            for i in range(grad_accum):
                batch_seqs = global_targets[i * bs : (i + 1) * bs]
                loss, ce, grads = microbatch_step(batch_seqs)
                
                if accum_grads is None:
                    accum_grads = grads
                else:
                    accum_grads = tree_map(lambda a, b: a + b, accum_grads, grads)
                
                accum_loss = accum_loss + loss
                accum_ce = accum_ce + ce

                # Evict microbatch graph without device-to-host scalar synchronization
                mx.eval(accum_grads, accum_loss, accum_ce)

            accum_grads = tree_map(lambda g: g / grad_accum, accum_grads)
            optimizer.update(self.model, accum_grads)
            
            # Cast AdamW moments to bf16 on the first step to save 50% optimizer memory
            if step == resume_step + 1:
                optimizer.state = cast_optimizer_moments_bf16(optimizer.state)
                
            mx.eval(self.model.parameters(), optimizer.state)

            # Periodic memory cache defragmentation
            if step % 100 == 0:
                mx.clear_cache()
                import gc
                gc.collect()

            if step % 50 == 0 or step == 1 or step == max_steps:
                avg_loss_val = accum_loss.item() / grad_accum
                avg_ce_val = accum_ce.item() / grad_accum

                elapsed = time.time() - start_time
                steps_taken = step - resume_step
                sps = steps_taken / elapsed if elapsed > 0 else 0
                tps = sps * bs * grad_accum * self.m_cfg["seq_len"]
                
                eta_mins = (max_steps - step) / sps / 60.0 if sps > 0 else 0.0
                mem_str = get_sys_mem_str()

                log_msg = f"  Step {step:>6d}/{max_steps} | ELBO Loss: {avg_loss_val:>6.2f} | CE: {avg_ce_val:>5.3f} | LR: {lr:.2e} | {sps:>5.1f} st/s | {tps:>9,.0f} tok/s | {mem_str} | ETA: {eta_mins:>4.1f}m"
                print(log_msg, flush=True)
                try:
                    Path("logs").mkdir(exist_ok=True)
                    with open("logs/overnight_suite.log", "a") as f_log:
                        f_log.write(log_msg + "\n")
                except Exception:
                    pass

            if step % self.c_cfg.get("save_every_steps", 1000) == 0:
                ckpt_file = ckpt_dir / f"checkpoint_step_{step}.safetensors"
                self.model.save_weights(str(ckpt_file))
                print(f"  [Checkpoint] Saved weights to {ckpt_file}")

        total_time = time.time() - start_time
        final_weights = ckpt_dir / "model.safetensors"
        self.model.save_weights(str(final_weights))

        import json
        with open(ckpt_dir / "config.json", "w") as f:
            json.dump(self.m_cfg, f, indent=2)

        tok_source = Path("configs/tokenizer_mac.json")
        if tok_source.exists():
            import shutil
            shutil.copy(tok_source, ckpt_dir / "tokenizer.json")

        print("=" * 70)
        print(f"  Training Complete! Total time: {total_time/60.0:.2f} minutes.")
        print(f"  Saved standalone model artifact to {ckpt_dir}/")
        print("=" * 70)
