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

import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from telos.model.transformer import TelosTransformer, TelosConfig
from telos.diffusion.loss import mdlm_loss
from telos.diffusion.sampler import MDLMSampler
from telos.training.lr_schedule import WarmupCosineLR


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
        """saves complete checkpoint and standalone weights-only file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_state = self.model.state_dict()

        checkpoint = {
            "global_step": self.global_step,
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "config": self.config,
        }
        torch.save(checkpoint, path)

        # Save weights-only file alongside full checkpoint
        weights_path = path.parent / f"weights_{path.name}"
        torch.save(model_state, weights_path)

        # Ratio milestone explicit tagging
        ratio_tags = {
            162: "ratio_1_1_step_162.pt",
            486: "ratio_1_3_step_486.pt",
            811: "ratio_1_5_step_811.pt",
            1621: "ratio_1_10_step_1621.pt",
            2741: "ratio_1_17_step_2741.pt",
        }
        if self.global_step in ratio_tags:
            milestone_path = path.parent / ratio_tags[self.global_step]
            torch.save({"config": self.config, "model_state_dict": model_state}, milestone_path)
            print(f"⭐ Milestone Checkpoint saved -> {milestone_path}")

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

            # checkpoint by step count or time interval
            current_time = time.time()
            time_since_last_save = (current_time - self.last_saved_time) / 60.0

            if (self.global_step % self.save_every_steps == 0) or (time_since_last_save >= self.save_every_minutes):
                ckpt_path = self.checkpoint_dir / f"checkpoint_step_{self.global_step}.pt"
                self.save_checkpoint(ckpt_path)
                self.last_saved_time = current_time

        # save final checkpoint
        final_path = self.checkpoint_dir / "checkpoint_final.pt"
        self.save_checkpoint(final_path)
        print("Training complete!")
