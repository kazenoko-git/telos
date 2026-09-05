"""
Unified PyTorch Trainer for all Telos paradigms (AR, MDLM, UNDLM, COROSred).
"""

import time
import math
import json
import numpy as np
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .lr_schedule import WarmupCosineLR
from .dataloader import get_global_targets_contiguous_pytorch

from telos.diffusion.ar import ar_loss_fn_pytorch
from telos.diffusion.mdlm import mdlm_loss_pytorch, apply_masking_pytorch, sample_beta_timesteps
from telos.diffusion.undlm import undlm_loss_pytorch, apply_uniform_noise_pytorch
from telos.diffusion.corosred import crsr_phase_a_loss_fn_pytorch, crsr_phase_b_loss_fn_pytorch


class UnifiedPyTorchTrainer:
    """Unified PyTorch Trainer orchestrator for all paradigms."""

    def __init__(self, paradigm: str, model, cfg: dict, device_type: str = "cpu"):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed. Cannot use UnifiedPyTorchTrainer.")

        if torch.cuda.is_available():
            # Enable TF32 for matrix multiplications on Ampere+ architectures
            torch.set_float32_matmul_precision("high")

        self.paradigm = paradigm.lower()
        self.model = model
        self.cfg = cfg
        self.m_cfg = cfg.setdefault("model", {})
        self.t_cfg = cfg.setdefault("training", {})
        self.c_cfg = cfg.setdefault("checkpoint", {})

        # Ensure paradigm and architectural metadata are mirrored into configuration
        if "paradigm" not in self.cfg:
            self.cfg["paradigm"] = self.paradigm
        if "paradigm" not in self.m_cfg:
            self.m_cfg["paradigm"] = self.paradigm
        if "is_causal" not in self.m_cfg:
            self.m_cfg["is_causal"] = getattr(self.model, "is_causal", self.paradigm in ("ar", "corosred"))
        if "use_reliability_head" not in self.m_cfg:
            self.m_cfg["use_reliability_head"] = getattr(self.model, "use_reliability_head", self.paradigm == "corosred")

        self.vocab_size = self.m_cfg.get("vocab_size", 8192)
        self.seq_len = self.m_cfg.get("seq_len", 512)
        self.precision = self.t_cfg.get("precision", "bf16")
        
        # Build special token LUT manually for PyTorch
        self.special_lut = torch.zeros(self.vocab_size, dtype=torch.bool)
        self.special_lut[:4] = True

        if str(device_type).lower() in ["tpu", "xla"]:
            try:
                import torch_xla.core.xla_model as xm
                self.device = xm.xla_device()
                self.is_tpu = True
                # Query world size: prefer modern torch_xla.runtime API (PyTorch-XLA 2.4+),
                # with fallback to xm.xrt_world_size() for legacy environments.
                try:
                    import torch_xla.runtime as xr
                    self.world_size = xr.world_size()
                except (ImportError, AttributeError):
                    self.world_size = xm.xrt_world_size()
                self.is_master = xm.is_master_ordinal()
                print(f"  [Hardware] Detected PyTorch-XLA TPU Topology ({self.world_size} Cores).")
            except ImportError:
                print("Warning: torch_xla not installed. Falling back to CPU.")
                self.device = torch.device("cpu")
                self.is_tpu = False
                self.world_size = 1
                self.is_master = True
        elif str(device_type).lower() == "cuda":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.is_tpu = False
            self.world_size = 1
            self.is_master = True
            self.n_gpus = torch.cuda.device_count()
            if self.n_gpus > 1:
                gpu_name = torch.cuda.get_device_name(0)
                print(f"  [Hardware] Multi-GPU Detected: {self.n_gpus}x {gpu_name}. Wrapping model in DataParallel.")
            # Check native BF16 support (e.g. Turing T4 does not support BF16, Ampere/Ada/Hopper do)
            if not torch.cuda.is_bf16_supported() and self.precision == "bf16":
                print("  [Notice] Hardware lacks native BF16 support. Automatically falling back to FP16 with GradScaler.")
                self.precision = "fp16"
        else:
            self.device = torch.device(device_type)
            self.is_tpu = False
            self.world_size = 1
            self.is_master = True

        self.model.to(self.device)
        self.special_lut = self.special_lut.to(self.device)

        # Multi-GPU wrapping
        if getattr(self, "n_gpus", 1) > 1 and not self.is_tpu:
            self.model = nn.DataParallel(self.model)

        if self.paradigm == "corosred":
            self.crsr_cfg = cfg.get("crsr", cfg.get("corosred", {}))
            self.phase = self.crsr_cfg.get("phase", "A").upper()

        self.max_steps = int(self.t_cfg.get("max_steps", 5000))
        self.max_lr = float(self.t_cfg.get("max_lr", 3e-4))
        self.min_lr = float(self.t_cfg.get("min_lr", 3e-5))
        self.warmup_steps = int(self.t_cfg.get("warmup_steps", 100))
        self.weight_decay = float(self.t_cfg.get("weight_decay", 0.1))
        self.grad_clip = float(self.t_cfg.get("grad_clip", 1.0))

        # Separate parameters into decayed and non-decayed groups:
        # Standard transformer optimization applies 0 weight decay to 1D parameters (biases, layer norms, RMS norms)
        # and embedding lookup tables to avoid regularizing scale/shift parameters.
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if param.ndim == 1 or "embed" in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

        # Fused AdamW merges kernel operations for faster gradient updates on CUDA
        use_fused = (self.device.type == "cuda") and hasattr(torch.optim.AdamW, "fused")
        opt_kwargs = {
            "lr": self.max_lr,
            "betas": (0.9, 0.95),
            "eps": 1e-8,
        }
        if use_fused:
            opt_kwargs["fused"] = True

        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": self.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            **opt_kwargs
        )

        self.scheduler = WarmupCosineLR(
            self.optimizer,
            warmup_steps=self.warmup_steps,
            max_steps=self.max_steps,
            min_lr=self.min_lr
        )

        # TPU uses native BF16 execution on XLA cores; exclude from torch.amp.autocast
        self.use_amp = (self.precision in ["fp16", "bf16"]) and (self.device.type in ["cuda", "mps"])
        self.amp_dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16

        # Modern GradScaler API for FP16 training on CUDA
        self.use_scaler = (self.precision == "fp16") and (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda") if self.use_scaler else None

        self.global_step = 0

    def save_checkpoint(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.cfg,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str | Path):
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device)
        # Unwrap DataParallel if state_dict keys match
        state_dict = checkpoint["model_state_dict"]
        if hasattr(self.model, "module"):
            self.model.module.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)

    def _execute_microbatch(self, batch_seqs):
        """Executes a single microbatch and returns loss and metrics."""
        loss = None
        metrics = None

        if self.paradigm == "ar":
            logits = self.model(batch_seqs, mask_override=True)
            loss, metrics = ar_loss_fn_pytorch(logits, batch_seqs, special_token_lut=self.special_lut)

        elif self.paradigm == "mdlm":
            bs = batch_seqs.shape[0]
            t_vals = torch.from_numpy(sample_beta_timesteps(bs)).to(self.device)
            masked_ids, mask_pos, t_vals_out = apply_masking_pytorch(batch_seqs, t_vals, mask_token_id=1, special_token_lut=self.special_lut)
            logits = self.model(masked_ids, mask_override=False)
            loss, metrics = mdlm_loss_pytorch(logits, batch_seqs, mask_pos, t_vals_out)

        elif self.paradigm == "undlm":
            bs = batch_seqs.shape[0]
            t_vals = torch.from_numpy(sample_beta_timesteps(bs)).to(self.device)
            noisy_ids, corrupt_mask, t_vals_out = apply_uniform_noise_pytorch(batch_seqs, t_vals, self.vocab_size, special_token_lut=self.special_lut)
            logits = self.model(noisy_ids, mask_override=False)
            loss, metrics = undlm_loss_pytorch(logits, batch_seqs, t_vals_out, special_token_lut=self.special_lut)

        elif self.paradigm == "corosred":
            if self.phase == "A":
                k_amb = self.crsr_cfg.get("k_amb", 5)
                loss, metrics = crsr_phase_a_loss_fn_pytorch(self.model, batch_seqs, self.vocab_size, special_token_lut=self.special_lut, k_amb=k_amb)
            else:
                mask_token_id = self.m_cfg.get("mask_token_id", 0)
                loss, metrics = crsr_phase_b_loss_fn_pytorch(self.model, batch_seqs, self.vocab_size, mask_token_id=mask_token_id)
        
        return loss, metrics

    def _print_benchmark_report(self, steps: int, elapsed: float, latencies: list[float], bs: int, grad_accum: int):
        """Prints a publication-style benchmark report table and saves results to JSON."""
        if not self.is_master:
            return

        tokens_processed = steps * bs * grad_accum * self.seq_len
        sps = steps / elapsed if elapsed > 0 else 0.0
        tps = tokens_processed / elapsed if elapsed > 0 else 0.0
        mean_lat = np.mean(latencies) if latencies else 0.0
        p50_lat = np.median(latencies) if latencies else 0.0
        p95_lat = np.percentile(latencies, 95) if latencies else 0.0

        device_desc = f"{self.device.type.upper()}"
        if getattr(self, "n_gpus", 1) > 1:
            device_desc += f" ({self.n_gpus} GPUs DataParallel)"
        elif self.is_tpu:
            device_desc += f" ({self.world_size} TPU Cores SPMD)"

        print("\n" + "=" * 76)
        print("  TELOS UNIFIED BENCHMARK REPORT (PyTorch)")
        print("=" * 76)
        print(f"  Paradigm:             {self.paradigm.upper()}")
        print(f"  Hardware Target:      {device_desc}")
        print(f"  Precision:            {self.precision.upper()} (AMP: {self.use_amp})")
        print(f"  Batch Config:         batch_size={bs}, grad_accum={grad_accum}, seq_len={self.seq_len}")
        print(f"  Total Effective Batch: {bs * grad_accum} sequences ({bs * grad_accum * self.seq_len:,} tokens/step)")
        print("-" * 76)
        print(f"  Benchmark Duration:   {elapsed:.2f} seconds (limit: 300.0s / 5.0m)")
        print(f"  Steps Completed:      {steps:,}")
        print(f"  Tokens Processed:     {tokens_processed:,}")
        print(f"  Throughput:           {sps:.2f} steps/s  |  {tps:,.1f} tokens/s")
        print(f"  Latency per Step:     Mean: {mean_lat:.1f} ms  |  p50: {p50_lat:.1f} ms  |  p95: {p95_lat:.1f} ms")
        print("=" * 76 + "\n")

        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        bench_payload = {
            "paradigm": self.paradigm,
            "backend": "pytorch",
            "device": str(self.device),
            "precision": self.precision,
            "batch_size": bs,
            "grad_accum": grad_accum,
            "seq_len": self.seq_len,
            "steps": steps,
            "elapsed_seconds": elapsed,
            "tokens_processed": tokens_processed,
            "steps_per_sec": sps,
            "tokens_per_sec": tps,
            "latency_ms": {
                "mean": float(mean_lat),
                "p50": float(p50_lat),
                "p95": float(p95_lat)
            }
        }
        report_file = log_dir / f"benchmark_{self.paradigm}_pytorch_{int(time.time())}.json"
        with open(report_file, "w") as f:
            json.dump(bench_payload, f, indent=2)
        print(f"  Saved benchmark metrics to {report_file}\n")

    def train(self, resume_step: int = 0, benchmark: bool = False, benchmark_duration: float = 300.0):
        self.model.train()
        d_cfg = self.cfg.get("data", {})
        train_path = d_cfg.get("train_path", d_cfg.get("dataset_path", d_cfg.get("path", None)))
        use_synthetic = d_cfg.get("synthetic", False)

        if train_path is not None and not Path(train_path).exists():
            raise FileNotFoundError(f"Specified training data binary not found: {train_path}")

        train_bin = Path(train_path) if train_path else Path("data/python_corpus.bin")
        if not train_bin.exists() and train_path is None:
            train_bin = Path("data/python_corpus_2.5b.bin")
        
        if train_bin.exists() and not use_synthetic:
            if self.is_master:
                print(f"  Loading pre-tokenized dataset from {train_bin}...")
            # Detect dtype from metadata sidecar if available
            dtype = None
            for meta_cand in [Path(str(train_bin) + ".json"), train_bin.with_suffix(".json")]:
                if meta_cand.exists():
                    try:
                        with open(meta_cand, "r") as mf:
                            m_info = json.load(mf)
                            dt_str = m_info.get("dtype", "uint16")
                            dtype = np.int32 if dt_str == "int32" else np.uint16
                            break
                    except Exception:
                        pass
            if dtype is None:
                dtype = np.int32 if "mac" in str(train_bin) or self.vocab_size > 65536 else np.uint16

            raw_data = np.memmap(train_bin, dtype=dtype, mode="r")
            n_seqs = len(raw_data) // self.seq_len
            dataset_matrix = raw_data[:n_seqs * self.seq_len].reshape(n_seqs, self.seq_len)
        elif use_synthetic:
            if self.is_master:
                print("  Notice: Using synthetic dataset stream...")
            dataset_matrix = np.random.randint(0, self.vocab_size, (10000, self.seq_len), dtype=np.uint16)
        else:
            raise FileNotFoundError(
                f"No training data found at '{train_bin}'. Run 'telos dataprep' to generate token data, "
                "or pass '--synthetic' to train on synthetic random tokens."
            )

        bs = int(self.t_cfg.get("batch_size", 16))
        grad_accum = int(self.t_cfg.get("gradient_accumulation", 1))

        # Benchmark duration strictly capped at 300.0 seconds (5 minutes)
        max_bench_duration = min(float(benchmark_duration), 300.0)

        idx_ptr = 0
        self.global_step = resume_step
        if resume_step > 0:
            seqs_consumed = resume_step * (bs * grad_accum)
            idx_ptr = seqs_consumed % len(dataset_matrix)
            # Advance scheduler
            for _ in range(resume_step):
                self.scheduler.step()

        ckpt_dir_name = f"checkpoints/{self.paradigm}"
        if self.paradigm == "corosred":
            ckpt_dir_name += f"/phase_{self.phase.lower()}"
        ckpt_dir = Path(self.c_cfg.get("checkpoint_dir", self.c_cfg.get("dir", ckpt_dir_name)))
        if self.is_master and not benchmark:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            print(f"  Checkpoint Directory: {ckpt_dir} (Paradigm: {self.paradigm.upper()})")
        elif self.is_master and benchmark:
            print(f"  [Benchmark Mode] Starting throughput benchmark (Maximum limit: {max_bench_duration:.0f}s)...")

        start_time = time.time()
        bench_start_time = None
        latencies = []
        bench_steps = 0
        self.optimizer.zero_grad()

        for step in range(resume_step + 1, self.max_steps + 1):
            t_step_start = time.perf_counter()

            global_targets, idx_ptr = get_global_targets_contiguous_pytorch(dataset_matrix, idx_ptr, bs * grad_accum, self.seq_len, self.device)

            last_metrics = None
            
            for i in range(grad_accum):
                batch_seqs = global_targets[i * bs : (i + 1) * bs]

                if self.use_amp:
                    with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                        loss, metrics = self._execute_microbatch(batch_seqs)
                else:
                    loss, metrics = self._execute_microbatch(batch_seqs)

                loss = loss / grad_accum
                if self.use_scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                last_metrics = metrics

            if self.use_scaler:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            elif self.is_tpu:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                import torch_xla.core.xla_model as xm
                # xm.optimizer_step() triggers gradient reduction and internal mark_step()
                xm.optimizer_step(self.optimizer)
            else:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            self.optimizer.zero_grad()
            self.scheduler.step()
            self.global_step = step

            step_time_ms = (time.perf_counter() - t_step_start) * 1000.0

            # For benchmark mode: 5 warmup steps before collecting benchmark timers
            if benchmark:
                if step >= resume_step + 5 and bench_start_time is None:
                    bench_start_time = time.time()
                elif bench_start_time is not None:
                    latencies.append(step_time_ms)
                    bench_steps += 1
                    # Check benchmark duration condition
                    if (time.time() - bench_start_time) >= max_bench_duration:
                        bench_elapsed = time.time() - bench_start_time
                        self._print_benchmark_report(bench_steps, bench_elapsed, latencies, bs, grad_accum)
                        return

            if self.is_master and (step % 50 == 0 or step == 1 or step == self.max_steps or (benchmark and step % 10 == 0)):
                lr = self.scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time
                steps_taken = step - resume_step
                sps = steps_taken / elapsed if elapsed > 0 else 0
                tps = sps * bs * grad_accum * self.seq_len

                l_val = last_metrics['loss'].item() if last_metrics and 'loss' in last_metrics else 0.0
                ce_val = last_metrics['unweighted_ce'].item() if last_metrics and 'unweighted_ce' in last_metrics else 0.0
                
                log_msg = f"  [{self.paradigm.upper()}] Step {step:>6d}/{self.max_steps} | Loss: {l_val:>6.4f} | CE: {ce_val:>5.3f} | LR: {lr:.2e} | {sps:>5.1f} st/s | {tps:>9,.0f} tok/s"
                if not benchmark:
                    eta_mins = (self.max_steps - step) / sps / 60.0 if sps > 0 else 0.0
                    log_msg += f" | ETA: {eta_mins:>4.1f}m"
                print(log_msg, flush=True)

            if not benchmark and self.is_master and step % self.c_cfg.get("save_every_steps", 1000) == 0:
                ckpt_file = ckpt_dir / f"checkpoint_step_{step}.pt"
                self.save_checkpoint(ckpt_file)
                print(f"  [Checkpoint] Saved weights to {ckpt_file}")

        total_time = time.time() - start_time
        if benchmark:
            bench_elapsed = time.time() - (bench_start_time if bench_start_time else start_time)
            self._print_benchmark_report(bench_steps, bench_elapsed, latencies, bs, grad_accum)
            return

        if self.is_master:
            self.save_checkpoint(ckpt_dir / "checkpoint_final.pt")
            # Write standalone config.json for eval loader and downstream tools
            with open(ckpt_dir / "config.json", "w") as f:
                json.dump(self.cfg, f, indent=2)
            print("=" * 70)
            print(f"  {self.paradigm.upper()} PyTorch Training Complete! Total time: {total_time/60.0:.2f} minutes.")
            print(f"  Saved standalone model artifact to {ckpt_dir}/")
            print("=" * 70)
