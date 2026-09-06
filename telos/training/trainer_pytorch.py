"""
Unified PyTorch Trainer for all Telos paradigms (AR, MDLM, UNDLM, COROSred).
"""

import os
import queue
import threading
import time
import math
import json
import inspect
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
from telos.diffusion.corosred import (
    crsr_phase_a_loss_fn_pytorch,
    crsr_phase_b_loss_fn_pytorch,
    crsr_phase_b_self_conditioned_loss_fn_pytorch
)


class UnifiedPyTorchTrainer:
    """Unified PyTorch Trainer orchestrator for all paradigms."""

    def __init__(self, paradigm: str, model, cfg: dict, device_type: str = "cpu"):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed. Cannot use UnifiedPyTorchTrainer.")

        if torch.cuda.is_available():
            # Enable TF32 for matrix multiplications on Ampere+ architectures
            torch.set_float32_matmul_precision("high")
            # Allocator hygiene: Set expandable_segments to eliminate memory fragmentation during autotuning
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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

        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if str(device_type).lower() in ["tpu", "xla"]:
            try:
                # Use cached XLA device and consistent xla_model context to guarantee
                # InitializeComputationClient() is invoked at most once per process lifetime.
                from .xla_utils import get_xla_device, get_xla_world_size, is_xla_master, get_xla_spmd_mesh
                self.device = get_xla_device()
                self.is_tpu = True
                self.spmd_mesh = get_xla_spmd_mesh()
                self.is_spmd = self.spmd_mesh is not None
                self.world_size = get_xla_world_size()
                self.is_master = is_xla_master()
                if self.is_spmd:
                    print(f"  [Hardware] Detected PyTorch-XLA SPMD Topology ({self.world_size} Cores).")
                else:
                    print(f"  [Hardware] Detected PyTorch-XLA TPU Topology ({self.world_size} Cores).")
            except ImportError as e:
                print(f"Warning: torch_xla not installed ({e}). Falling back to CPU.")
                self.device = torch.device("cpu")
                self.is_tpu = False
                self.spmd_mesh = None
                self.is_spmd = False
                self.world_size = 1
                self.is_master = True
            except RuntimeError:
                # Re-raise hardware lock / initialization errors so users don't silently train on CPU
                raise
        elif str(device_type).lower() == "cuda":
            if local_rank != -1 and torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
                self.device = torch.device(f"cuda:{local_rank}")
                if not torch.distributed.is_initialized():
                    torch.distributed.init_process_group(backend="nccl")
                self.world_size = int(os.environ.get("WORLD_SIZE", 1))
                self.is_master = (int(os.environ.get("RANK", 0)) == 0)
                self.is_ddp = True
            else:
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.world_size = 1
                self.is_master = True
                self.is_ddp = False

            self.is_tpu = False
            self.spmd_mesh = None
            self.is_spmd = False
            self.n_gpus = torch.cuda.device_count()
            # Check native BF16 support (e.g. Turing T4 does not support BF16, Ampere/Ada/Hopper do)
            if not torch.cuda.is_bf16_supported() and self.precision == "bf16":
                print("  [Notice] Hardware lacks native BF16 support. Automatically falling back to FP16 with GradScaler.")
                self.precision = "fp16"
        else:
            self.device = torch.device(device_type)
            self.is_tpu = False
            self.spmd_mesh = None
            self.is_spmd = False
            self.world_size = 1
            self.is_master = True
            self.is_ddp = False

        self.model.to(self.device)
        self.special_lut = self.special_lut.to(self.device)

        # Explicit bfloat16 casting for TPU MXU hardware saturation
        if self.is_tpu and self.precision in ["bfloat16", "bf16"]:
            self.model.to(dtype=torch.bfloat16)
            print("  [Precision] TPU model weights cast to native torch.bfloat16.")

        # Multi-GPU wrapping: Prefer DDP over deprecated DataParallel
        if getattr(self, "is_ddp", False):
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank])
            if self.is_master:
                print(f"  [Hardware] Multi-GPU DDP initialized across {self.world_size} processes.")
        elif getattr(self, "n_gpus", 1) > 1 and not self.is_tpu and self.device.type == "cuda":
            print("  [Hardware Notice] Multi-GPU detected without torchrun. For 1.8x-7x throughput scaling, launch with 'torchrun'. Falling back to DataParallel.")
            self.model = nn.DataParallel(self.model)

        # Gradient checkpointing activation
        if self.t_cfg.get("gradient_checkpointing", False) or self.m_cfg.get("use_grad_checkpoint", False):
            if hasattr(self.model, "use_grad_checkpoint"):
                self.model.use_grad_checkpoint = True
            elif hasattr(self.model, "module") and hasattr(self.model.module, "use_grad_checkpoint"):
                self.model.module.use_grad_checkpoint = True
            print("  [Memory] PyTorch Gradient Checkpointing Enabled.")

        # torch.compile integration (fusing RMSNorm, SwiGLU, and RoPE)
        if self.t_cfg.get("compile", False) and hasattr(torch, "compile") and self.device.type == "cuda":
            mode = self.t_cfg.get("compile_mode", "reduce-overhead")
            try:
                self.model = torch.compile(self.model, mode=mode)
                print(f"  [Compiler] torch.compile enabled (mode={mode}).")
            except Exception as e:
                print(f"  [Compiler] torch.compile skipped: {e}")

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
        use_fused = (self.device.type == "cuda") and (
            "fused" in inspect.signature(torch.optim.AdamW).parameters
        )
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

        # On TPU, quantize LR updates to 10-step cadence to avoid XLA graph recompilations from Python float changes
        lr_cadence = 10 if self.is_tpu else 1
        self.scheduler = WarmupCosineLR(
            self.optimizer,
            warmup_steps=self.warmup_steps,
            max_steps=self.max_steps,
            min_lr=self.min_lr,
            update_cadence=lr_cadence
        )

        # Autocast AMP precision
        if self.is_tpu:
            self.use_amp = (self.precision in ["fp16", "bf16"]) and hasattr(torch.amp, "autocast")
            self.amp_device = "xla"
            self.amp_dtype = torch.bfloat16
        else:
            self.use_amp = (self.precision in ["fp16", "bf16"]) and (self.device.type in ["cuda", "mps"])
            self.amp_device = self.device.type
            self.amp_dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16

        # Modern GradScaler API for FP16 training on CUDA
        self.use_scaler = (self.precision == "fp16") and (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda") if self.use_scaler else None

        # Pre-sampled device-resident Beta(1.5, 1.5) buffer to eliminate per-microbatch host RNG & PCIe syncs
        if self.paradigm in ["mdlm", "undlm"]:
            self.beta_buffer_size = 1_000_000
            beta_np = sample_beta_timesteps(self.beta_buffer_size)
            beta_dtype = torch.bfloat16 if (self.is_tpu or self.precision == "bf16") else torch.float32
            self.beta_buffer = torch.from_numpy(beta_np).to(device=self.device, dtype=beta_dtype)
            self.beta_idx = 0

        self.global_step = 0

    def _sample_beta_timesteps(self, bs: int) -> torch.Tensor:
        """Fetches bs timesteps directly from the device-resident Beta buffer without CPU copy."""
        if self.beta_idx + bs > self.beta_buffer_size:
            self.beta_idx = 0
        start = self.beta_idx
        self.beta_idx += bs
        return self.beta_buffer[start : start + bs]

    def save_checkpoint(self, path: str | Path, sync: bool = False):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.cfg,
        }
        if self.is_tpu:
            import torch_xla.core.xla_model as xm
            xm.mark_step()
            # xm.save serializes XLA tensors to CPU host memory safely without graph synchronization stalls
            xm.save(checkpoint, path)
        elif sync:
            torch.save(checkpoint, path)
        else:
            # Asynchronous checkpoint saving on a background worker thread to prevent hot-loop sync stalls
            def _async_save():
                torch.save(checkpoint, path)
            save_thread = threading.Thread(target=_async_save, daemon=True)
            save_thread.start()

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
            t_vals = self._sample_beta_timesteps(bs)
            masked_ids, mask_pos, t_vals_out = apply_masking_pytorch(batch_seqs, t_vals, mask_token_id=1, special_token_lut=self.special_lut)
            logits = self.model(masked_ids, mask_override=False)
            loss, metrics = mdlm_loss_pytorch(logits, batch_seqs, mask_pos, t_vals_out)

        elif self.paradigm == "undlm":
            bs = batch_seqs.shape[0]
            t_vals = self._sample_beta_timesteps(bs)
            noisy_ids, corrupt_mask, t_vals_out = apply_uniform_noise_pytorch(batch_seqs, t_vals, self.vocab_size, special_token_lut=self.special_lut)
            logits = self.model(noisy_ids, mask_override=False)
            loss, metrics = undlm_loss_pytorch(logits, batch_seqs, t_vals_out, special_token_lut=self.special_lut)

        elif self.paradigm == "corosred":
            if self.phase == "A":
                k_amb = self.crsr_cfg.get("k_amb", 5)
                loss, metrics = crsr_phase_a_loss_fn_pytorch(self.model, batch_seqs, self.vocab_size, special_token_lut=self.special_lut, k_amb=k_amb)
            else:
                mask_token_id = self.m_cfg.get("mask_token_id", 1)
                mask_prob = float(self.crsr_cfg.get("mask_prob", 0.15))
                self_cond = self.crsr_cfg.get("self_condition", True)
                self_cond_prob = float(self.crsr_cfg.get("self_cond_prob", 0.5))
                if self_cond and self_cond_prob > 0.0:
                    loss, metrics = crsr_phase_b_self_conditioned_loss_fn_pytorch(
                        self.model,
                        batch_seqs,
                        self.vocab_size,
                        mask_token_id=mask_token_id,
                        mask_prob=mask_prob,
                        self_cond_prob=self_cond_prob
                    )
                else:
                    loss, metrics = crsr_phase_b_loss_fn_pytorch(
                        self.model,
                        batch_seqs,
                        self.vocab_size,
                        mask_token_id=mask_token_id,
                        mask_prob=mask_prob
                    )
        
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

        # Background asynchronous batch prefetcher
        prefetch_queue = queue.Queue(maxsize=1)

        def prefetch_worker():
            nonlocal idx_ptr
            for _ in range(resume_step + 1, self.max_steps + 1):
                batch_t, idx_ptr = get_global_targets_contiguous_pytorch(
                    dataset_matrix, idx_ptr, bs * grad_accum, self.seq_len, self.device, non_blocking=True
                )
                prefetch_queue.put(batch_t)

        prefetch_thread = threading.Thread(target=prefetch_worker, daemon=True)
        prefetch_thread.start()

        for step in range(resume_step + 1, self.max_steps + 1):
            t_step_start = time.perf_counter()

            global_targets = prefetch_queue.get()

            last_metrics = None
            
            for i in range(grad_accum):
                batch_seqs = global_targets[i * bs : (i + 1) * bs]

                # In PyTorch-XLA SPMD mode, shard batch dimension across TPU chips so PJRT
                # produces valid PjRtShardedData buffers and avoids ExecuteReplicated NULL pointer SIGSEGV.
                if getattr(self, "is_spmd", False) and self.spmd_mesh is not None:
                    import torch_xla.distributed.spmd as xs
                    xs.mark_sharding(batch_seqs, self.spmd_mesh, ("data", None))

                if self.use_amp:
                    with torch.amp.autocast(device_type=getattr(self, "amp_device", self.device.type), dtype=self.amp_dtype):
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
                import torch_xla.core.xla_model as xm
                # In SPMD mode, xs.mark_sharding on the batch automatically triggers the partitioner all-reduce.
                # Only call xm.reduce_gradients in multi-process non-SPMD mode.
                if not getattr(self, "is_spmd", False):
                    xm.reduce_gradients(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                # xm.optimizer_step() applies updates and triggers internal mark_step()
                xm.optimizer_step(self.optimizer)
                xm.mark_step()
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

                l_val = float(last_metrics['loss'].detach().cpu().item()) if last_metrics and 'loss' in last_metrics else 0.0
                ce_val = float(last_metrics['unweighted_ce'].detach().cpu().item()) if last_metrics and 'unweighted_ce' in last_metrics else 0.0
                
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
            if self.is_tpu:
                import torch_xla.core.xla_model as xm
                xm.mark_step()
            self.save_checkpoint(ckpt_dir / "checkpoint_final.pt", sync=True)
            # Write standalone config.json for eval loader and downstream tools
            with open(ckpt_dir / "config.json", "w") as f:
                json.dump(self.cfg, f, indent=2, default=str)
            print("=" * 70)
            print(f"  {self.paradigm.upper()} PyTorch Training Complete! Total time: {total_time/60.0:.2f} minutes.")
            print(f"  Saved standalone model artifact to {ckpt_dir}/")
            print("=" * 70)
