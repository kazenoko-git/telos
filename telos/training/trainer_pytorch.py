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
from contextlib import nullcontext
import numpy as np
from pathlib import Path

# Allocator hygiene: Set expandable_segments BEFORE torch is imported to prevent memory fragmentation during autotuning
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
from telos.training.schedule import COROSredSchedule, DynamicMetricTracker
from telos.diffusion.corosred_unified import (
    RoutingMaskCache,
    corosred_unified_step_pytorch,
)
from telos.training.monitor import DualMetricMonitor


class UnifiedPyTorchTrainer:
    """Unified PyTorch Trainer orchestrator for all paradigms."""

    def __init__(self, paradigm: str, model, cfg: dict, device_type: str = "cpu"):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed. Cannot use UnifiedPyTorchTrainer.")

        if torch.cuda.is_available():
            # Enable TF32 for matrix multiplications on Ampere+ architectures
            torch.set_float32_matmul_precision("high")

        self.paradigm = paradigm.lower()
        self.crsr_cfg = cfg.get("crsr", cfg.get("corosred", {}))
        
        # Determine if COROSred should run in unified continuous mode (default) or legacy multi-phase
        if self.paradigm == "corosred":
            if self.crsr_cfg.get("unified", None) is not None:
                self.is_unified = bool(self.crsr_cfg["unified"])
            elif self.crsr_cfg.get("legacy_phases", False):
                self.is_unified = False
            elif "phase" in cfg and cfg["phase"] in ["A", "B", "C"]:
                self.is_unified = False
            else:
                self.is_unified = True
            raw_phase = self.crsr_cfg.get("phase", cfg.get("phase", "unified"))
            self.phase = str(raw_phase).upper() if raw_phase else "UNIFIED"
        else:
            self.is_unified = False
            self.phase = "A"
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
                try:
                    import torch_xla.core.xla_model as xm
                    self.rank = xm.get_ordinal()
                except Exception:
                    self.rank = 0
                if self.is_spmd:
                    print(f"  [Hardware] Detected PyTorch-XLA SPMD Topology ({self.world_size} Cores).")
                else:
                    print(f"  [Hardware] Detected PyTorch-XLA TPU Topology ({self.world_size} Cores, Rank {self.rank}).")
            except ImportError as e:
                print(f"Warning: torch_xla not installed ({e}). Falling back to CPU.")
                self.device = torch.device("cpu")
                self.is_tpu = False
                self.spmd_mesh = None
                self.is_spmd = False
                self.world_size = 1
                self.rank = 0
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
                self.rank = int(os.environ.get("RANK", 0))
                self.is_master = (self.rank == 0)
                self.is_ddp = True
            else:
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.world_size = 1
                self.rank = 0
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
            self.rank = 0
            self.is_master = True
            self.is_ddp = False

        self.model.to(self.device)
        self.special_lut = self.special_lut.to(self.device)

        # In mixed-precision training (bfloat16), model parameters in memory MUST remain in float32
        # so AdamW gradient updates (order 10^-5 to 10^-6) do not mathematically underflow to zero
        # against bfloat16's 7-bit mantissa (machine epsilon 2^-7 ~ 0.0078).
        # On TPU, Matrix Multiply Units (MXUs) execute systolic GEMMs in bfloat16 via hardware autocast
        # while optimizer states and parameter weights remain unadulterated in float32.
        self.use_master_weights = False
        if self.is_tpu and self.precision in ["bfloat16", "bf16"]:
            self.model.to(dtype=torch.float32)
            print("  [Precision] TPU model parameters kept in float32 (AdamW updates preserved; forward pass autocast to bfloat16).")

        # Multi-GPU wrapping: Prefer DDP over deprecated DataParallel
        if getattr(self, "is_ddp", False):
            # In legacy COROSred Phase B and C, reliability_head is frozen / only used for routing,
            # so find_unused_parameters=True prevents DDP unused parameter reduction assertions.
            # In Unified COROSred, both causal backbone and reliability head receive gradients,
            # so find_unused_parameters=False eliminates per-step DDP parameter tree traversal overhead.
            find_unused = (
                self.paradigm == "corosred"
                and not getattr(self, "is_unified", False)
                and str(self.phase).upper() in ["B", "C"]
            )
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[local_rank],
                find_unused_parameters=find_unused
            )
            if self.is_master:
                print(f"  [Hardware] Multi-GPU DDP initialized across {self.world_size} processes.")
        elif getattr(self, "n_gpus", 1) > 1 and not self.is_tpu and self.device.type == "cuda":
            print("  [Hardware Notice] Multi-GPU detected without torchrun. For 1.8x-7x throughput scaling, launch with 'torchrun'. Falling back to DataParallel.")
            self.model = nn.DataParallel(self.model)

        # Gradient checkpointing activation:
        # Avoid auto-enabling on high-VRAM CUDA GPUs (A100/H100/4090) for small/medium models (<=100M),
        # preserving ~33% compute FLOPs by eliminating backward activation recomputation.
        cuda_needs_chkpt = False
        if self.device.type == "cuda" and not self.is_tpu:
            cuda_gb = 0.0
            try:
                cuda_gb = torch.cuda.get_device_properties(self.device).total_memory / (1024 ** 3)
            except Exception:
                pass
            # Auto-enable only if VRAM <= 16GB or model width >= 1024
            if (cuda_gb > 0 and cuda_gb <= 16.0) or (self.m_cfg.get("d_model", 512) >= 1024):
                cuda_needs_chkpt = True

        # Only enable gradient checkpointing for large models (>=100M, d_model >= 768)
        # or when explicitly requested. On 15M/25M/50M, HBM footprint is <3.5GB (out of 16GB),
        # so gradient checkpointing unnecessarily burns ~35% throughput recomputing activations.
        tpu_needs_chkpt = self.is_tpu and (self.m_cfg.get("d_model", 512) >= 768 or self.t_cfg.get("batch_size", 32) > 64)
        auto_chkpt = (
            tpu_needs_chkpt
            or cuda_needs_chkpt
            or self.t_cfg.get("gradient_checkpointing", False)
            or self.m_cfg.get("use_grad_checkpoint", False)
        )
        if auto_chkpt:
            if hasattr(self.model, "use_grad_checkpoint"):
                self.model.use_grad_checkpoint = True
            elif hasattr(self.model, "module") and hasattr(self.model.module, "use_grad_checkpoint"):
                self.model.module.use_grad_checkpoint = True
            print("  [Memory] PyTorch Gradient Checkpointing Enabled.")


        # torch.compile integration (fusing RMSNorm, SwiGLU, and RoPE)
        if self.t_cfg.get("compile", False) and hasattr(torch, "compile") and self.device.type == "cuda":
            # Default to "default" instead of "reduce-overhead" to avoid static CUDA Graph buffer
            # aliasing crashes when self-conditioned diffusion invokes the model multiple times per step.
            mode = self.t_cfg.get("compile_mode", "default")
            try:
                self.model = torch.compile(self.model, mode=mode)
                print(f"  [Compiler] torch.compile enabled (mode={mode}).")
            except Exception as e:
                print(f"  [Compiler] torch.compile skipped: {e}")

        self.max_steps = int(self.t_cfg.get("max_steps", 5000))
        self.max_lr = float(self.t_cfg.get("max_lr", 3e-4))
        self.min_lr = float(self.t_cfg.get("min_lr", 3e-5))
        self.warmup_steps = int(self.t_cfg.get("warmup_steps", 100))
        self.weight_decay = float(self.t_cfg.get("weight_decay", 0.1))
        self.grad_clip = float(self.t_cfg.get("grad_clip", 1.0))

        if self.paradigm == "corosred":
            if getattr(self, "is_unified", False):
                self.schedule = COROSredSchedule(
                    max_steps=self.max_steps,
                    alpha_max=float(self.crsr_cfg.get("alpha_max", 0.85)),
                    alpha_min=float(self.crsr_cfg.get("alpha_min", 0.20)),
                    beta_min=float(self.crsr_cfg.get("beta_min", 0.15)),
                    beta_max=float(self.crsr_cfg.get("beta_max", 0.70)),
                    gamma_max=float(self.crsr_cfg.get("gamma_max", 0.10)),
                    hold_fraction=float(self.crsr_cfg.get("hold_fraction", 0.20)),
                    decay_power=float(self.crsr_cfg.get("decay_power", 2.5)),
                    acc_gate_threshold=float(self.crsr_cfg.get("acc_gate_threshold", 0.65)),
                )
                self.metric_tracker = DynamicMetricTracker(
                    trust_region=float(self.crsr_cfg.get("trust_region", 0.20)),
                    rebalance_temp=float(self.crsr_cfg.get("rebalance_temp", 0.25)),
                )
                self.routing_cache = RoutingMaskCache(
                    refresh_every_steps=int(self.crsr_cfg.get("routing_cache_steps", 50)),
                    k_targeted_ratio=float(self.crsr_cfg.get("k_targeted_ratio", 0.70)),
                )
                self.adaptive_rebalance = bool(self.crsr_cfg.get("adaptive_rebalance", False))
                self.causal_ratio = float(self.crsr_cfg.get("causal_ratio", 0.75))
                self.dual_monitor = None
                if self.is_master:
                    print(
                        f"  [COROSred Unified] Initialized continuous multi-objective loop: "
                        f"alpha=[{self.schedule.alpha_max:.2f}->{self.schedule.alpha_min:.2f} floor], "
                        f"beta=[{self.schedule.beta_min:.2f}->{self.schedule.beta_max:.2f}], "
                        f"gamma_max={self.schedule.gamma_max:.2f}, hold={self.schedule.hold_fraction*100:.0f}%, "
                        f"adaptive_rebalance={self.adaptive_rebalance}."
                    )

        # Separate parameters into decayed and non-decayed groups:
        # Standard transformer optimization applies 0 weight decay to 1D parameters (biases, layer norms, RMS norms)
        # and embedding lookup tables to avoid regularizing scale/shift parameters.
        self.param_to_master = []
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if self.use_master_weights:
                    # Allocate fp32 master parameter copy to prevent updates from rounding to zero in bf16
                    mp = param.detach().clone().float().requires_grad_(True)
                    self.param_to_master.append((param, mp))
                    target_param = mp
                else:
                    target_param = param

                if param.ndim == 1 or "embed" in name:
                    no_decay_params.append(target_param)
                else:
                    decay_params.append(target_param)

        self.master_params = [mp for _, mp in self.param_to_master] if self.use_master_weights else None

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
            # Enable hardware bfloat16 autocast on TPU to utilize systolic MXU arrays
            # while parameters in HBM remain in float32 for unadulterated AdamW updates.
            self.use_amp = (self.precision in ["fp16", "bf16", "bfloat16"])
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
            # Retain float32 precision across all backends (MLX, CUDA, TPU) for exact masking probabilities
            self.beta_buffer = torch.from_numpy(beta_np).to(device=self.device, dtype=torch.float32)
            self.beta_counter = torch.zeros((), dtype=torch.int64, device=self.device)
            self.beta_idx = 0

        # Checkpoint thread primitives to eliminate torn async saves and filesystem race conditions
        self._save_lock = threading.Lock()
        self._save_thread = None

        self.global_step = 0

    def _sample_beta_timesteps(self, bs: int) -> torch.Tensor:
        """
        Fetches bs timesteps. On TPU, indexes device-resident Beta(1.5, 1.5) buffer
        with device-computed indices to ensure static XLA shapes and exact Beta distribution.
        """
        if self.is_tpu:
            # Device-computed tensor index: static shape (bs,), zero dynamic slice recompilations in XLA
            idx = (self.beta_counter + torch.arange(bs, device=self.device, dtype=torch.int64)) % self.beta_buffer_size
            self.beta_counter = (self.beta_counter + bs) % self.beta_buffer_size
            return self.beta_buffer[idx]

        if self.beta_idx + bs > self.beta_buffer_size:
            self.beta_idx = 0
        start = self.beta_idx
        self.beta_idx += bs
        return self.beta_buffer[start : start + bs]

    def save_checkpoint(self, path: str | Path, sync: bool = False):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Join any prior background save thread to ensure two saves never overlap or race
        if self._save_thread is not None and self._save_thread.is_alive():
            self._save_thread.join()

        if self.is_tpu:
            import torch_xla.core.xla_model as xm
            xm.mark_step()
            checkpoint = {
                "global_step": self.global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "config": self.cfg,
            }
            str_path = str(path)
            # xm.save serializes XLA tensors to CPU host memory safely.
            # master_only=False is required because self.is_master already guards this method,
            # and PyTorch-XLA PJRT can evaluate is_master_ordinal inconsistently across processes.
            try:
                xm.save(checkpoint, str_path, master_only=False)
            except Exception as e:
                print(f"  [Checkpoint Warning] xm.save error: {e}. Falling back to CPU serialization.")

            # Verification and robust fallback: ensure file was actually written to disk and is non-empty
            if not path.exists() or path.stat().st_size == 0:
                print(f"  [Checkpoint] xm.save produced no file at {str_path}. Converting state tensors to CPU explicitly...")
                cpu_model = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                raw_opt = self.optimizer.state_dict()
                cpu_opt = {
                    "state": {
                        k: {sk: sv.detach().cpu().clone() if isinstance(sv, torch.Tensor) else sv for sk, sv in v.items()}
                        for k, v in raw_opt.get("state", {}).items()
                    },
                    "param_groups": raw_opt.get("param_groups", [])
                }
                cpu_checkpoint = {
                    "global_step": self.global_step,
                    "model_state_dict": cpu_model,
                    "optimizer_state_dict": cpu_opt,
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "config": self.cfg,
                }
                torch.save(cpu_checkpoint, str_path)

            if not path.exists() or path.stat().st_size == 0:
                raise RuntimeError(f"FATAL: Checkpoint file could not be created at {str_path}!")

            print(f"  [Checkpoint Verified] Successfully saved {str_path} ({path.stat().st_size / 1e6:.1f} MB)")
            return

        # Clone state dicts to detached CPU tensors to prevent torn-state corruption from concurrent in-place mutations
        cpu_model_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        raw_opt_state = self.optimizer.state_dict()
        cpu_opt_state = {
            "state": {
                k: {sk: sv.detach().cpu().clone() if isinstance(sv, torch.Tensor) else sv for sk, sv in v.items()}
                for k, v in raw_opt_state.get("state", {}).items()
            },
            "param_groups": raw_opt_state.get("param_groups", [])
        }

        checkpoint = {
            "global_step": self.global_step,
            "model_state_dict": cpu_model_state,
            "optimizer_state_dict": cpu_opt_state,
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.cfg,
        }

        if sync:
            torch.save(checkpoint, path)
        else:
            # Asynchronous checkpoint saving on a background worker thread with cloned CPU tensors
            def _async_save():
                with self._save_lock:
                    torch.save(checkpoint, path)
            self._save_thread = threading.Thread(target=_async_save, daemon=True)
            self._save_thread.start()

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

        # Synchronize fp32 master weights from loaded model parameters to prevent overwriting with stale init values
        if getattr(self, "use_master_weights", False):
            for p, mp in self.param_to_master:
                mp.data.copy_(p.data.float())

    def _execute_microbatch(self, batch_seqs, compute_metrics: bool = False):
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
            if getattr(self, "is_unified", False):
                mask_token_id = self.m_cfg.get("mask_token_id", 1)
                mask_prob = float(self.crsr_cfg.get("mask_prob", 0.15))
                k_amb = int(self.crsr_cfg.get("k_amb", 5))
                
                # Fetch continuous schedule weights driven by step progress and empirical EMAs
                lrh_acc_ema = self.metric_tracker.lrh_acc_ema if hasattr(self, "metric_tracker") else None
                lrh_auc_ema = self.metric_tracker.lrh_auc_ema if hasattr(self, "metric_tracker") else None
                sched_w = self.schedule.get_weights(self.global_step, lrh_acc_ema, lrh_auc_ema)

                loss, metrics = corosred_unified_step_pytorch(
                    self.model,
                    batch_seqs,
                    self.vocab_size,
                    schedule_weights=sched_w,
                    mask_token_id=mask_token_id,
                    mask_prob=mask_prob,
                    special_token_lut=self.special_lut,
                    k_amb=k_amb,
                    causal_ratio=getattr(self, "causal_ratio", 0.75),
                    routing_cache=getattr(self, "routing_cache", None),
                    metric_tracker=getattr(self, "metric_tracker", None),
                    adaptive_rebalance=getattr(self, "adaptive_rebalance", False),
                    compute_metrics=compute_metrics,
                )
            elif self.phase == "A":
                # Phase A: Causal Autoregressive backbone pretraining + detachable Learned Reliability Head
                k_amb = self.crsr_cfg.get("k_amb", 5)
                loss, metrics = crsr_phase_a_loss_fn_pytorch(self.model, batch_seqs, self.vocab_size, special_token_lut=self.special_lut, k_amb=k_amb)
            elif self.phase == "B":
                # Phase B: 15% Uniform Random Masking on clean ground-truth tokens (bidirectional infilling)
                mask_token_id = self.m_cfg.get("mask_token_id", 1)
                mask_prob = float(self.crsr_cfg.get("mask_prob", 0.15))
                loss, metrics = crsr_phase_b_loss_fn_pytorch(
                    self.model,
                    batch_seqs,
                    self.vocab_size,
                    mask_token_id=mask_token_id,
                    mask_prob=mask_prob
                )
            elif self.phase == "C":
                # Phase C: Self-Conditioned Model Drafts + Confidence Routing (70% low-confidence, 30% exploration)
                mask_token_id = self.m_cfg.get("mask_token_id", 1)
                mask_prob = float(self.crsr_cfg.get("mask_prob", 0.15))
                self_cond_prob = float(self.crsr_cfg.get("self_cond_prob", 0.5))
                loss, metrics = crsr_phase_b_self_conditioned_loss_fn_pytorch(
                    self.model,
                    batch_seqs,
                    self.vocab_size,
                    mask_token_id=mask_token_id,
                    mask_prob=mask_prob,
                    self_cond_prob=self_cond_prob
                )
            else:
                raise ValueError(f"Unknown COROSred phase: '{self.phase}'. Supported phases are 'A', 'B', 'C', or 'unified'.")
        
        return loss, metrics

    def _print_benchmark_report(self, steps: int, elapsed: float, latencies: list[float], bs: int, grad_accum: int):
        """Prints a publication-style benchmark report table and saves results to JSON."""
        if not self.is_master:
            return

        # In SPMD mode, batch size is already global across TPU mesh; avoid inflating tokens_processed by world_size
        world_mult = getattr(self, "world_size", 1) if not getattr(self, "is_spmd", False) else 1
        tokens_processed = steps * bs * grad_accum * self.seq_len * world_mult
        sps = steps / elapsed if elapsed > 0 else 0.0
        tps = tokens_processed / elapsed if elapsed > 0 else 0.0
        mean_lat = np.mean(latencies) if latencies else 0.0
        p50_lat = np.median(latencies) if latencies else 0.0
        p95_lat = np.percentile(latencies, 95) if latencies else 0.0

        device_desc = f"{self.device.type.upper()}"
        if getattr(self, "n_gpus", 1) > 1:
            device_desc += f" ({self.n_gpus} GPUs DataParallel)"
        elif self.is_tpu:
            device_desc += f" ({self.world_size} TPU Cores)"

        eff_batch = bs * grad_accum * world_mult

        print("\n" + "=" * 76)
        print("  TELOS UNIFIED BENCHMARK REPORT (PyTorch)")
        print("=" * 76)
        print(f"  Paradigm:             {self.paradigm.upper()}")
        print(f"  Hardware Target:      {device_desc}")
        print(f"  Precision:            {self.precision.upper()} (AMP: {self.use_amp})")
        print(f"  Batch Config:         batch_size={bs}, grad_accum={grad_accum}, seq_len={self.seq_len}")
        print(f"  Total Effective Batch: {eff_batch} sequences ({eff_batch * self.seq_len:,} tokens/step)")
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

        # Initialize DualMetricMonitor on held-out validation data if running unified COROSred (master rank only)
        if getattr(self, "is_unified", False) and self.is_master:
            val_path = d_cfg.get("val_path", d_cfg.get("val_dataset_path", None))
            if val_path and Path(val_path).exists():
                try:
                    val_data = np.memmap(val_path, dtype=dtype, mode="r")
                    val_seqs = len(val_data) // self.seq_len
                    val_matrix = val_data[:val_seqs * self.seq_len].reshape(val_seqs, self.seq_len)
                    self.dual_monitor = DualMetricMonitor(val_matrix, self.vocab_size, self.seq_len, self.device)
                    if self.is_master:
                        print(f"  [Dual Monitor] Initialized on validation split ({val_seqs} sequences).")
                except Exception as e:
                    if self.is_master:
                        print(f"  [Dual Monitor] Could not load val data: {e}")
            if getattr(self, "dual_monitor", None) is None and len(dataset_matrix) > 200:
                self.dual_monitor = DualMetricMonitor(dataset_matrix[-200:], self.vocab_size, self.seq_len, self.device)
                if self.is_master:
                    print("  [Dual Monitor] Initialized on tail 200 sequences of training corpus.")

        bs = int(self.t_cfg.get("batch_size", 16))
        grad_accum = int(self.t_cfg.get("gradient_accumulation", 1))
        local_step_seqs = bs * grad_accum
        world_mult = getattr(self, "world_size", 1) if not getattr(self, "is_spmd", False) else 1
        cluster_step_seqs = local_step_seqs * world_mult
        rank = getattr(self, "rank", 0) if not getattr(self, "is_spmd", False) else 0

        # Benchmark duration strictly capped at 300.0 seconds (5 minutes)
        max_bench_duration = min(float(benchmark_duration), 300.0)

        n_rows = len(dataset_matrix)
        self.global_step = resume_step
        if resume_step > 0:
            seqs_consumed = resume_step * cluster_step_seqs
            idx_ptr = (seqs_consumed + rank * local_step_seqs) % n_rows
            # Advance scheduler
            for _ in range(resume_step):
                self.scheduler.step()
        else:
            idx_ptr = (rank * local_step_seqs) % n_rows

        ckpt_dir_name = f"checkpoints/{self.paradigm}"
        if self.paradigm == "corosred":
            if getattr(self, "is_unified", False):
                ckpt_dir_name += "/unified"
            else:
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
        if self.use_master_weights:
            self.model.zero_grad()

        # Background asynchronous batch prefetcher (CUDA only)
        # PyTorch-XLA does NOT support multi-threaded tensor allocation/transfer to device;
        # transfers on background worker threads cause heavy XLA runtime mutex serialization.
        if self.is_tpu:
            prefetch_queue = None
        else:
            prefetch_queue = queue.Queue(maxsize=1)

            def prefetch_worker():
                nonlocal idx_ptr
                try:
                    for _ in range(resume_step + 1, self.max_steps + 1):
                        batch_t, _ = get_global_targets_contiguous_pytorch(
                            dataset_matrix, idx_ptr, local_step_seqs, self.seq_len, self.device, non_blocking=True
                        )
                        idx_ptr = (idx_ptr + cluster_step_seqs) % n_rows
                        prefetch_queue.put((batch_t, None))
                except Exception as e:
                    # Pass exception through the queue so consumer thread does not hang indefinitely
                    prefetch_queue.put((None, e))

            prefetch_thread = threading.Thread(target=prefetch_worker, daemon=True)
            prefetch_thread.start()

        for step in range(resume_step + 1, self.max_steps + 1):
            t_step_start = time.perf_counter()

            if prefetch_queue is not None:
                item, exc = prefetch_queue.get()
                if exc is not None:
                    raise exc
                global_targets = item
            else:
                global_targets, _ = get_global_targets_contiguous_pytorch(
                    dataset_matrix, idx_ptr, local_step_seqs, self.seq_len, self.device, non_blocking=False
                )
                idx_ptr = (idx_ptr + cluster_step_seqs) % n_rows
            # Periodically replenish routing cache only once confidence-routed masking is active (after hold phase)
            has_routing = getattr(self, "is_unified", False) and hasattr(self, "routing_cache")
            if has_routing and self.routing_cache.should_refresh(step):
                sched_w_check = self.schedule.get_weights(step)
                if sched_w_check.get("mask_blend", 0.0) > 0.0:
                    self.routing_cache.refresh(
                        self.model,
                        upcoming_seqs=global_targets,
                        current_step=step,
                        mask_prob=float(self.crsr_cfg.get("mask_prob", 0.15)),
                    )

            last_metrics = None
            
            for i in range(grad_accum):
                batch_seqs = global_targets[i * bs : (i + 1) * bs]

                # In PyTorch-XLA SPMD mode, shard batch dimension across TPU chips so PJRT
                # produces valid PjRtShardedData buffers and avoids ExecuteReplicated NULL pointer SIGSEGV.
                if getattr(self, "is_spmd", False) and self.spmd_mesh is not None:
                    import torch_xla.distributed.spmd as xs
                    xs.mark_sharding(batch_seqs, self.spmd_mesh, ("data", None))

                is_log_step = (
                    step <= 5
                    or (step <= 50 and step % 10 == 0)
                    or step % 50 == 0
                    or step == self.max_steps
                    or (benchmark and step % 10 == 0)
                )
                eval_metrics = (i == grad_accum - 1) and is_log_step
                # DDP gradient accumulation optimization: disable all-reduce on non-final microbatches
                sync_ctx = self.model.no_sync() if (getattr(self, "is_ddp", False) and i < grad_accum - 1) else nullcontext()
                with sync_ctx:
                    if self.use_amp:
                        with torch.amp.autocast(device_type=getattr(self, "amp_device", self.device.type), dtype=self.amp_dtype):
                            loss, metrics = self._execute_microbatch(batch_seqs, compute_metrics=eval_metrics)
                    else:
                        loss, metrics = self._execute_microbatch(batch_seqs, compute_metrics=eval_metrics)

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
                # Copy gradients from bfloat16 model parameters to float32 master parameters
                if self.use_master_weights:
                    for p, mp in self.param_to_master:
                        if p.grad is not None:
                            mp.grad = p.grad.float()

                # In SPMD mode, xs.mark_sharding on the batch automatically triggers the partitioner all-reduce.
                if not getattr(self, "is_spmd", False):
                    xm.reduce_gradients(self.optimizer)
                if self.grad_clip > 0:
                    target_params = self.master_params if self.use_master_weights else self.model.parameters()
                    nn.utils.clip_grad_norm_(target_params, self.grad_clip)
                xm.optimizer_step(self.optimizer)

                # Synchronize updated float32 master weights back to bfloat16 model parameters
                if self.use_master_weights:
                    for p, mp in self.param_to_master:
                        p.data.copy_(mp.data.to(p.dtype))

                xm.mark_step()

            else:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            self.optimizer.zero_grad()
            if self.use_master_weights:
                self.model.zero_grad()
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

            if self.is_master and (
                step <= 5
                or (step <= 50 and step % 10 == 0)
                or step % 50 == 0
                or step == self.max_steps
                or (benchmark and step % 10 == 0)
            ):
                lr = self.scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time
                steps_taken = step - resume_step
                sps = steps_taken / elapsed if elapsed > 0 else 0
                world_mult = getattr(self, "world_size", 1) if not getattr(self, "is_spmd", False) else 1
                tps = sps * bs * grad_accum * self.seq_len * world_mult

                l_val = float(last_metrics['loss'].detach().cpu().item()) if last_metrics and 'loss' in last_metrics else 0.0
                ce_val = float(last_metrics['unweighted_ce'].detach().cpu().item()) if last_metrics and 'unweighted_ce' in last_metrics else 0.0
                
                if getattr(self, "is_unified", False) and last_metrics and "causal_ce" in last_metrics:
                    c_ce = float(last_metrics["causal_ce"].detach().cpu().item())
                    i_ce = float(last_metrics["infill_ce"].detach().cpu().item())
                    a_w = float(last_metrics.get("alpha", 0.0))
                    b_w = float(last_metrics.get("beta", 0.0))
                    g_w = float(last_metrics.get("gamma", 0.0))
                    raw_acc = last_metrics.get("lrh_acc", 0.0)
                    l_acc = float(raw_acc.detach().cpu().item()) if isinstance(raw_acc, torch.Tensor) else float(raw_acc)
                    l_auc = float(last_metrics.get("lrh_auc", 0.0))
                    log_msg = (
                        f"  [COROSRED-UNIFIED] Step {step:>6d}/{self.max_steps} | Loss: {l_val:>6.4f} | "
                        f"C-CE: {c_ce:>5.3f} | I-CE: {i_ce:>5.3f} | "
                        f"α={a_w:.2f} β={b_w:.2f} γ={g_w:.2f} | Acc={l_acc:.2f} AUC={l_auc:.2f} | "
                        f"{sps:>5.1f} st/s | {tps:>9,.0f} tok/s"
                    )
                else:
                    log_msg = f"  [{self.paradigm.upper()}] Step {step:>6d}/{self.max_steps} | Loss: {l_val:>6.4f} | CE: {ce_val:>5.3f} | LR: {lr:.2e} | {sps:>5.1f} st/s | {tps:>9,.0f} tok/s"

                if not benchmark:
                    eta_mins = (self.max_steps - step) / sps / 60.0 if sps > 0 else 0.0
                    log_msg += f" | ETA: {eta_mins:>4.1f}m"
                print(log_msg, flush=True)

            if not benchmark and self.is_master and step % self.c_cfg.get("save_every_steps", 1000) == 0:
                if getattr(self, "is_unified", False) and getattr(self, "dual_monitor", None) is not None:
                    probe_res = self.dual_monitor.evaluate(self.model, current_step=step)
                    print(
                        f"  [Dual Probe Step {step:>6d}] Causal Top-1: {probe_res['causal_top1']*100:.1f}% | "
                        f"C-CE: {probe_res['causal_ce']:.3f} | Infill Top-1: {probe_res['infill_top1']*100:.1f}% | "
                        f"I-CE: {probe_res['infill_ce']:.3f} | LRH AUC: {probe_res['lrh_auc']:.3f}"
                    )
                    div_warn = self.dual_monitor.check_divergence()
                    if div_warn:
                        print(f"  {div_warn}")
                ckpt_file = ckpt_dir / f"checkpoint_step_{step}.pt"
                self.save_checkpoint(ckpt_file)
                print(f"  [Checkpoint] Saved weights to {ckpt_file}")

        total_time = time.time() - start_time
        if benchmark:
            bench_elapsed = time.time() - (bench_start_time if bench_start_time else start_time)
            self._print_benchmark_report(bench_steps, bench_elapsed, latencies, bs, grad_accum)
            return

        if self._save_thread is not None and self._save_thread.is_alive():
            self._save_thread.join()

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
