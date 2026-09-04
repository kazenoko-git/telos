"""
Unified MLX Trainer for all Telos paradigms (AR, MDLM, UNDLM, COROSred).
"""

import math
import time
import json
import numpy as np
from pathlib import Path

try:
    import mlx.core as mx
    import mlx.nn as mx_nn
    import mlx.optimizers as mx_optim
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

from .core import (
    clip_grad_norm_mlx,
    build_special_token_lut,
    get_sys_mem_str,
    execute_mlx_training_step,
)
from .dataloader import get_global_targets_contiguous_mlx

from telos.diffusion.ar import ar_loss_fn_mlx
from telos.diffusion.mdlm import mdlm_loss_mlx, apply_masking_mlx, sample_beta_timesteps
from telos.diffusion.undlm import undlm_loss_mlx, apply_uniform_noise_mlx
from telos.diffusion.corosred import crsr_phase_a_loss_fn_mlx, crsr_phase_b_loss_fn_mlx


from .hardware import detect_apple_silicon_profile


class UnifiedMLXTrainer:
    """Unified MLX Trainer orchestrator for all paradigms."""

    def __init__(self, paradigm: str, model, cfg: dict, eval_policy: str = "auto"):
        if not MLX_AVAILABLE:
            raise ImportError("MLX is not installed. Cannot use UnifiedMLXTrainer.")
        
        self.paradigm = paradigm.lower()
        self.model = model
        self.cfg = cfg
        self.m_cfg = cfg.get("model", {})
        self.t_cfg = cfg.get("training", {})
        self.c_cfg = cfg.get("checkpoint", {})
        
        # Hardware Profile & Memory Management
        self.hw_profile = detect_apple_silicon_profile(user_policy=eval_policy)
        print(f"  [Hardware] Detected {self.hw_profile.device_name} ({self.hw_profile.total_memory_gb:.1f}GB Unified Memory)")
        print(f"  [Hardware] Evaluation Policy: {self.hw_profile.eval_policy.upper()}")

        self.vocab_size = self.m_cfg.get("vocab_size", 8192)
        self.seq_len = self.m_cfg.get("seq_len", 512)
        self.special_lut = build_special_token_lut(self.vocab_size)
        self.grad_clip = float(self.t_cfg.get("grad_clip", 1.0))

        if self.t_cfg.get("gradient_checkpointing", False) or self.m_cfg.get("use_grad_checkpoint", False):
            self.model.use_grad_checkpoint = True
            print("  [Memory] Gradient Checkpointing Enabled.")

        if self.paradigm == "corosred":
            self.crsr_cfg = cfg.get("crsr", cfg.get("corosred", {}))
            self.phase = self.crsr_cfg.get("phase", "A").upper()

    def _get_lr(self, step, warmup_steps, max_steps, max_lr, min_lr):
        if step <= warmup_steps:
            return max_lr * step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

    def _setup_training_functions(self):
        """Returns the appropriate loss_and_grad_fn and compilation_targets based on paradigm."""
        special_lut = self.special_lut
        vocab_size = self.vocab_size

        if self.paradigm == "ar":
            loss_and_grad_fn = mx_nn.value_and_grad(self.model, ar_loss_fn_mlx)
            compilation_targets = [self.model.state]
            
            def microbatch_step_uncompiled(batch_seqs):
                (loss, ce), grads = loss_and_grad_fn(self.model, batch_seqs, vocab_size, special_token_lut=special_lut)
                return loss, ce, grads
                
        elif self.paradigm == "mdlm":
            loss_and_grad_fn = mx_nn.value_and_grad(self.model, mdlm_loss_mlx)
            compilation_targets = [self.model.state]
            
            def microbatch_step_uncompiled(batch_seqs, t_vals):
                masked_ids, mask_pos, t_vals_out = apply_masking_mlx(batch_seqs, t_vals, mask_token_id=1, special_token_lut=special_lut)
                (loss, ce), grads = loss_and_grad_fn(self.model, masked_ids, batch_seqs, mask_pos, t_vals_out, vocab_size)
                return loss, ce, grads

        elif self.paradigm == "undlm":
            loss_and_grad_fn = mx_nn.value_and_grad(self.model, undlm_loss_mlx)
            compilation_targets = [self.model.state]
            
            def microbatch_step_uncompiled(batch_seqs, t_vals):
                noisy_ids, corrupt_mask, t_vals_out = apply_uniform_noise_mlx(batch_seqs, t_vals, vocab_size, special_token_lut=special_lut)
                (loss, ce), grads = loss_and_grad_fn(self.model, noisy_ids, batch_seqs, t_vals_out, vocab_size, special_token_lut=special_lut)
                return loss, ce, grads

        elif self.paradigm == "corosred":
            if self.phase == "A":
                k_amb = self.crsr_cfg.get("k_amb", 5)
                loss_and_grad_fn = mx_nn.value_and_grad(self.model, crsr_phase_a_loss_fn_mlx)
                compilation_targets = [self.model.state]
                
                def microbatch_step_uncompiled(batch_seqs):
                    (loss, ce), grads = loss_and_grad_fn(self.model, batch_seqs, vocab_size, special_token_lut=special_lut, k_amb=k_amb)
                    return loss, ce, grads
            else:
                mask_token_id = self.m_cfg.get("mask_token_id", 0)
                loss_and_grad_fn = mx_nn.value_and_grad(self.model, crsr_phase_b_loss_fn_mlx)
                compilation_targets = [self.model.state]
                
                def microbatch_step_uncompiled(batch_seqs):
                    (loss, ce), grads = loss_and_grad_fn(self.model, batch_seqs, vocab_size, mask_token_id=mask_token_id)
                    return loss, ce, grads
        else:
            raise ValueError(f"Unknown paradigm: {self.paradigm}")

        return microbatch_step_uncompiled, compilation_targets

    def _warmup_trace(self, step_fn, bs):
        """Runs a dummy forward pass to compile the MLX graph."""
        dummy_seqs = mx.random.randint(0, self.vocab_size, (bs, self.seq_len))
        if self.paradigm in ["mdlm", "undlm"]:
            dummy_t_vals = mx.clip(mx.array(np.random.beta(1.5, 1.5, size=(bs, 1)).astype(np.float32)), 1e-5, 1.0)
            d_loss, d_ce, d_grads = step_fn(dummy_seqs, dummy_t_vals)
        else:
            d_loss, d_ce, d_grads = step_fn(dummy_seqs)
        mx.eval(d_loss, d_ce, d_grads)

    def _print_benchmark_report(self, steps: int, elapsed: float, latencies: list[float], bs: int, grad_accum: int):
        """Prints a publication-style benchmark report table and saves results to JSON."""
        tokens_processed = steps * bs * grad_accum * self.seq_len
        sps = steps / elapsed if elapsed > 0 else 0.0
        tps = tokens_processed / elapsed if elapsed > 0 else 0.0
        mean_lat = np.mean(latencies) if latencies else 0.0
        p50_lat = np.median(latencies) if latencies else 0.0
        p95_lat = np.percentile(latencies, 95) if latencies else 0.0

        print("\n" + "=" * 76)
        print("  TELOS UNIFIED BENCHMARK REPORT (Apple Silicon / MLX)")
        print("=" * 76)
        print(f"  Paradigm:             {self.paradigm.upper()}")
        print(f"  Hardware:             {self.hw_profile.device_name}")
        print(f"  Unified Memory:       {self.hw_profile.total_memory_gb:.1f} GB")
        print(f"  Evaluation Policy:    {self.hw_profile.eval_policy.upper()}")
        print(f"  Batch Config:         batch_size={bs}, grad_accum={grad_accum}, seq_len={self.seq_len}")
        print(f"  Total Effective Batch: {bs * grad_accum} sequences ({bs * grad_accum * self.seq_len:,} tokens/step)")
        print("-" * 76)
        print(f"  Benchmark Duration:   {elapsed:.2f} seconds (limit: 300.0s / 5.0m)")
        print(f"  Steps Completed:      {steps:,}")
        print(f"  Tokens Processed:     {tokens_processed:,}")
        print(f"  Throughput:           {sps:.2f} steps/s  |  {tps:,.1f} tokens/s")
        print(f"  Latency per Step:     Mean: {mean_lat:.1f} ms  |  p50: {p50_lat:.1f} ms  |  p95: {p95_lat:.1f} ms")
        print(f"  Memory Usage:         {get_sys_mem_str()}")
        print("=" * 76 + "\n")

        # Save benchmark artifact to logs
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        bench_payload = {
            "paradigm": self.paradigm,
            "backend": "mlx",
            "device": self.hw_profile.device_name,
            "memory_gb": self.hw_profile.total_memory_gb,
            "eval_policy": self.hw_profile.eval_policy,
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
        report_file = log_dir / f"benchmark_{self.paradigm}_mlx_{int(time.time())}.json"
        with open(report_file, "w") as f:
            json.dump(bench_payload, f, indent=2)
        print(f"  Saved benchmark metrics to {report_file}\n")

    def train(self, resume_step: int = 0, benchmark: bool = False, benchmark_duration: float = 300.0):
        d_cfg = self.cfg.get("data", {})
        train_path = d_cfg.get("train_path", d_cfg.get("dataset_path", d_cfg.get("path", None)))
        use_synthetic = d_cfg.get("synthetic", False)

        train_bin = Path(train_path) if train_path else Path("data/python_corpus_mac.bin")
        if not train_bin.exists() and train_path is None:
            train_bin = Path("data/python_corpus.bin")
        if not train_bin.exists() and train_path is None:
            train_bin = Path("data/python_corpus_2.5b.bin")

        if train_bin.exists() and not use_synthetic and (self.vocab_size >= 1024 or train_path is not None):
            print(f"  Loading pre-tokenized dataset from {train_bin}...")
            dtype = np.int32 if "mac" in str(train_bin) else np.uint16
            raw_data = np.memmap(train_bin, dtype=dtype, mode="r")
            n_seqs = len(raw_data) // self.seq_len
            dataset_matrix = raw_data[:n_seqs * self.seq_len].reshape(n_seqs, self.seq_len)
        else:
            print("  Notice: Using synthetic dataset stream...")
            dataset_matrix = np.random.randint(0, self.vocab_size, (10000, self.seq_len), dtype=np.uint16)

        max_steps = int(self.t_cfg.get("max_steps", 5000))
        warmup_steps = int(self.t_cfg.get("warmup_steps", 100))
        max_lr = float(self.t_cfg.get("max_lr", 3e-4))
        min_lr = float(self.t_cfg.get("min_lr", 3e-5))
        weight_decay = float(self.t_cfg.get("weight_decay", 0.1))
        bs = int(self.t_cfg.get("batch_size", 16))
        grad_accum = int(self.t_cfg.get("gradient_accumulation", 1))

        # Benchmark duration strictly capped at 300.0 seconds (5 minutes)
        max_bench_duration = min(float(benchmark_duration), 300.0)

        idx_ptr = 0
        if resume_step > 0:
            seqs_consumed = resume_step * (bs * grad_accum)
            idx_ptr = seqs_consumed % len(dataset_matrix)

        optimizer = mx_optim.AdamW(
            learning_rate=max_lr,
            weight_decay=weight_decay,
            betas=[0.9, 0.95],
            bias_correction=True
        )

        uncompiled_step_fn, compilation_targets = self._setup_training_functions()
        self._warmup_trace(uncompiled_step_fn, bs)
        compiled_step = mx.compile(uncompiled_step_fn, inputs=compilation_targets, outputs=compilation_targets)

        ckpt_dir_name = f"checkpoints/{self.paradigm}"
        if self.paradigm == "corosred":
            ckpt_dir_name += f"/phase_{self.phase.lower()}"
        ckpt_dir = Path(self.c_cfg.get("checkpoint_dir", self.c_cfg.get("dir", ckpt_dir_name)))
        if not benchmark:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            print(f"  Checkpoint Directory: {ckpt_dir} (Paradigm: {self.paradigm.upper()})")
        else:
            print(f"  [Benchmark Mode] Starting throughput benchmark (Maximum limit: {max_bench_duration:.0f}s)...")

        # Dynamic memory evaluation policy:
        # Low RAM (<24GB): eager microbatch eval
        # High RAM (>=24GB): evaluate only at step boundaries
        eval_micro = (self.hw_profile.eval_policy == "eager")

        start_time = time.time()
        bench_start_time = None
        latencies = []
        bench_steps = 0

        for step in range(resume_step + 1, max_steps + 1):
            t_step_start = time.perf_counter()

            lr = self._get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
            optimizer.learning_rate = lr

            global_targets, idx_ptr = get_global_targets_contiguous_mlx(dataset_matrix, idx_ptr, bs * grad_accum, self.seq_len)

            def batch_gen():
                for i in range(grad_accum):
                    batch_seqs = global_targets[i * bs : (i + 1) * bs]
                    if self.paradigm in ["mdlm", "undlm"]:
                        t_vals = mx.clip(mx.array(np.random.beta(1.5, 1.5, size=(bs, 1)).astype(np.float32)), 1e-5, 1.0)
                        yield (batch_seqs, t_vals)
                    else:
                        yield batch_seqs

            accum_loss, accum_ce = execute_mlx_training_step(
                model=self.model,
                optimizer=optimizer,
                compiled_step_fn=compiled_step,
                batch_iterator=batch_gen(),
                grad_accum=grad_accum,
                grad_clip=self.grad_clip,
                is_first_step=(step == resume_step + 1),
                eval_every_microbatch=eval_micro
            )

            step_time_ms = (time.perf_counter() - t_step_start) * 1000.0

            # For benchmark mode: 5 warmup steps before collecting benchmark timers
            if benchmark:
                if step == resume_step + 5:
                    bench_start_time = time.time()
                elif bench_start_time is not None:
                    latencies.append(step_time_ms)
                    bench_steps += 1
                    # Check benchmark duration condition
                    if (time.time() - bench_start_time) >= max_bench_duration:
                        bench_elapsed = time.time() - bench_start_time
                        self._print_benchmark_report(bench_steps, bench_elapsed, latencies, bs, grad_accum)
                        return

            if step % 200 == 0:
                mx.clear_cache()

            if step % 50 == 0 or step == 1 or step == max_steps or (benchmark and step % 10 == 0):
                avg_loss_val = accum_loss.item() / grad_accum
                avg_ce_val = accum_ce.item() / grad_accum
                elapsed = time.time() - start_time
                steps_taken = step - resume_step
                sps = steps_taken / elapsed if elapsed > 0 else 0
                tps = sps * bs * grad_accum * self.seq_len
                
                eta_mins = (max_steps - step) / sps / 60.0 if sps > 0 else 0.0
                mem_str = get_sys_mem_str()

                log_msg = f"  [{self.paradigm.upper()}] Step {step:>6d}/{max_steps} | Loss: {avg_loss_val:>6.4f} | CE: {avg_ce_val:>5.3f} | LR: {lr:.2e} | {sps:>5.1f} st/s | {tps:>9,.0f} tok/s | {mem_str}"
                if not benchmark:
                    log_msg += f" | ETA: {eta_mins:>4.1f}m"
                print(log_msg, flush=True)

            if not benchmark and step % self.c_cfg.get("save_every_steps", 1000) == 0:
                ckpt_file = ckpt_dir / f"checkpoint_step_{step}.safetensors"
                self.model.save_weights(str(ckpt_file))
                print(f"  [Checkpoint] Saved weights to {ckpt_file}")

        total_time = time.time() - start_time
        if benchmark:
            bench_elapsed = time.time() - (bench_start_time if bench_start_time else start_time)
            self._print_benchmark_report(bench_steps, bench_elapsed, latencies, bs, grad_accum)
            return

        final_weights = ckpt_dir / "model.safetensors"
        self.model.save_weights(str(final_weights))

        with open(ckpt_dir / "config.json", "w") as f:
            json.dump(self.m_cfg, f, indent=2)

        print("=" * 70)
        print(f"  {self.paradigm.upper()} Training Complete! Total time: {total_time/60.0:.2f} minutes.")
        print(f"  Saved standalone model artifact to {ckpt_dir}/")
        print("=" * 70)
