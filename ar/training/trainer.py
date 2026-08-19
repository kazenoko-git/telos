"""
Trainer for télos Autoregressive (AR) Language Model Baseline.

Includes:
- MLX-native compiled training loop with causal next-token cross entropy loss
- bfloat16 AdamW optimizer moment casting for 50% RAM savings
- Automatic gradient accumulation and checkpoint management
- Parity with MDLM and UNDLM training pipelines
"""

import numpy as np
import time
import math
from pathlib import Path

try:
    import mlx.core as mx
    import mlx.nn as mx_nn
    import mlx.optimizers as mx_optim
    from mlx.utils import tree_map
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


def ar_loss_fn_mlx(model, batch_seqs, vocab_size):
    """Computes next-token cross-entropy loss for Autoregressive models.

    Inputs:  x = [t0, t1, t2, ..., tN-1]
    Logits:  predicts next token [t1, t2, ..., tN]
    """
    logits = model(batch_seqs)  # [B, T, V]
    B, T, V = logits.shape

    shift_logits = logits[:, :-1, :].reshape(-1, V)   # [B*(T-1), V]
    shift_targets = batch_seqs[:, 1:].reshape(-1)      # [B*(T-1)]

    ce_per_token = mx_nn.losses.cross_entropy(shift_logits, shift_targets, reduction="none").reshape(B, T - 1)
    loss = mx.mean(ce_per_token)
    return loss, loss  # Return (loss, ce) — identical for AR baseline


def get_sys_mem_str() -> str:
    """Returns Apple Silicon Metal unified memory and swap usage string."""
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


def get_global_targets_contiguous(dataset_matrix, idx_ptr, total_batch, seq_len):
    """Fetches contiguous batches from the memory-mapped token array with wraparound."""
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


class TelosMLXARTrainer:
    """MLX-native Trainer orchestrator for Autoregressive (AR) Causal Models."""

    def __init__(self, model, cfg):
        if not MLX_AVAILABLE:
            raise ImportError("MLX is not installed. Cannot use TelosMLXARTrainer.")
        self.model = model
        self.cfg = cfg
        self.m_cfg = cfg["model"]
        self.t_cfg = cfg["training"]
        self.c_cfg = cfg.get("checkpoint", {})

        if self.t_cfg.get("gradient_checkpointing", False) or self.m_cfg.get("use_grad_checkpoint", False):
            self.model.use_grad_checkpoint = True
            print("  [Memory] Gradient Checkpointing Enabled.")

    def train(self, resume_step: int = 0):
        train_bin = Path("data/python_corpus_mac.bin")
        if train_bin.exists():
            print(f"  Loading pre-tokenized dataset from {train_bin}...")
            raw_data = np.memmap(train_bin, dtype=np.int32, mode="r")
            n_seqs = len(raw_data) // self.m_cfg["seq_len"]
            dataset_matrix = raw_data[:n_seqs * self.m_cfg["seq_len"]].reshape(n_seqs, self.m_cfg["seq_len"])
        else:
            print("  Notice: Pre-tokenized dataset file not found. Generating synthetic stream...")
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
            seqs_consumed = resume_step * (bs * grad_accum)
            idx_ptr = seqs_consumed % len(dataset_matrix)

        def get_lr(step):
            if step < warmup_steps:
                return max_lr * (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
            return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

        optimizer = mx_optim.AdamW(learning_rate=max_lr, weight_decay=weight_decay)

        loss_and_grad_fn = mx_nn.value_and_grad(self.model, ar_loss_fn_mlx)
        vocab_size = self.m_cfg["vocab_size"]

        def microbatch_step_uncompiled(batch_seqs):
            (loss, ce), grads = loss_and_grad_fn(self.model, batch_seqs, vocab_size)
            return loss, ce, grads

        # Graph trace warmup to compile kernel without polluting AdamW state
        dummy_seqs = mx.random.randint(0, self.m_cfg["vocab_size"], (self.t_cfg["batch_size"], self.m_cfg["seq_len"]))
        dummy_loss, dummy_ce, dummy_grads = microbatch_step_uncompiled(dummy_seqs)
        mx.eval(dummy_loss, dummy_ce, dummy_grads)
        del dummy_loss, dummy_ce, dummy_grads

        state = [self.model.state]
        microbatch_step = mx.compile(microbatch_step_uncompiled, inputs=state, outputs=state)

        base_dir_str = self.c_cfg.get("checkpoint_dir", self.c_cfg.get("dir", "checkpoints/ar/25m/kappa_25m_1to35_mlx"))
        ckpt_dir = Path(base_dir_str)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Checkpoint Directory: {ckpt_dir} (AR Canonical)")

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

                mx.eval(accum_grads, accum_loss, accum_ce)

            accum_grads = tree_map(lambda g: g / grad_accum, accum_grads)
            optimizer.update(self.model, accum_grads)

            if step == resume_step + 1:
                optimizer.state = cast_optimizer_moments_bf16(optimizer.state)

            mx.eval(self.model.parameters(), optimizer.state)

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

                log_msg = f"  [AR Baseline] Step {step:>6d}/{max_steps} | Loss: {avg_loss_val:>6.2f} | CE: {avg_ce_val:>5.3f} | LR: {lr:.2e} | {sps:>5.1f} st/s | {tps:>9,.0f} tok/s | {mem_str} | ETA: {eta_mins:>4.1f}m"
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

        tok_source = Path("configs/shared/tokenizer_mac.json")
        if tok_source.exists():
            import shutil
            shutil.copy(tok_source, ckpt_dir / "tokenizer.json")

        print("=" * 70)
        print(f"  AR Baseline Training Complete! Total time: {total_time/60.0:.2f} minutes.")
        print(f"  Saved standalone model artifact to {ckpt_dir}/")
        print("=" * 70)

# Alias for compatibility
TelosMLXTrainer = TelosMLXARTrainer
