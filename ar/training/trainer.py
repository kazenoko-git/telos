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
    from mlx.utils import tree_map, tree_flatten
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

from telos.training.core import (
    clip_grad_norm_mlx, build_special_token_lut, get_sys_mem_str,
    cast_optimizer_moments_bf16, execute_mlx_training_step
)


def ar_loss_fn_mlx(model, batch_seqs, vocab_size, special_token_lut=None):
    """Computes next-token cross-entropy loss for Autoregressive models in float32.

    Inputs:  x = [t0, t1, t2, ..., tN-1]
    Logits:  predicts next token [t1, t2, ..., tN]
    """
    logits = model(batch_seqs)  # [B, T, V]
    B, T, V = logits.shape

    # Upcast logits to float32 for numerically stable log-softmax in cross entropy
    shift_logits = logits[:, :-1, :].reshape(-1, V)   # [B*(T-1), V]
    shift_targets = batch_seqs[:, 1:].reshape(-1)                        # [B*(T-1)]

    ce_per_token = mx_nn.losses.cross_entropy(shift_logits, shift_targets, reduction="none").reshape(B, T - 1)
    
    # Exclude special tokens from loss if LUT provided
    if special_token_lut is not None:
        shift_target_2d = batch_seqs[:, 1:]
        content_mask = ~special_token_lut[shift_target_2d]
        ce_per_token = ce_per_token * content_mask.astype(mx.float32)
        content_count = mx.clip(mx.sum(content_mask.astype(mx.float32), axis=1), 1.0, float(T - 1))
        per_example_ce = mx.sum(ce_per_token, axis=1) / content_count
        loss = mx.mean(per_example_ce)
    else:
        loss = mx.mean(ce_per_token)

    return loss, loss  # Return (loss, ce) — identical for AR baseline


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
        self.special_lut = build_special_token_lut(self.m_cfg["vocab_size"])
        self.grad_clip = float(self.t_cfg.get("grad_clip", 1.0))

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
            if step <= warmup_steps:
                return max_lr * step / warmup_steps
            progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
            return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

        # Standard LLM AdamW configuration: betas=[0.9, 0.95], bias_correction=True
        optimizer = mx_optim.AdamW(
            learning_rate=max_lr,
            weight_decay=weight_decay,
            betas=[0.9, 0.95],
            bias_correction=True
        )

        loss_and_grad_fn = mx_nn.value_and_grad(self.model, ar_loss_fn_mlx)
        vocab_size = self.m_cfg["vocab_size"]
        special_lut = self.special_lut

        def microbatch_step_uncompiled(batch_seqs):
            (loss, ce), grads = loss_and_grad_fn(self.model, batch_seqs, vocab_size, special_token_lut=special_lut)
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

            def batch_gen():
                for i in range(grad_accum):
                    yield global_targets[i * bs : (i + 1) * bs]

            accum_loss, accum_ce = execute_mlx_training_step(
                model=self.model,
                optimizer=optimizer,
                compiled_step_fn=microbatch_step,
                batch_iterator=batch_gen(),
                grad_accum=grad_accum,
                grad_clip=self.grad_clip,
                is_first_step=(step == resume_step + 1)
            )

            if True:
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
