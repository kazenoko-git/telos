"""
COROSred Trainer for MLX (Phases A and B).

Includes:
- Phase A: Frozen AR backbone, training Reliability Head via Ambiguity-Exclusion (K_AMB).
- Phase B: Bidirectional Masked Denoising Loss (MDLM) on masked positions.
- Strict pre-decision indexing rule (h_{i-1} -> y_i) to prevent label leakage.
"""

import numpy as np
import time
import math
from pathlib import Path

try:
    import mlx.core as mx
    import mlx.nn as mx_nn
    import mlx.optimizers as mx_optim
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

from telos.training.core import (
    clip_grad_norm_mlx,
    build_special_token_lut,
    get_sys_mem_str,
    execute_mlx_training_step,
)


def crsr_phase_a_loss_fn(model, batch_seqs, vocab_size, special_token_lut=None, k_amb: int = 5):
    """
    Phase A: Binary Cross Entropy for Reliability Head using Ambiguity Exclusion.
    """
    B, T = batch_seqs.shape

    # Forward pass under causal mask to satisfy the router indexing rule
    logits, raw_r_scores = model(batch_seqs, is_causal=True, return_reliability=True)

    # Shift indices: hidden state h[i] predicts token x[i+1]
    shift_logits = logits[:, :-1, :].astype(mx.float32)
    shift_r_scores = raw_r_scores[:, :-1].astype(mx.float32)
    shift_targets = batch_seqs[:, 1:]

    # Check top-1 match
    argmax_indices = mx.argmax(shift_logits, axis=-1)
    is_exact_match = (argmax_indices == shift_targets)

    # Check if target token is in top-K plausible predictions (Ambiguity Boundary)
    top_k_indices = mx.argpartition(shift_logits, -k_amb, axis=-1)[..., -k_amb:]
    expanded_targets = mx.expand_dims(shift_targets, -1)
    is_target_in_top_k = mx.any(top_k_indices == expanded_targets, axis=-1)

    # Clean Binary Labels: 1.0 if top-1 matches target, 0.0 otherwise
    labels = mx.where(is_exact_match, mx.ones_like(shift_r_scores), mx.zeros_like(shift_r_scores))

    # Numerically stable BCE loss formulation: max(x, 0) - x * y + log(1 + exp(-abs(x)))
    bce_raw = mx.maximum(shift_r_scores, 0) - (shift_r_scores * labels) + mx.log1p(mx.exp(-mx.abs(shift_r_scores)))

    # Drop ambiguous tokens (in top-k, but not top-1) from training loss entirely
    is_ambiguous = mx.logical_and(is_target_in_top_k, mx.logical_not(is_exact_match))
    valid_mask = mx.logical_not(is_ambiguous)

    if special_token_lut is not None:
        content_mask = ~special_token_lut[shift_targets]
        valid_mask = mx.logical_and(valid_mask, content_mask)

    valid_mask_f32 = valid_mask.astype(mx.float32)
    masked_bce = bce_raw * valid_mask_f32

    # Avoid zero-division on heavily masked sequences
    valid_count = mx.clip(mx.sum(valid_mask_f32, axis=1), 1.0, float(T - 1))
    per_example_loss = mx.sum(masked_bce, axis=1) / valid_count

    loss = mx.mean(per_example_loss)
    return loss, loss


def crsr_phase_b_loss_fn(model, batch_seqs, vocab_size, mask_token_id: int, mask_prob: float = 0.15):
    """
    Phase B: Bidirectional Masked Denoising Language Model (MDLM) loss.
    """
    B, T = batch_seqs.shape

    # Sample random binary mask determining positions to corrupt with [MASK]
    rand_probs = mx.random.uniform(shape=(B, T))
    mask_positions = rand_probs < mask_prob

    # Corrupt tokens with mask_token_id
    corrupted_seqs = mx.where(mask_positions, mx.full((B, T), mask_token_id, dtype=batch_seqs.dtype), batch_seqs)

    # Forward pass in Bidirectional Attention Mode
    logits = model(corrupted_seqs, is_causal=False, return_reliability=False)
    logits_f32 = logits.astype(mx.float32).reshape(-1, vocab_size)
    targets_flat = batch_seqs.reshape(-1)

    ce_all = mx_nn.losses.cross_entropy(logits_f32, targets_flat, reduction="none").reshape(B, T)
    masked_ce = ce_all * mask_positions.astype(mx.float32)

    denom = mx.clip(mx.sum(mask_positions.astype(mx.float32), axis=1), 1.0, float(T))
    loss = mx.mean(mx.sum(masked_ce, axis=1) / denom)

    return loss, loss


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


class TelosMLXCOROSredTrainer:
    """Trainer orchestrator for COROSred Phases A and B."""

    def __init__(self, model, cfg):
        if not MLX_AVAILABLE:
            raise ImportError("MLX is not installed. Cannot use TelosMLXCOROSredTrainer.")
        self.model = model
        self.cfg = cfg
        self.m_cfg = cfg.get("model", {})
        self.t_cfg = cfg.get("training", {})
        self.crsr_cfg = cfg.get("crsr", cfg.get("corosred", {}))
        self.c_cfg = cfg.get("checkpoint", {})

        self.special_lut = build_special_token_lut(self.m_cfg.get("vocab_size", 8192))
        self.grad_clip = float(self.t_cfg.get("grad_clip", 1.0))

        if self.t_cfg.get("gradient_checkpointing", False) or self.m_cfg.get("use_grad_checkpoint", False):
            self.model.use_grad_checkpoint = True

    def train(self, resume_step: int = 0):
        k_amb = self.crsr_cfg.get("k_amb", 5)
        phase = self.crsr_cfg.get("phase", "A").upper()

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

        def get_lr(step):
            if step < warmup_steps:
                return max_lr * (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
            return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

        optimizer = mx_optim.AdamW(
            learning_rate=max_lr,
            weight_decay=weight_decay,
            betas=[0.9, 0.95],
            bias_correction=True,
        )

        vocab_size = self.m_cfg["vocab_size"]
        special_lut = self.special_lut

        if phase == "A":
            print(f"  [COROSred] Phase A: Frozen AR Backbone, Training Reliability Head (k_amb={k_amb})")
            loss_and_grad_fn = mx_nn.value_and_grad(self.model, crsr_phase_a_loss_fn)
            # Only reliability head parameters participate in backward pass
            compilation_targets = [self.model.reliability_head.state]

            def microbatch_step_uncompiled(batch_seqs):
                (loss, _metric), grads = loss_and_grad_fn(
                    self.model, batch_seqs, vocab_size,
                    special_token_lut=special_lut, k_amb=k_amb,
                )
                return loss, _metric, grads

        elif phase == "B":
            print("  [COROSred] Phase B: Full Backbone Bidirectional Masked Denoising Training (MDLM)")
            loss_and_grad_fn = mx_nn.value_and_grad(self.model, crsr_phase_b_loss_fn)
            # Whole model state is updated
            compilation_targets = [self.model.state]
            mask_token_id = self.m_cfg.get("mask_token_id", 0)

            def microbatch_step_uncompiled(batch_seqs):
                (loss, _metric), grads = loss_and_grad_fn(
                    self.model, batch_seqs, vocab_size, mask_token_id=mask_token_id,
                )
                return loss, _metric, grads
        else:
            raise ValueError(f"Unknown COROSred training phase: {phase}")

        # Warmup trace to compile execution graph
        dummy_seqs = mx.random.randint(0, self.m_cfg["vocab_size"], (self.t_cfg["batch_size"], self.m_cfg["seq_len"]))
        d_loss, d_ce, d_grads = microbatch_step_uncompiled(dummy_seqs)
        mx.eval(d_loss, d_ce, d_grads)
        del d_loss, d_ce, d_grads

        microbatch_step = mx.compile(microbatch_step_uncompiled, inputs=compilation_targets, outputs=compilation_targets)

        ckpt_dir = Path(self.c_cfg.get("checkpoint_dir", f"checkpoints/corosred/phase_{phase.lower()}"))
        ckpt_dir.mkdir(parents=True, exist_ok=True)

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
                is_first_step=(step == resume_step + 1),
            )

            if step % 50 == 0 or step == 1 or step == max_steps:
                avg_loss_val = accum_loss.item() / grad_accum
                elapsed = time.time() - start_time
                steps_taken = step - resume_step
                sps = steps_taken / elapsed if elapsed > 0 else 0
                tps = sps * bs * grad_accum * self.m_cfg["seq_len"]
                mem_str = get_sys_mem_str()

                print(
                    f"  [COROSred Phase {phase}] Step {step:>6d}/{max_steps} | Loss: {avg_loss_val:>6.4f} | "
                    f"LR: {lr:.2e} | {sps:>5.1f} st/s | {tps:>9,.0f} tok/s | {mem_str}",
                    flush=True,
                )

            if step % self.c_cfg.get("save_every_steps", 1000) == 0:
                ckpt_file = ckpt_dir / f"checkpoint_step_{step}.safetensors"
                self.model.save_weights(str(ckpt_file))
                print(f"  [Checkpoint] Saved weights to {ckpt_file}")

        final_weights = ckpt_dir / "model.safetensors"
        self.model.save_weights(str(final_weights))
        print(f"  COROSred Phase {phase} Completed. Saved artifacts to {ckpt_dir}/")


# Alias for compatibility with trainer harnesses
TelosMLXTrainer = TelosMLXCOROSredTrainer
