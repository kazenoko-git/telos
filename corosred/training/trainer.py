# Trainer for COROSred

import time, math
import numpy as np
from pathlib import Path

try:
    import mlx.core as mx
    import mlx.nn as mx_nn
    import mlx.optimizers as mx_optim
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

from telos.training.core import (
    clip_grad_norm_mlx, build_special_token_lut, get_sys_mem_str, cast_optimizer_moments_bf16, execute_mlx_training_step
)

def A_loss_fn(model, batch_seqs, vocab_size, special_token_lut=None, k_amb=5):
    """Compute Binary CE for the Reliability Head"""

    B, T = batch_seqs.shape

    # forward pass explicitly in casual mode, extracting r scores (router indexing rule)
    logits, raw_r_scores = model(batch_seqs, is_casual=True, return_reliability=True)

    # shift representations to align pre-decision state i with target i+1
    shift_logits = logits[:, :-1, :].astype(mx.float32)
    shift_r_scores = raw_r_scores[:, :-1].astype(mx.float32)
    shift_targets = batch_seqs[:, 1:]

    # find top 1 prediction (exact match)
    argmax_indices = mx.argmax(shift_logits, axis=-1)
    is_exact_match = (argmax_indices == shift_targets)

    # find if target sits within top k ambiguity threshold
    top_k_indices = mx.argpartition(shift_logits, -k_amb, axis=-1)[..., -k_amb:]
    expanded_targets = mx.expand_dims(shift_targets, -1)
    is_target_in_top_k = mx.any(top_k_indices == expanded_targets, axis=-1)

    labels = mx.where(is_exact_match, mx.ones_like(shift_r_scores), mx.zeros_like(shift_r_scores)) # 1 if model top pick matches GT, 0 if it doesn't

    bce_raw = mx.maximum(shift_r_scores, 0) - (shift_r_scores*labels) + mx.log1p(mx.exp(-mx.abs(shift_r_scores)))
    # numerically stable BCEWithLogits: max(x, 0) - x*y + log(1 + exp(-abs(x)))

    # ambiguity exclusion mask (i.e, exclude positions that are in top K, but not top1)
    is_ambiguous = mx.logical_and(is_target_in_top_k, is_exact_match, mx.logical_not(is_exact_match))
    valid_mask = mx.logical_and(is_ambiguous)

    # optional: exclude special tokens (PAD/EOS) matching logic in core AR base
    if special_token_lut is not None:
        content_mask = ~special_token_lut[shift_targets]
        valid_mask = mx.logical_and(valid_mask, content_mask)

    valid_mask_f32 = valid_mask.astype(mx.float32)
    masked_bce = bce_raw * valid_mask_f32

    # provide safe denom using mx.clip to avoid zero-division collapse
    valid_count = mx.clip(mx.sum(valid_mask_f32, axis=1), 1.0, float(T-1))
    per_example_loss = mx.sum(masked_bce, axis=1) / valid_count

    loss = mx.mean(per_example_loss)

    return loss, loss # return empty secondary metric for tuple

def get_global_targets_contiguous(dataset_matrix, idx_ptr, total_batch, seq_len):
    """ Fetch contiguous batches from memory-mapped token array with wraparound. """
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
    def __init__(self, model, cfg):
        if not MLX_AVAILABLE:
            raise ImportError("MLX is not installed. Cannot use TelosMLXCOROSredTrainer.")
        self.model = model
        self.cfg = cfg
        self.m_cfg = cfg.get("model", {})
        self.t_cfg = cfg.get("training", {})
        self.COROSred_cfg = cfg.get("corosred", {}) # pull k_amb, phase
        self.c_cfg = cfg.get("checkpoint", {})

        self.special_lut = build_special_token_lut(self.m_cfg.get("vocab_size", 8192))
        self.grad_clip = float(self.t_cfg.get("grad_clip", 1.0))

        if self.t_cfg.get("gradient_checkpointing", False) or self.m_cfg.get("use_grad_checkpoint", False):
            self.model.use_grad_checkpoint = True
            print("  [Memory] Gradient Checkpointing Enabled.")

        def train(self, resume_step: int = 0):
            # extract specific COROSred configs
            k_amb = self.COROSred_cfg.get("k_amb", 5)
            phase = self.COROSred_cfg.get("phase", "A")

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

            # Standard LLM AdamW configuration: betas=[0.9, 0.95], bias_correction=True
            optimizer = mx_optim.AdamW(
                learning_rate=max_lr,
                weight_decay=weight_decay,
                betas=[0.9, 0.95],
                bias_correction=True
            )

            vocab_size = self.m_cfg["vocab_size"]
            special_lut = self.special_lut

            if phase == "A":
                print(f"  [COROSred] Operating in Phase A (Frozen AR backbone, training head with k_amb = {k_amb})")
                loss_and_grad_fn = mx_nn.value_and_grad(self.model, A_loss_fn)

                # explicitly scope MLX graph inputs to only head's parameters to truly freeze the backbone
                compilation_targets = [self.model.reliability_head.state]

            else:
                # placeholder for phase B / C which would train the whole fucking backbone against MDLM loss somehow
                raise NotImplementedError(f"COROSred Phase {phase} trainer not entirely implemented yet.")

            def microbatch_step_uncompiled(batch_seqs):
                (loss, _ce_metric), grads = loss_and_grad_fn(self.model, batch_seqs, vocab_size, special_token_lut=special_lut, k_amb=k_amb)
                return loss, _ce_metric, grads

            # Graph trace warmup to compile kernel without polluting AdamW state
            dummy_seqs = mx.random.randint(0, self.m_cfg["vocab_size"],
                                           (self.t_cfg["batch_size"], self.m_cfg["seq_len"]))
            dummy_loss, dummy_ce, dummy_grads = microbatch_step_uncompiled(dummy_seqs)
            mx.eval(dummy_loss, dummy_ce, dummy_grads)
            del dummy_loss, dummy_ce, dummy_grads

            # Compiling over explicitly constrained states freezes unlisted sub-modules instantly
            microbatch_step = mx.compile(microbatch_step_uncompiled, inputs=compilation_targets,
                                         outputs=compilation_targets)

            base_dir_str = self.c_cfg.get("checkpoint_dir", f"checkpoints/crsr/phase_{phase.lower()}_head")
            ckpt_dir = Path(base_dir_str)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            print(f"  Checkpoint Directory: {ckpt_dir}")

            start_time = time.time()

            for step in range(resume_step + 1, max_steps + 1):
                lr = get_lr(step)
                optimizer.learning_rate = lr

                global_targets, idx_ptr = get_global_targets_contiguous(dataset_matrix, idx_ptr, bs * grad_accum,
                                                                        self.m_cfg["seq_len"])

                def batch_gen():
                    for i in range(grad_accum):
                        yield global_targets[i * bs: (i + 1) * bs]

                accum_loss, accum_fake_ce = execute_mlx_training_step(
                    model=self.model,
                    optimizer=optimizer,
                    compiled_step_fn=microbatch_step,
                    batch_iterator=batch_gen(),
                    grad_accum=grad_accum,
                    grad_clip=self.grad_clip,
                    is_first_step=(step == resume_step + 1)
                )

                if step % 100 == 0:
                    mx.clear_cache()
                    import gc
                    gc.collect()

                if step % 50 == 0 or step == 1 or step == max_steps:
                    avg_bce_loss = accum_loss.item() / grad_accum

                    elapsed = time.time() - start_time
                    steps_taken = step - resume_step
                    sps = steps_taken / elapsed if elapsed > 0 else 0
                    tps = sps * bs * grad_accum * self.m_cfg["seq_len"]

                    eta_mins = (max_steps - step) / sps / 60.0 if sps > 0 else 0.0
                    mem_str = get_sys_mem_str()

                    log_msg = f"  [CRSR Phase {phase}] Step {step:>6d}/{max_steps} | Head BCE: {avg_bce_loss:>6.3f} | LR: {lr:.2e} | {sps:>5.1f} st/s | {tps:>9,.0f} tok/s | {mem_str} | ETA: {eta_mins:>4.1f}m"
                    print(log_msg, flush=True)
                    try:
                        Path("logs").mkdir(exist_ok=True)
                        with open("logs/crsr_training.log", "a") as f_log:
                            f_log.write(log_msg + "\n")
                    except Exception:
                        pass

                if step % self.c_cfg.get("save_every_steps", 1000) == 0:
                    ckpt_file = ckpt_dir / f"checkpoint_step_{step}.safetensors"
                    # For Phase A, it's safer to just save the `reliability_head` out to keep weights extremely small.
                    # Here we save entire model, but you can slice logic specific to Phase A.
                    self.model.save_weights(str(ckpt_file))
                    print(f"  [Checkpoint] Saved weights to {ckpt_file}")

            total_time = time.time() - start_time
            final_weights = ckpt_dir / "model.safetensors"
            self.model.save_weights(str(final_weights))

            import json
            with open(ckpt_dir / "config.json", "w") as f:
                json.dump(self.cfg, f, indent=2)

            print("="*70)
            print(f"  CRSR Phase {phase} Training Complete! Total time: {total_time/60.0:.2f} minutes.")
            print(f"  Saved standalone model artifact to {ckpt_dir}/")
            print("="*70)

TelosMLXTrainer = TelosMLXCOROSredTrainer