#!/usr/bin/env python3
"""Train a 100M-token Learned Reliability Head (LRH) on 2x Nvidia T4 (FP16 Autocast).

Architecture:
- Frozen Backbone: Qwen/Qwen2.5-Coder-0.5B-Instruct (eval mode, zero grad).
- Learned Reliability Head (LRH): Linear(896, 896) -> SiLU -> Linear(896, 1) (~805K params).
- Training Target: Predicts token-level causal correctness with Ambiguity Exclusion (k=5).
- Hardware: Optimized for Kaggle 2x Nvidia T4 (mixed-precision FP16 via torch.cuda.amp).
- Telemetry: Real-time tracking of Balanced Accuracy, ROC-AUC, and gamma gate readiness (AUC >= 0.55).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file


def compute_vectorized_roc_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Computes Wilcoxon-Mann-Whitney ROC-AUC efficiently using PyTorch rank-sum."""
    with torch.no_grad():
        s = scores.reshape(-1).float()
        y = labels.reshape(-1).float()

        n_pos = torch.sum(y == 1.0).item()
        n_neg = torch.sum(y == 0.0).item()

        if n_pos == 0 or n_neg == 0:
            return 0.50

        ranks = torch.argsort(torch.argsort(s)) + 1
        r1 = torch.sum(ranks[y == 1.0]).float().item()

        u1 = r1 - (n_pos * (n_pos + 1.0)) / 2.0
        auc = u1 / (n_pos * n_neg)
        return float(max(0.0, min(1.0, auc)))


class LearnedReliabilityHead(nn.Module):
    """Scalar Learned Reliability Head (LRH) for token correctness prediction."""

    def __init__(self, d_model: int = 896):
        super().__init__()
        self.d_model = d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, T, d_model] -> returns [B, T] continuous log-odds
        return self.net(h).squeeze(-1)


def build_data_stream(
    tokenizer,
    dataset_name: str,
    seq_len: int = 512,
    batch_size: int = 16,
    max_tokens: int = 100_000_000,
    device: str = "cuda",
) -> Iterator[torch.Tensor]:
    """Streams and packs code tokens into fixed-length batches."""
    from datasets import load_dataset

    print(f"Streaming code dataset '{dataset_name}' from Hugging Face...")
    ds = load_dataset(dataset_name, split="train", streaming=True)

    buffer: List[int] = []
    tokens_produced = 0

    for item in ds:
        # Support common code dataset text keys
        content = item.get("content") or item.get("text") or item.get("code") or ""
        if not content.strip():
            continue

        token_ids = tokenizer.encode(content, add_special_tokens=False)
        buffer.extend(token_ids)

        # Emit full batches
        while len(buffer) >= batch_size * seq_len:
            batch_tokens = buffer[: batch_size * seq_len]
            buffer = buffer[batch_size * seq_len :]

            tensor_batch = torch.tensor(batch_tokens, dtype=torch.long).view(batch_size, seq_len)
            tokens_produced += batch_size * seq_len
            yield tensor_batch.to(device=device)

            if tokens_produced >= max_tokens:
                return


def main():
    parser = argparse.ArgumentParser(description="Train 100M LRH on 2x Nvidia T4")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    parser.add_argument("--target-tokens", type=int, default=100_000_000, help="Target token count (default: 100M)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size per GPU (default: 16)")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length (default: 512)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate for LRH (default: 1e-3)")
    parser.add_argument("--k-amb", type=int, default=5, help="Top-k ambiguity exclusion threshold (default: 5)")
    parser.add_argument("--gamma-gate-auc", type=float, default=0.55, help="AUC threshold to declare gamma readiness")
    parser.add_argument("--output-dir", type=str, default="/kaggle/working/corosred_lrh_100m")
    parser.add_argument("--save-every-steps", type=int, default=1000)
    parser.add_argument("--dataset-name", type=str, default="bigcode/the-stack-smol-python")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick 20-step smoke test")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Device and Precision Configuration
    if torch.cuda.is_available():
        device = "cuda"
        num_gpus = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
        print(f"CUDA detected: {num_gpus} GPU(s) -> {gpu_names}")
        # T4 architecture requires float16 mixed precision (no bf16 hardware support)
        use_amp = True
        amp_dtype = torch.float16
    elif torch.backends.mps.is_available():
        device = "mps"
        num_gpus = 1
        use_amp = False
        amp_dtype = torch.bfloat16
        print("Apple MPS detected -> running local bfloat16")
    else:
        device = "cpu"
        num_gpus = 1
        use_amp = False
        amp_dtype = torch.float32
        print("CPU detected")

    total_batch_size = args.batch_size * max(1, num_gpus)
    tokens_per_step = total_batch_size * args.seq_len
    total_steps = math.ceil(args.target_tokens / tokens_per_step)
    if args.smoke_test:
        total_steps = 20

    print("=" * 80)
    print("COROSred LEARNED RELIABILITY HEAD (LRH) 100M TRAINING RUN")
    print("=" * 80)
    print(f"Backbone Model: {args.model_id} (FROZEN)")
    print(f"Target Tokens: {args.target_tokens:,} across {total_steps} steps")
    print(f"Batch Size: {args.batch_size} per GPU * {num_gpus} GPUs = {total_batch_size} ({tokens_per_step:,} tok/step)")
    print(f"Sequence Length: {args.seq_len}")
    print(f"Ambiguity Exclusion: Top-{args.k_amb}")
    print(f"Precision: {amp_dtype} via AMP={use_amp}")
    print(f"Output Directory: {out_dir}")

    # 2. Tokenizer and Frozen Backbone
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\nLoading tokenizer and backbone weights for {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load backbone in half precision
    backbone = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=amp_dtype if device != "cpu" else torch.float32,
        trust_remote_code=True,
    ).to(device)

    # Freeze entire backbone
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    print(f"Backbone frozen successfully ({sum(p.numel() for p in backbone.parameters()):,} parameters).")

    # 3. Instantiate LRH Head
    d_model = backbone.config.hidden_size
    lrh_head = LearnedReliabilityHead(d_model=d_model).to(device)

    # Multi-GPU DataParallel wrapping if 2x T4 available
    if device == "cuda" and num_gpus > 1:
        backbone = nn.DataParallel(backbone)
        lrh_head_parallel = nn.DataParallel(lrh_head)
    else:
        lrh_head_parallel = lrh_head

    optimizer = torch.optim.AdamW(lrh_head.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device == "cuda"))

    # Learning rate schedule (cosine decay with 5% warmup)
    warmup_steps = max(10, int(total_steps * 0.05))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 4. Data Stream Setup
    data_stream = build_data_stream(
        tokenizer=tokenizer,
        dataset_name=args.dataset_name,
        seq_len=args.seq_len,
        batch_size=total_batch_size,
        max_tokens=args.target_tokens,
        device=device,
    )

    # 5. Training Loop
    print("\nStarting training loop...")
    print("-" * 80)

    step = 0
    tokens_seen = 0
    t_start = time.perf_counter()
    metrics_history: List[Dict[str, Any]] = []

    auc_ema: Optional[float] = None
    decay_ema = 0.95

    for batch in data_stream:
        step += 1
        tokens_seen += batch.numel()

        # Step A: Causal Backbone Forward Pass (Zero Grad)
        with torch.no_grad():
            outputs = backbone(batch, output_hidden_states=True)
            # Hidden state from final transformer block (before output projection)
            h = outputs.hidden_states[-1]
            logits = outputs.logits

            # Next-token prediction alignment
            shift_logits = logits[:, :-1, :]
            shift_targets = batch[:, 1:]
            shift_h = h[:, :-1, :]

            # Exact match labels: 1.0 if top-1 argmax matches target, else 0.0
            pred_tokens = shift_logits.argmax(dim=-1)
            is_exact_match = (pred_tokens == shift_targets)
            labels = is_exact_match.float()

            # Ambiguity Exclusion (k_amb): exclude tokens where target was in top-k but not top-1
            topk_preds = torch.topk(shift_logits, k=args.k_amb, dim=-1).indices
            is_target_in_topk = (topk_preds == shift_targets.unsqueeze(-1)).any(dim=-1)
            is_ambiguous = is_target_in_topk & (~is_exact_match)
            valid_mask = ~is_ambiguous

            # Exclude padding or special tokens
            if tokenizer.pad_token_id is not None:
                valid_mask = valid_mask & (shift_targets != tokenizer.pad_token_id)

        # Step B: LRH Forward Pass & Loss
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
            r_scores = lrh_head_parallel(shift_h)
            bce_raw = F.binary_cross_entropy_with_logits(r_scores, labels, reduction="none")
            masked_bce = bce_raw * valid_mask.float()
            valid_token_count = valid_mask.float().sum().clamp(min=1.0)
            loss = masked_bce.sum() / valid_token_count

        # Step C: Optimization Step
        if use_amp and device == "cuda":
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(lrh_head.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(lrh_head.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        # Step D: Metrics Telemetry
        with torch.no_grad():
            # Classification Accuracy on valid tokens
            preds_binary = (r_scores > 0.0)
            correct = (preds_binary == (labels > 0.5)) & valid_mask
            acc = float((correct.float().sum() / valid_token_count).item())

            # Balanced Accuracy (TPR + TNR) / 2
            pos_mask = (labels > 0.5) & valid_mask
            neg_mask = (labels <= 0.5) & valid_mask
            tpr = float((preds_binary & pos_mask).float().sum() / pos_mask.float().sum().clamp(min=1.0))
            tnr = float(((~preds_binary) & neg_mask).float().sum() / neg_mask.float().sum().clamp(min=1.0))
            bal_acc = 0.5 * (tpr + tnr)

            # ROC-AUC computation
            flat_scores = r_scores[valid_mask]
            flat_labels = labels[valid_mask]
            batch_auc = compute_vectorized_roc_auc(flat_scores, flat_labels)

            if auc_ema is None:
                auc_ema = batch_auc
            else:
                auc_ema = decay_ema * auc_ema + (1.0 - decay_ema) * batch_auc

            gamma_ready = (auc_ema >= args.gamma_gate_auc)

        # Logging
        elapsed = time.perf_counter() - t_start
        tok_per_sec = tokens_seen / max(1e-6, elapsed)

        if step % 50 == 0 or step == total_steps or args.smoke_test:
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"Step [{step:>5d}/{total_steps}] | "
                f"Tokens: {tokens_seen:>10,d} ({tokens_seen / args.target_tokens * 100:>5.1f}%) | "
                f"Loss: {loss.item():.4f} | "
                f"Acc: {acc * 100:.1f}% | "
                f"BalAcc: {bal_acc * 100:.1f}% | "
                f"AUC: {batch_auc:.3f} (EMA: {auc_ema:.3f}) | "
                f"γ-Ready: {'YES' if gamma_ready else 'WAIT'} | "
                f"Speed: {tok_per_sec:,.0f} tok/s"
            )

        metrics_history.append({
            "step": step,
            "tokens_seen": tokens_seen,
            "loss": float(loss.item()),
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "auc": batch_auc,
            "auc_ema": float(auc_ema),
            "gamma_ready": gamma_ready,
        })

        # Step E: Periodic Checkpoints
        if step % args.save_every_steps == 0 or step == total_steps:
            ckpt_path = out_dir / f"lrh_checkpoint_step_{step}.pt"
            torch.save({
                "step": step,
                "tokens_seen": tokens_seen,
                "lrh_state_dict": lrh_head.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "auc_ema": auc_ema,
            }, ckpt_path)
            print(f"  [Checkpoint] Saved step {step} to {ckpt_path.name}")

        if step >= total_steps:
            break

    total_time = time.perf_counter() - t_start
    print("-" * 80)
    print(f"Training completed in {total_time / 60:.2f} minutes ({tokens_seen:,.0f} tokens processed).")
    print(f"Average Throughput: {tokens_seen / total_time:,.0f} tokens/second.")
    print(f"Final LRH ROC-AUC EMA: {auc_ema:.4f} (Gamma Gate Threshold: {args.gamma_gate_auc})")

    # 6. Save Final Standalone Artifacts
    final_pt_path = out_dir / "lrh_head_100m.pt"
    torch.save(lrh_head.state_dict(), final_pt_path)

    # Save safetensors for modular distribution
    final_safetensors_path = out_dir / "lrh_head_100m.safetensors"
    # Ensure float32 or bfloat16 for safetensors export
    st_dict = {k: v.contiguous().cpu() for k, v in lrh_head.state_dict().items()}
    save_file(st_dict, str(final_safetensors_path))

    # Save metadata configuration
    config_dict = {
        "model_id": args.model_id,
        "d_model": d_model,
        "lrh_architecture": "Linear(896, 896) -> SiLU -> Linear(896, 1)",
        "parameter_count": sum(p.numel() for p in lrh_head.parameters()),
        "target_tokens": args.target_tokens,
        "actual_tokens_seen": tokens_seen,
        "total_steps": step,
        "final_auc_ema": float(auc_ema) if auc_ema else 0.50,
        "gamma_gate_auc": args.gamma_gate_auc,
        "gamma_ready": bool(auc_ema >= args.gamma_gate_auc) if auc_ema else False,
        "precision": str(amp_dtype),
        "total_training_minutes": total_time / 60,
    }
    with open(out_dir / "lrh_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    with open(out_dir / "metrics_history.json", "w") as f:
        json.dump(metrics_history, f, indent=2)

    print(f"\nFinal artifacts saved to {out_dir}:")
    print(f"  - Model: {final_safetensors_path.name}")
    print(f"  - Torch: {final_pt_path.name}")
    print(f"  - Config: lrh_config.json")
    print(f"  - Metrics: metrics_history.json")


if __name__ == "__main__":
    main()
