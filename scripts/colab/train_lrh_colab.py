#!/usr/bin/env python3
"""Google Colab LRH Trainer for Qwen2.5-Coder (Hardware-Agnostic: GPU & TPU v5e).

Optimized for Google Colab runtimes:
- GPU: Nvidia T4 (FP16 AMP), L4 / A100 (BF16 native)
- TPU: v5e-1 / v2 / v3 via PyTorch-XLA (BF16 native)
- Memory-Mapped Binaries: Streams pre-tokenized uint32 data with zero CPU tokenization overhead.

Usage in Google Colab:
    # Stage 1: 10M Tokens (~1.5 min on A100, ~2 min on TPU v5e, ~8 min on T4)
    python scripts/colab/train_lrh_colab.py --target-tokens 10000000

    # Stage 2: 100M Tokens (~15 min on A100, ~20 min on TPU v5e, ~80 min on T4)
    python scripts/colab/train_lrh_colab.py --target-tokens 100000000 --resume-from /content/corosred_lrh_qwen25/lrh_qwen25_10m.pt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
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
        return self.net(h).squeeze(-1)


class MemmapTokenDataset:
    """Zero-overhead memory-mapped token loader."""

    def __init__(self, bin_path: str | Path, seq_len: int = 512):
        self.bin_path = Path(bin_path)
        assert self.bin_path.exists(), f"Binary file not found: {bin_path}"
        self.data = np.memmap(str(self.bin_path), dtype=np.uint32, mode="r")
        self.seq_len = seq_len
        self.total_tokens = len(self.data)
        self.total_sequences = self.total_tokens // seq_len
        print(f"Loaded memory-mapped binary: {self.total_tokens:,} tokens ({self.total_sequences:,} sequences of len {seq_len}).")

    def get_batch(self, batch_idx: int, batch_size: int, device: Any) -> torch.Tensor:
        start_seq = (batch_idx * batch_size) % max(1, self.total_sequences - batch_size)
        start_tok = start_seq * self.seq_len
        end_tok = start_tok + batch_size * self.seq_len

        chunk = np.array(self.data[start_tok:end_tok], dtype=np.int64)
        tensor = torch.from_numpy(chunk).view(batch_size, self.seq_len)
        return tensor.to(device=device, non_blocking=True)


def detect_colab_hardware() -> Tuple[Any, str, torch.dtype, bool]:
    """Detects Colab hardware accelerator (CUDA GPU, TPU-XLA, or CPU)."""
    # 1. Check for TPU (PyTorch-XLA)
    try:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        print("Hardware Accelerator: Google Cloud TPU detected via PyTorch-XLA.")
        return device, "tpu", torch.bfloat16, False
    except ImportError:
        pass
    except Exception as e:
        print(f"Notice: torch_xla import check passed but device init failed ({e}), checking CUDA...")

    # 2. Check for CUDA GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        num_gpus = torch.cuda.device_count()
        print(f"Hardware Accelerator: CUDA detected ({num_gpus}x {gpu_name}).")

        # BF16 for Ampere / Hopper / Ada (A100, L4, V100/A10G)
        if torch.cuda.is_bf16_supported():
            return torch.device("cuda"), "cuda", torch.bfloat16, True
        # FP16 for Turing (T4)
        return torch.device("cuda"), "cuda", torch.float16, True

    # 3. Fallback
    print("Hardware Accelerator: None (CPU mode).")
    return torch.device("cpu"), "cpu", torch.float32, False


def ensure_binary_dataset(repo_id: str, target_tokens: int, cache_dir: str = "/content/data") -> Path:
    """Ensures the appropriate binary dataset exists locally, downloading from HF if needed."""
    os.makedirs(cache_dir, exist_ok=True)
    bin_name = "qwen25_coder_100m.bin" if target_tokens > 10_000_000 else "qwen25_coder_10m.bin"
    local_path = Path(cache_dir) / bin_name

    if local_path.exists():
        print(f"Found local binary dataset: {local_path} ({os.path.getsize(local_path) / (1024*1024):.1f} MB)")
        return local_path

    print(f"Downloading {bin_name} from Hugging Face repo '{repo_id}'...")
    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=f"data/{bin_name}",
        repo_type="model",
        local_dir=cache_dir,
    )
    print(f"Downloaded {bin_name} successfully: {downloaded_path}")
    return Path(downloaded_path)


def main():
    parser = argparse.ArgumentParser(description="Google Colab LRH Trainer for Qwen2.5-Coder")
    parser.add_argument("--repo-id", type=str, default="Kazenowoko/telos", help="Hugging Face repo holding pre-tokenized data")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    parser.add_argument("--target-tokens", type=int, default=10_000_000, help="Target token count: 10,000,000 or 100,000,000")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size per forward pass (default: 32)")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length (default: 512)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--k-amb", type=int, default=5, help="Top-k ambiguity exclusion (default: 5)")
    parser.add_argument("--gamma-gate-auc", type=float, default=0.55, help="Gamma gate readiness threshold")
    parser.add_argument("--output-dir", type=str, default="/content/corosred_lrh_qwen25")
    parser.add_argument("--resume-from", type=str, default=None, help="Checkpoint path to resume from")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device, hw_type, precision, use_amp = detect_colab_hardware()

    # Adjust batch size for T4 if needed
    batch_size = args.batch_size
    if hw_type == "cuda" and not torch.cuda.is_bf16_supported() and batch_size > 16:
        # T4 GPU (16GB VRAM) performs best with batch_size=16
        batch_size = 16
        print(f"Adjusted batch size to {batch_size} for optimal T4 VRAM utilization.")

    tokens_per_step = batch_size * args.seq_len
    total_steps = math.ceil(args.target_tokens / tokens_per_step)

    print("=" * 80)
    print("COLAB LRH TRAINING PIPELINE FOR QWEN2.5-CODER")
    print("=" * 80)
    print(f"Target Tokens:    {args.target_tokens:,} across {total_steps:,} steps")
    print(f"Batch Size:       {batch_size} ({tokens_per_step:,} tok/step)")
    print(f"Sequence Length:  {args.seq_len}")
    print(f"Precision:        {precision} (AMP: {use_amp})")
    print(f"Hardware:         {hw_type.upper()}")

    # 1. Ensure Dataset
    bin_file = ensure_binary_dataset(args.repo_id, args.target_tokens)
    dataset = MemmapTokenDataset(bin_file, seq_len=args.seq_len)

    # 2. Frozen Backbone
    from transformers import AutoModelForCausalLM

    print(f"\nLoading frozen backbone: {args.model_id}...")
    backbone = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=precision if hw_type != "cpu" else torch.float32,
        trust_remote_code=True,
    ).to(device)

    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    print("Backbone frozen successfully.")

    # 3. LRH Head
    d_model = backbone.config.hidden_size
    lrh_head = LearnedReliabilityHead(d_model=d_model).to(device)

    optimizer = torch.optim.AdamW(lrh_head.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and hw_type == "cuda" and precision == torch.float16))

    start_step = 0
    tokens_seen = 0
    auc_ema: Optional[float] = None

    if args.resume_from and os.path.exists(args.resume_from):
        print(f"Resuming from checkpoint: {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location=device)
        lrh_head.load_state_dict(ckpt["lrh_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt.get("step", 0)
        tokens_seen = ckpt.get("tokens_seen", 0)
        auc_ema = ckpt.get("auc_ema", None)
        print(f"Resumed from step {start_step} ({tokens_seen:,} tokens).")

    warmup_steps = max(10, int(total_steps * 0.05))

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=start_step - 1)

    # 4. Training Loop
    print("\nStarting Colab training loop...")
    print("-" * 80)

    step = start_step
    t_start = time.perf_counter()
    metrics_history: List[Dict[str, Any]] = []
    decay_ema = 0.95
    label_suffix = "100m" if args.target_tokens >= 100_000_000 else "10m"

    while step < total_steps:
        step += 1
        tokens_seen += tokens_per_step

        batch = dataset.get_batch(step, batch_size, device=device)

        # Step A: Causal Backbone Forward Pass (Zero Grad)
        with torch.no_grad():
            outputs = backbone(batch, output_hidden_states=True)
            h = outputs.hidden_states[-1]
            logits = outputs.logits

            shift_logits = logits[:, :-1, :]
            shift_targets = batch[:, 1:]
            shift_h = h[:, :-1, :]

            pred_tokens = shift_logits.argmax(dim=-1)
            is_exact_match = (pred_tokens == shift_targets)
            labels = is_exact_match.float()

            topk_preds = torch.topk(shift_logits, k=args.k_amb, dim=-1).indices
            is_target_in_topk = (topk_preds == shift_targets.unsqueeze(-1)).any(dim=-1)
            is_ambiguous = is_target_in_topk & (~is_exact_match)
            valid_mask = ~is_ambiguous

        # Step B: LRH Forward Pass & Loss
        optimizer.zero_grad()

        if hw_type == "cuda":
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=precision):
                r_scores = lrh_head(shift_h)
                bce_raw = F.binary_cross_entropy_with_logits(r_scores, labels, reduction="none")
                masked_bce = bce_raw * valid_mask.float()
                valid_count = valid_mask.float().sum().clamp(min=1.0)
                loss = masked_bce.sum() / valid_count
        else:
            r_scores = lrh_head(shift_h)
            bce_raw = F.binary_cross_entropy_with_logits(r_scores, labels, reduction="none")
            masked_bce = bce_raw * valid_mask.float()
            valid_count = valid_mask.float().sum().clamp(min=1.0)
            loss = masked_bce.sum() / valid_count

        # Step C: Optimizer Step
        if hw_type == "tpu":
            loss.backward()
            nn.utils.clip_grad_norm_(lrh_head.parameters(), max_norm=1.0)
            import torch_xla.core.xla_model as xm
            xm.optimizer_step(optimizer)
        elif scaler.is_enabled():
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

        # Step D: Telemetry
        with torch.no_grad():
            preds_binary = (r_scores > 0.0)
            correct = (preds_binary == (labels > 0.5)) & valid_mask
            acc = float((correct.float().sum() / valid_count).item())

            pos_mask = (labels > 0.5) & valid_mask
            neg_mask = (labels <= 0.5) & valid_mask
            tpr = float((preds_binary & pos_mask).float().sum() / pos_mask.float().sum().clamp(min=1.0))
            tnr = float(((~preds_binary) & neg_mask).float().sum() / neg_mask.float().sum().clamp(min=1.0))
            bal_acc = 0.5 * (tpr + tnr)

            flat_scores = r_scores[valid_mask].cpu()
            flat_labels = labels[valid_mask].cpu()
            batch_auc = compute_vectorized_roc_auc(flat_scores, flat_labels)

            if auc_ema is None:
                auc_ema = batch_auc
            else:
                auc_ema = decay_ema * auc_ema + (1.0 - decay_ema) * batch_auc

            gamma_ready = (auc_ema >= args.gamma_gate_auc)

        elapsed = time.perf_counter() - t_start
        tok_per_sec = (tokens_seen - (start_step * tokens_per_step)) / max(1e-6, elapsed)

        if step % 25 == 0 or step == total_steps:
            print(
                f"Step [{step:>5d}/{total_steps}] | "
                f"Tokens: {tokens_seen:>10,d} ({tokens_seen / args.target_tokens * 100:>5.1f}%) | "
                f"Loss: {loss.item():.4f} | "
                f"BalAcc: {bal_acc * 100:.1f}% | "
                f"AUC: {batch_auc:.3f} (EMA: {auc_ema:.3f}) | "
                f"γ-Gate: {'READY' if gamma_ready else 'WAIT'} | "
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

    total_time = time.perf_counter() - t_start
    print("-" * 80)
    print(f"Colab training completed in {total_time / 60:.2f} minutes!")
    print(f"Average Speed: {tokens_seen / total_time:,.0f} tokens/second.")
    print(f"Final LRH ROC-AUC EMA: {auc_ema:.4f}")

    # 5. Save Final Artifacts
    final_pt = out_dir / f"lrh_qwen25_{label_suffix}.pt"
    final_st = out_dir / f"lrh_qwen25_{label_suffix}.safetensors"
    torch.save({
        "step": step,
        "tokens_seen": tokens_seen,
        "lrh_state_dict": lrh_head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "auc_ema": auc_ema,
    }, final_pt)
    save_file({k: v.contiguous().cpu() for k, v in lrh_head.state_dict().items()}, str(final_st))

    config_dict = {
        "model_id": args.model_id,
        "d_model": d_model,
        "target_tokens": args.target_tokens,
        "actual_tokens_seen": tokens_seen,
        "total_steps": step,
        "final_auc_ema": float(auc_ema) if auc_ema else 0.50,
        "gamma_gate_auc": args.gamma_gate_auc,
        "gamma_ready": bool(auc_ema >= args.gamma_gate_auc) if auc_ema else False,
        "precision": str(precision),
        "hardware": hw_type,
        "total_training_minutes": total_time / 60,
    }
    with open(out_dir / f"lrh_config_{label_suffix}.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    with open(out_dir / f"metrics_{label_suffix}.json", "w") as f:
        json.dump(metrics_history, f, indent=2)

    print(f"\nArtifacts successfully written to {out_dir}:")
    print(f"  - Model weights: {final_st.name}")
    print(f"  - Checkpoint:    {final_pt.name}")
    print(f"  - Metadata:      lrh_config_{label_suffix}.json")


if __name__ == "__main__":
    main()
