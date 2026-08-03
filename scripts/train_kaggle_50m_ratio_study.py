"""Automated Kaggle 50M Ratio Study Training Runner.

Executes sequential training across 5 overtraining ratios:
1:1 (50M tokens), 1:10 (500M tokens), 1:20 (1.0B tokens), 1:30 (1.5B tokens), 1:40 (2.0B tokens)

Applies dedicated per-run Cosine LR decay schedules so every ratio checkpoint
fully decays to min_lr at its exact final step.
"""

import os
import sys
import time
from pathlib import Path
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

from telos.model.transformer import TelosTransformer, TelosConfig
from telos.data.tokenizer import TelosTokenizer
from telos.data.dataset import TelosDataset
from telos.diffusion.forward_process import apply_masking
from telos.diffusion.loss import mdlm_loss
from telos.training.lr_schedule import get_cosine_schedule_with_warmup


def train_ratio_checkpoint(ratio_name: str, ratio_cfg: dict, global_cfg: dict, dataset: TelosDataset, tokenizer: TelosTokenizer):
    """Trains a single ratio checkpoint with dedicated Cosine LR decay."""
    print(f"\n" + "=" * 80)
    print(f"STARTING 50M MODEL RATIO RUN: {ratio_name} ({ratio_cfg['tokens']:,} tokens)")
    print(f"Target Steps: {ratio_cfg['max_steps']} | Effective Batch Size: {global_cfg['batch_size'] * global_cfg['gradient_accumulation']}")
    print("=" * 80 + "\n")

    device = torch.device(global_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    # 1. Model Configuration
    model_config = TelosConfig(
        vocab_size=global_cfg["vocab_size"],
        d_model=global_cfg["d_model"],
        n_layers=global_cfg["n_layers"],
        n_heads=global_cfg["n_heads"],
        max_seq_len=global_cfg["seq_len"],
        dropout=global_cfg.get("dropout", 0.0),
        tied_embeddings=global_cfg.get("tied_embeddings", True)
    )

    model = TelosTransformer(model_config).to(device)

    # 2. Optimizer and Dedicated LR Scheduler for this specific ratio
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=global_cfg["max_lr"],
        weight_decay=global_cfg.get("weight_decay", 0.1)
    )

    max_steps = ratio_cfg["max_steps"]
    warmup_steps = ratio_cfg["warmup_steps"]
    min_lr = global_cfg.get("min_lr", 4e-5)
    max_lr = global_cfg["max_lr"]

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
        min_lr=min_lr,
        max_lr=max_lr
    )

    # 3. DataLoader
    batch_size = global_cfg["batch_size"]
    grad_accum = global_cfg["gradient_accumulation"]
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    ckpt_dir = Path(global_cfg.get("checkpoint_dir", "checkpoints/phase_b_50m_ratio_study"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    mask_token_id = tokenizer.token_to_id("[MASK]")
    if mask_token_id is None:
        mask_token_id = 4

    model.train()
    step = 0
    start_time = time.time()
    data_iter = iter(dataloader)

    while step < max_steps:
        optimizer.zero_grad()
        accum_loss = 0.0

        for _ in range(grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device)

            masked_ids, mask_positions, t_values = apply_masking(
                input_ids,
                mask_token_id=mask_token_id,
                special_token_ids={0, 2, 3}
            )

            logits = model(masked_ids)
            loss, unweighted_ce = mdlm_loss(
                logits=logits,
                target_ids=input_ids,
                mask_positions=mask_positions,
                t_values=t_values,
                mask_token_id=mask_token_id
            )

            loss = loss / grad_accum
            loss.backward()
            accum_loss += loss.item() * grad_accum

        torch.nn.utils.clip_grad_norm_(model.parameters(), global_cfg.get("grad_clip", 1.0))
        optimizer.step()
        scheduler.step()
        step += 1

        if step % 50 == 0 or step == max_steps:
            current_lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - start_time
            steps_per_sec = step / max(1.0, elapsed)
            tok_per_sec = steps_per_sec * batch_size * grad_accum * global_cfg["seq_len"]
            print(f"Ratio {ratio_name:<5} | Step {step:>5}/{max_steps} | Loss: {accum_loss:.4f} | Unweighted CE: {unweighted_ce:.4f} | LR: {current_lr:.6f} | Tok/s: {tok_per_sec:,.0f}")

    # Save final fully-decayed ratio checkpoint
    ckpt_path = ckpt_dir / f"checkpoint_50m_ratio_{ratio_name.replace(':', '_')}_step_{step}.pt"
    torch.save({
        "step": step,
        "ratio": ratio_name,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": model_config,
        "final_loss": accum_loss,
        "unweighted_ce": unweighted_ce
    }, ckpt_path)

    print(f"\nSuccessfully saved fully decayed ratio checkpoint: {ckpt_path}\n")


def run_ratio_study(config_path: str = "configs/phase_b_50m_ratio_study.yaml"):
    """Runs all 5 ratio checkpoints sequentially."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    train_cfg = config["training"]
    train_cfg.update(model_cfg)
    train_cfg["checkpoint_dir"] = config["checkpoint"]["dir"]

    # Load tokenizer & dataset
    tokenizer = TelosTokenizer("configs/tokenizer_mac.json")
    dataset = TelosDataset(
        data_path="data/train.bin",
        seq_len=model_cfg["seq_len"]
    )

    ratios = config["ratios"]
    for ratio_name, ratio_cfg in ratios.items():
        train_ratio_checkpoint(ratio_name, ratio_cfg, train_cfg, dataset, tokenizer)


if __name__ == "__main__":
    cfg_file = sys.argv[1] if len(sys.argv) > 1 else "configs/phase_b_50m_ratio_study.yaml"
    run_ratio_study(cfg_file)
