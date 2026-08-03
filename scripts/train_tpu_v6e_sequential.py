"""Sequential High-Density TPU v6e-1 Training Script.

Executes sequential 1:1 scaling across 3 model sizes:
1. 125M Model (125M tokens, ~238 steps, ~4 min)
2. 250M Model (250M tokens, ~476 steps, ~12 min)
3. 500M Model (500M tokens, ~953 steps, ~45 min)

Effective Batch Size = 1024 (524,288 tokens/step)
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
from tokenizers import Tokenizer
from telos.data.dataset import TelosDataset
from telos.diffusion.forward_process import apply_masking
from telos.diffusion.loss import mdlm_loss
from telos.training.lr_schedule import get_cosine_schedule_with_warmup


def train_tpu_model(model_name: str, model_cfg: dict, global_cfg: dict, dataset: TelosDataset, tokenizer: TelosTokenizer):
    """Trains a single model size on TPU v6e-1 at effective batch size 1024."""
    print(f"\n" + "=" * 80)
    print(f"STARTING TPU v6e-1 TRAINING: {model_name} MODEL (1:1 RATIO)")
    print(f"Token Target: {model_cfg['tokens']:,} tokens | Target Steps: {model_cfg['max_steps']}")
    print(f"Effective Batch Size: {global_cfg['batch_size'] * global_cfg['gradient_accumulation']} sequences (524,288 tokens/step)")
    print("=" * 80 + "\n")

    # Detect TPU XLA device or fallback to CUDA/MPS/CPU
    try:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        print(">> Using TPU XLA Device")
    except ImportError:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f">> TPU XLA not detected. Falling back to {device}")

    # 1. Model Config
    config = TelosConfig(
        vocab_size=global_cfg["vocab_size"],
        d_model=model_cfg["d_model"],
        n_layers=model_cfg["n_layers"],
        n_heads=model_cfg["n_heads"],
        max_seq_len=global_cfg["seq_len"],
        dropout=0.0,
        tied_embeddings=True
    )

    model = TelosTransformer(config).to(device)

    # 2. Optimizer & LR Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=global_cfg["max_lr"],
        weight_decay=global_cfg["weight_decay"]
    )

    max_steps = model_cfg["max_steps"]
    warmup_steps = model_cfg["warmup_steps"]

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
        min_lr=global_cfg["min_lr"],
        max_lr=global_cfg["max_lr"]
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

    ckpt_dir = Path(model_cfg["checkpoint_dir"])
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

        torch.nn.utils.clip_grad_norm_(model.parameters(), global_cfg["grad_clip"])

        try:
            import torch_xla.core.xla_model as xm
            xm.optimizer_step(optimizer)
        except ImportError:
            optimizer.step()

        scheduler.step()
        step += 1

        if step % 25 == 0 or step == max_steps:
            current_lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - start_time
            steps_per_sec = step / max(1.0, elapsed)
            tok_per_sec = steps_per_sec * batch_size * grad_accum * global_cfg["seq_len"]
            print(f"{model_name:<5} | Step {step:>4}/{max_steps} | Loss: {accum_loss:.4f} | Unweighted CE: {unweighted_ce:.4f} | LR: {current_lr:.6f} | Tok/s: {tok_per_sec:,.0f}")

    # Save final model checkpoint
    ckpt_path = ckpt_dir / f"checkpoint_tpu_{model_name}_final_step_{step}.pt"
    torch.save({
        "step": step,
        "model_name": model_name,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "final_loss": accum_loss,
        "unweighted_ce": unweighted_ce
    }, ckpt_path)

    print(f"\nSuccessfully completed {model_name} model training! Saved: {ckpt_path}\n")


def run_sequential_tpu_scaling(config_path: str = "configs/phase_c_tpu_v6e.yaml"):
    """Runs 125M -> 250M -> 500M sequentially on TPU v6e-1."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    global_cfg = config["global"]
    models = config["models"]

    from tokenizers import Tokenizer
    tokenizer_path = "configs/tokenizer_mac.json" if Path("configs/tokenizer_mac.json").exists() else "configs/tokenizer.json"
    tokenizer = Tokenizer.from_file(tokenizer_path)

    data_path = "data/python_corpus_1.7b.bin" if Path("data/python_corpus_1.7b.bin").exists() else "data/train.bin"
    dataset = TelosDataset(
        data_path=data_path,
        seq_len=global_cfg["seq_len"]
    )

    for model_name, model_cfg in models.items():
        train_tpu_model(model_name, model_cfg, global_cfg, dataset, tokenizer)


if __name__ == "__main__":
    cfg_file = sys.argv[1] if len(sys.argv) > 1 else "configs/phase_c_tpu_v6e.yaml"
    run_sequential_tpu_scaling(cfg_file)
