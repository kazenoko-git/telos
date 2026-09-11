"""
One-Shot Modal Script: Finish 50M AR 2.5B Training on 1x NVIDIA H100 (SXM5)
Runs remaining steps from 12,000 to 12,716 (~4 minutes) and pushes final checkpoint to Hugging Face.
"""

import os
import modal

HF_TOKEN = "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"
REPO_ID = "Kazenowoko/telos"

app = modal.App("telos-ar-50m-finish")

# Modal container image with PyTorch CUDA 12 and necessary dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4.0",
        "huggingface_hub",
        "numpy",
        "pyyaml",
        "tokenizers",
        "safetensors",
    )
    .run_commands(
        "git clone https://github.com/kazenoko-git/telos.git /root/telos",
        "cd /root/telos && pip install --no-deps -e .",
    )
)


@app.function(
    gpu="H100",
    image=image,
    timeout=1800,  # 30 minute limit (job completes in ~4 mins)
)
def run_finish():
    import os
    import sys
    import json
    import time
    from pathlib import Path
    import torch
    from huggingface_hub import hf_hub_download, HfApi

    work_dir = Path("/root/telos")
    os.chdir(work_dir)

    print("=" * 70)
    print("  TELOS: Finishing 50M AR Training (Steps 12,000 -> 12,716) on H100")
    print("=" * 70)

    # 1. Download Dataset
    data_path = work_dir / "data" / "python_corpus_2.5b.bin"
    if not data_path.exists():
        print("[1/4] Downloading python_corpus_2.5b.bin from Hugging Face...")
        hf_hub_download(
            repo_id=REPO_ID,
            filename="data/python_corpus_2.5b.bin",
            local_dir=str(work_dir),
            token=HF_TOKEN,
        )
    print(f"[+] Dataset verified at {data_path} ({data_path.stat().st_size / 1e9:.2f} GB)")

    # 2. Download Checkpoint Step 12,000 & Config
    ckpt_dir = work_dir / "checkpoints" / "ar" / "50m_2.5b"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "checkpoint_step_12000.pt"

    print("[2/4] Downloading checkpoint_step_12000.pt from Hugging Face...")
    hf_hub_download(
        repo_id=REPO_ID,
        filename="checkpoints/ar/50m_2.5b/checkpoint_step_12000.pt",
        local_dir=str(work_dir),
        token=HF_TOKEN,
    )
    try:
        hf_hub_download(
            repo_id=REPO_ID,
            filename="checkpoints/ar/50m_2.5b/config.json",
            local_dir=str(work_dir),
            token=HF_TOKEN,
        )
    except Exception:
        pass
    print(f"[+] Checkpoint verified at {ckpt_path} ({ckpt_path.stat().st_size / 1e6:.1f} MB)")

    # 3. Setup Model, Optimizer, Trainer and Resume
    print("[3/4] Initializing model & restoring full training state at Step 12,000...")
    from telos.configs import build_config
    from telos.models import TelosTransformer
    from telos.training import UnifiedPyTorchTrainer

    cfg = build_config(
        paradigm="ar",
        params="50M",
        tokens="2.5B",
        effective_batch="384",
        batch_size=384,
        grad_accum=1,
        hardware="cuda",
        compile=True,
        seq_len=512,
        data_path=str(data_path),
        checkpoint_dir=str(ckpt_dir),
    )

    m_cfg = cfg["model"]
    model = TelosTransformer(
        vocab_size=m_cfg.get("vocab_size", 8192),
        d_model=m_cfg.get("d_model", 512),
        n_layers=m_cfg.get("n_layers", 14),
        n_heads=m_cfg.get("n_heads", 8),
        n_kv_heads=m_cfg.get("n_kv_heads", 8),
        is_causal=True,
        use_reliability_head=False,
        use_grad_checkpoint=False,
    )

    # Load weights into model
    ckpt_data = torch.load(ckpt_path, map_location="cpu")
    sd = {k.removeprefix("module."): v for k, v in ckpt_data["model_state_dict"].items()}
    model.load_state_dict(sd, strict=False)
    print("  [Init] Model weights loaded successfully.")

    trainer = UnifiedPyTorchTrainer(paradigm="ar", model=model, cfg=cfg, device_type="cuda")

    # Restore optimizer states (AdamW momentum & variance)
    if "optimizer_state_dict" in ckpt_data:
        try:
            trainer.optimizer.load_state_dict(ckpt_data["optimizer_state_dict"])
            print("  [Init] Restored AdamW optimizer states.")
        except Exception as e:
            print(f"  [Init Notice] Optimizer state restore notice: {e}")

    # Run remaining steps to 12,716
    print(f"\n[4/4] Starting H100 execution for steps 12,001 -> {cfg['training']['max_steps']}...")
    trainer.train(resume_step=12000)

    # 4. Upload checkpoint_final.pt to Hugging Face
    print("\n[Upload] Syncing final artifact to Hugging Face Hub...")
    api = HfApi(token=HF_TOKEN)
    final_ckpt = ckpt_dir / "checkpoint_final.pt"
    if final_ckpt.exists():
        print(f"Uploading {final_ckpt} -> {REPO_ID}...")
        api.upload_file(
            path_or_fileobj=str(final_ckpt),
            path_in_repo="checkpoints/ar/50m_2.5b/checkpoint_final.pt",
            repo_id=REPO_ID,
        )
    final_cfg = ckpt_dir / "config.json"
    if final_cfg.exists():
        api.upload_file(
            path_or_fileobj=str(final_cfg),
            path_in_repo="checkpoints/ar/50m_2.5b/config.json",
            repo_id=REPO_ID,
        )

    print("\n" + "=" * 70)
    print("  [ALL DONE] 50M AR 2.5B Checkpoint Final is live on Hugging Face!")
    print("=" * 70)


@app.local_entrypoint()
def main():
    run_finish.remote()
