#!/usr/bin/env python3
"""Single-chip TPU v5e-1 canary runner for Telos v1.9.2 (BitMoE + GDN/SWA + MTP).

Runs a fast two-phase hardware & numerical verification without waiting in the v5e-8 queue:
  Phase 1: Train steps 1..N on configs/ar/bitmoe_canary_v5e1.yaml with fail_on_nonfinite_loss
           checked post-mark_step (zero mid-step XLA graph splits) and save checkpoint_step_K.pt
           to /tmp (with optional HF Hub upload + local delete + remote re-download).
  Phase 2: Construct a fresh TelosTransformer + UnifiedPyTorchTrainer, reload
           checkpoint_step_K.pt, verify finite weights + FP32 optimizer restoration,
           resume steps K+1..N, and assert numerical continuity against Phase 1.

Exits non-zero on ANY non-finite loss, non-finite parameter, or resume mismatch.

Usage:
  # On Kaggle / Colab TPU v5e-1:
  python -u scripts/run_bitmoe_v5e1_canary.py --device xla

  # With HF Hub round-trip test (/tmp -> HF upload -> delete -> download -> reload):
  python -u scripts/run_bitmoe_v5e1_canary.py --device xla --hf-repo Kazenowoko/telos
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import shutil
import sys
import time
from pathlib import Path

import torch
import yaml

from telos.models.transformer import TelosTransformer
from telos.training.trainer_pytorch import UnifiedPyTorchTrainer
from telos.training.xla_utils import clean_tpu_environment


def _ensure_unbuffered_stdout() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass


def _report_disk_headroom(stage_dir: Path) -> None:
    stage_dir.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(stage_dir)
    free_gb = usage.free / (1024 ** 3)
    total_gb = usage.total / (1024 ** 3)
    print(f"  [Disk] Staging dir {stage_dir}: {free_gb:.1f} GiB free / {total_gb:.1f} GiB total", flush=True)


def _report_xla_hbm(tag: str) -> None:
    try:
        import torch_xla.core.xla_model as xm
        info = xm.get_memory_info(xm.xla_device())
        used_gb = info.get("bytes_used", 0) / (1024 ** 3)
        peak_gb = info.get("peak_bytes_used", info.get("bytes_limit", 0)) / (1024 ** 3)
        limit_gb = info.get("bytes_limit", 0) / (1024 ** 3)
        print(f"  [HBM {tag}] used={used_gb:.2f} GiB | peak={peak_gb:.2f} GiB | limit={limit_gb:.2f} GiB", flush=True)
    except Exception:
        pass


def _assert_all_params_finite(model: torch.nn.Module, label: str) -> None:
    for name, p in model.named_parameters():
        if not torch.isfinite(p.detach().cpu()).all():
            raise RuntimeError(f"CANARY FAILED [{label}]: non-finite parameter detected in '{name}'")


def _upload_and_delete_hf(local_path: Path, repo_id: str, path_in_repo: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    t0 = time.time()
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="model",
    )
    dt = time.time() - t0
    size_mb = local_path.stat().st_size / (1024 ** 2)
    print(f"  [HF Hub] Uploaded {local_path.name} ({size_mb:.1f} MiB) to {repo_id}/{path_in_repo} in {dt:.1f}s", flush=True)
    local_path.unlink()
    print(f"  [Disk] Deleted local copy {local_path} after confirmed HF commit.", flush=True)


def _download_from_hf(repo_id: str, path_in_repo: str, dest_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=path_in_repo,
        repo_type="model",
        local_dir=str(dest_dir),
    )
    return Path(downloaded)


def main() -> None:
    _ensure_unbuffered_stdout()

    parser = argparse.ArgumentParser(description="Run Telos v1.9.2 single-chip TPU v5e-1 canary.")
    parser.add_argument("--config", default="configs/ar/bitmoe_canary_v5e1.yaml", help="Path to YAML config.")
    parser.add_argument("--device", default="xla", choices=["xla", "tpu", "cpu", "cuda", "mps"], help="Target device.")
    parser.add_argument("--stage-dir", default="/tmp/telos_v5e1_canary", help="Ephemeral checkpoint directory.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override total max_steps (default from YAML: 20).")
    parser.add_argument("--save-every", type=int, default=None, help="Override save_every_steps (default from YAML: 10).")
    parser.add_argument("--hf-repo", default=None, help="Optional HF repo_id to test /tmp -> HF upload -> delete -> reload.")
    args = parser.parse_args()

    if args.device in ("xla", "tpu"):
        os.environ.setdefault("TPU_ACCELERATOR_TYPE", "v5e-1")
        clean_tpu_environment()

    with open(args.config, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    stage_dir = Path(args.stage_dir)
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    _report_disk_headroom(stage_dir)

    total_steps = args.max_steps if args.max_steps is not None else int(base_cfg["training"]["max_steps"])
    save_step = args.save_every if args.save_every is not None else int(base_cfg["checkpoint"]["save_every_steps"])
    if not 1 <= save_step < total_steps:
        raise ValueError(f"save_every ({save_step}) must be strictly between 1 and max_steps ({total_steps})")

    # Build Phase 1 config (full run target = total_steps, checkpoint saved at save_step)
    cfg1 = copy.deepcopy(base_cfg)
    cfg1["training"]["max_steps"] = total_steps
    cfg1["training"]["fail_on_nonfinite_loss"] = True
    if args.device == "cpu":
        cfg1["training"]["precision"] = "fp32"
        cfg1["training"]["xla_dynamic_optimizer"] = False
    cfg1["checkpoint"]["checkpoint_dir"] = str(stage_dir)
    cfg1["checkpoint"]["save_every_steps"] = save_step

    print("=" * 72, flush=True)
    print(f"  TELOS v1.9.2 CANARY — Phase 1: Train 1..{total_steps} (checkpoint at step {save_step})", flush=True)
    print("=" * 72, flush=True)

    model1 = TelosTransformer(**cfg1["model"])
    trainer1 = UnifiedPyTorchTrainer(cfg1.get("paradigm", "ar"), model1, cfg1, device_type=args.device)
    trainer1.train()
    p1_final_loss = getattr(trainer1, "last_logged_loss", float("nan"))
    if not math.isfinite(p1_final_loss):
        raise RuntimeError(f"CANARY FAILED [phase1]: final logged loss is non-finite ({p1_final_loss})")
    _assert_all_params_finite(model1, "phase1_end")
    _report_xla_hbm("after_phase1")

    ckpt_path = stage_dir / f"checkpoint_step_{save_step}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Expected canary checkpoint missing: {ckpt_path}")

    ckpt_mb = ckpt_path.stat().st_size / (1024 ** 2)
    print(f"  [Canary] Verified Phase 1 checkpoint: {ckpt_path} ({ckpt_mb:.1f} MiB, all params finite)", flush=True)

    if args.hf_repo:
        remote_name = f"canary/{ckpt_path.name}"
        _upload_and_delete_hf(ckpt_path, args.hf_repo, remote_name)
        ckpt_path = _download_from_hf(args.hf_repo, remote_name, stage_dir / "hf_reload")
        print(f"  [HF Hub] Re-downloaded checkpoint to {ckpt_path}", flush=True)

    print("=" * 72, flush=True)
    print(f"  TELOS v1.9.2 CANARY — Phase 2: Fresh Reload at step {save_step} -> {total_steps}", flush=True)
    print("=" * 72, flush=True)

    cfg2 = copy.deepcopy(cfg1)
    model2 = TelosTransformer(**cfg2["model"])
    trainer2 = UnifiedPyTorchTrainer(cfg2.get("paradigm", "ar"), model2, cfg2, device_type=args.device)
    trainer2.load_checkpoint(ckpt_path)
    _assert_all_params_finite(model2, "phase2_reloaded")

    # Verify BF16-compressed optimizer states were upcast cleanly back to param dtype
    for param, st in trainer2.optimizer.state.items():
        if isinstance(st, dict) and "exp_avg" in st:
            assert st["exp_avg"].dtype == param.dtype, f"exp_avg dtype mismatch: {st['exp_avg'].dtype} vs {param.dtype}"
            assert st["exp_avg_sq"].dtype == param.dtype, f"exp_avg_sq dtype mismatch: {st['exp_avg_sq'].dtype} vs {param.dtype}"

    trainer2.train(resume_step=save_step)
    p2_final_loss = getattr(trainer2, "last_logged_loss", float("nan"))
    if not math.isfinite(p2_final_loss):
        raise RuntimeError(f"CANARY FAILED [phase2]: final logged loss is non-finite ({p2_final_loss})")
    _assert_all_params_finite(model2, "phase2_end")
    _report_xla_hbm("after_phase2")

    assert trainer2.global_step == total_steps, f"Expected final step {total_steps}, got {trainer2.global_step}"

    diff = abs(p1_final_loss - p2_final_loss)
    print(
        f"  [Continuity] Step {total_steps}: phase1_loss={p1_final_loss:.4f} vs "
        f"phase2_loss={p2_final_loss:.4f} (diff={diff:.4e})",
        flush=True,
    )
    if diff > 0.1:
        raise RuntimeError(f"CANARY FAILED: resume loss diverged at step {total_steps} (diff={diff:.4e})")

    print("=" * 72, flush=True)
    print(
        f"  CANARY PASSED: {save_step}->{total_steps} finite loss, finite params, and reload continuity verified on {args.device.upper()}.",
        flush=True,
    )
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
