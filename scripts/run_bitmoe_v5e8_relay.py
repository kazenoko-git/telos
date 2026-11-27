#!/usr/bin/env python3
"""Two-account hands-free Kaggle TPU v5e-8 HF Hub relay trainer for Telos Alpha 1.

Designed for back-to-back 9-hour Kaggle TPU v5e-8 sessions alternating between two
Kaggle accounts via a shared Hugging Face Hub model repository:
  - Initializes PyTorch/XLA SPMD (`torch_xla.runtime.use_spmd()`) BEFORE model
    construction so GSPMD FSDP=8 shards weights + optimizer states across all 8 chips.
  - Streams read-only multi-shard `.bin` token directories (e.g. `/kaggle/input/...`)
    with exact `(shard_idx, row_idx)` resume stored in both checkpoint and baton JSON.
  - Stages every checkpoint in `/tmp` (never `/kaggle/working`), uploads to HF Hub
    with exponential retry, atomically updates `<run_id>/relay_state.json`, and
    deletes the local `/tmp` checkpoint after confirmed HF commit.
  - Exits cleanly at `--max-wall-minutes` (default 510 = 8h30m, 30 min before Kaggle's
    9h hard kill) after uploading the final session checkpoint and baton.

Usage:
  # Session 1 (Account A — fresh start):
  python -u scripts/run_bitmoe_v5e8_relay.py \\
    --config configs/ar/bitmoe_2.25b_a377m_v5e8.yaml \\
    --data-dir /kaggle/input/telos-alpha1-shards \\
    --hf-repo Kazenowoko/telos-alpha1 \\
    --run-id alpha1-2.25b

  # Session 2+ (Account B, Account A, ... — automatic resume from HF baton):
  python -u scripts/run_bitmoe_v5e8_relay.py \\
    --config configs/ar/bitmoe_2.25b_a377m_v5e8.yaml \\
    --data-dir /kaggle/input/telos-alpha1-shards \\
    --hf-repo Kazenowoko/telos-alpha1 \\
    --run-id alpha1-2.25b \\
    --continue
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

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


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class RelayHubBackend:
    """Thin abstraction over Hugging Face Hub or a local directory (for tests/dry-runs)."""

    def __init__(self, hf_repo: str | None = None, local_hub_dir: Path | None = None):
        if not hf_repo and not local_hub_dir:
            raise ValueError("Provide either --hf-repo or --local-hub-dir")
        self.hf_repo = hf_repo
        self.local_hub_dir = Path(local_hub_dir) if local_hub_dir else None
        self._api = None
        if self.local_hub_dir is not None:
            self.local_hub_dir.mkdir(parents=True, exist_ok=True)
        else:
            from huggingface_hub import HfApi

            self._api = HfApi()
            self._api.create_repo(repo_id=self.hf_repo, repo_type="model", exist_ok=True)

    def upload_file(self, local_path: Path, path_in_repo: str, max_retries: int = 4) -> None:
        if self.local_hub_dir is not None:
            dest = self.local_hub_dir / path_in_repo
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(dest.suffix + ".tmp")
            shutil.copy2(local_path, tmp)
            os.replace(tmp, dest)
            return

        last_err: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                t0 = time.time()
                self._api.upload_file(
                    path_or_fileobj=str(local_path),
                    path_in_repo=path_in_repo,
                    repo_id=self.hf_repo,
                    repo_type="model",
                )
                dt_s = time.time() - t0
                size_mb = local_path.stat().st_size / (1024 ** 2)
                print(
                    f"  [HF Hub] Uploaded {local_path.name} ({size_mb:.1f} MiB) -> "
                    f"{self.hf_repo}/{path_in_repo} in {dt_s:.1f}s",
                    flush=True,
                )
                return
            except Exception as e:
                last_err = e
                backoff = min(60, 2 ** attempt)
                print(
                    f"  [HF Hub] Upload attempt {attempt}/{max_retries} failed ({e}); retrying in {backoff}s...",
                    flush=True,
                )
                time.sleep(backoff)
        raise RuntimeError(f"Failed to upload {local_path} after {max_retries} attempts: {last_err}")

    def upload_json(self, payload: dict[str, Any], path_in_repo: str, stage_dir: Path) -> None:
        tmp_json = stage_dir / f".{Path(path_in_repo).name}.tmp"
        tmp_json.parent.mkdir(parents=True, exist_ok=True)
        tmp_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        try:
            self.upload_file(tmp_json, path_in_repo)
        finally:
            tmp_json.unlink(missing_ok=True)

    def download_file(self, path_in_repo: str, dest_dir: Path) -> Path:
        dest_dir.mkdir(parents=True, exist_ok=True)
        if self.local_hub_dir is not None:
            src = self.local_hub_dir / path_in_repo
            if not src.exists():
                raise FileNotFoundError(f"Remote file not found in local hub: {src}")
            dest = dest_dir / Path(path_in_repo).name
            shutil.copy2(src, dest)
            return dest

        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id=self.hf_repo,
            filename=path_in_repo,
            repo_type="model",
            local_dir=str(dest_dir),
            force_download=True,
        )
        return Path(downloaded)

    def delete_file(self, path_in_repo: str) -> None:
        if self.local_hub_dir is not None:
            target = self.local_hub_dir / path_in_repo
            target.unlink(missing_ok=True)
            return
        try:
            self._api.delete_file(path_in_repo=path_in_repo, repo_id=self.hf_repo, repo_type="model")
        except Exception as e:
            print(f"  [HF Hub] Non-fatal prune skip for {path_in_repo}: {e}", flush=True)


def run_relay(args: argparse.Namespace) -> dict[str, Any]:
    _ensure_unbuffered_stdout()

    if args.device in ("xla", "tpu"):
        os.environ.setdefault("TPU_ACCELERATOR_TYPE", "v5e-8")
        clean_tpu_environment()
        if args.spmd:
            import torch_xla.runtime as xr

            xr.use_spmd()
            print("  [XLA] Enabled SPMD mode before model initialization.", flush=True)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg = copy.deepcopy(cfg)
    stage_dir = Path(args.stage_dir) / args.run_id
    stage_dir.mkdir(parents=True, exist_ok=True)

    # Configure dataset (multi-shard directory, single .bin, or synthetic for smoke tests)
    data_cfg = cfg.setdefault("data", {})
    if args.synthetic:
        data_cfg["synthetic"] = True
        data_cfg.pop("train_path", None)
    elif args.data_dir:
        data_cfg["synthetic"] = False
        data_cfg["train_path"] = str(Path(args.data_dir).resolve())

    if args.tokenizer:
        data_cfg["tokenizer"] = str(Path(args.tokenizer).resolve())

    # Operational training overrides (keep core config stable across --continue sessions)
    t_cfg = cfg.setdefault("training", {})
    t_cfg.setdefault("xla_cleanup_every", 250)
    t_cfg.setdefault("fail_on_nonfinite_loss", True)
    if args.max_steps is not None:
        t_cfg["max_steps"] = int(args.max_steps)
    if args.device == "cpu":
        t_cfg["precision"] = "fp32"
        t_cfg["xla_fsdp"] = False
        t_cfg["xla_dynamic_optimizer"] = False

    c_cfg = cfg.setdefault("checkpoint", {})
    c_cfg["checkpoint_dir"] = str(stage_dir / "checkpoints")
    if args.save_every_steps is not None:
        c_cfg["save_every_steps"] = int(args.save_every_steps)

    hub = RelayHubBackend(
        hf_repo=args.hf_repo,
        local_hub_dir=Path(args.local_hub_dir) if args.local_hub_dir else None,
    )
    baton_remote_path = f"{args.run_id}/relay_state.json"

    relay_state: dict[str, Any] = {
        "run_id": args.run_id,
        "session_index": 1,
        "global_step": 0,
        "max_steps": int(t_cfg["max_steps"]),
        "latest_checkpoint": None,
        "shard_cursor": None,
        "last_loss": None,
        "status": "in_progress",
        "history": [],
        "updated_utc": _utc_now_iso(),
    }

    resume_ckpt_local: Path | None = None
    resume_step = 0
    if args.continue_run:
        baton_local = hub.download_file(baton_remote_path, stage_dir / "baton")
        relay_state = json.loads(baton_local.read_text(encoding="utf-8"))
        if relay_state.get("status") == "completed":
            print(
                f"  [Relay] Run '{args.run_id}' is already marked completed at step {relay_state.get('global_step')}. Exiting.",
                flush=True,
            )
            return relay_state
        latest_remote = relay_state.get("latest_checkpoint")
        if not latest_remote:
            raise RuntimeError(f"relay_state.json for '{args.run_id}' has no latest_checkpoint to resume from.")
        resume_step = int(relay_state["global_step"])
        relay_state["session_index"] = int(relay_state.get("session_index", 1)) + 1
        print(
            f"  [Relay] --continue session #{relay_state['session_index']}: downloading {latest_remote} "
            f"(step {resume_step}, cursor={relay_state.get('shard_cursor')})...",
            flush=True,
        )
        resume_ckpt_local = hub.download_file(latest_remote, stage_dir / "resume")

    model = TelosTransformer(**cfg["model"])
    trainer = UnifiedPyTorchTrainer(cfg.get("paradigm", "ar"), model, cfg, device_type=args.device)

    if resume_ckpt_local is not None:
        trainer.load_checkpoint(resume_ckpt_local)
        if args.delete_local_after_upload:
            resume_ckpt_local.unlink(missing_ok=True)
            print(f"  [Disk] Deleted downloaded resume checkpoint {resume_ckpt_local.name} after load.", flush=True)

    def _on_checkpoint_saved(ckpt_file: Path, step: int, meta: dict[str, Any]) -> None:
        remote_ckpt = f"{args.run_id}/checkpoints/{ckpt_file.name}"
        hub.upload_file(ckpt_file, remote_ckpt)

        history = list(relay_state.get("history", []))
        if remote_ckpt not in history:
            history.append(remote_ckpt)

        relay_state.update({
            "global_step": int(step),
            "max_steps": int(t_cfg["max_steps"]),
            "latest_checkpoint": remote_ckpt,
            "shard_cursor": meta.get("shard_cursor"),
            "next_row": meta.get("next_row"),
            "last_loss": meta.get("last_loss"),
            "last_save_reason": meta.get("reason"),
            "status": "completed" if meta.get("reason") == "final" else "in_progress",
            "history": history,
            "updated_utc": _utc_now_iso(),
        })
        hub.upload_json(relay_state, baton_remote_path, stage_dir)
        print(
            f"  [Relay] Baton updated: step={step}/{t_cfg['max_steps']}, "
            f"cursor={meta.get('shard_cursor')}, status={relay_state['status']}",
            flush=True,
        )

        if args.delete_local_after_upload:
            ckpt_file.unlink(missing_ok=True)
            print(f"  [Disk] Deleted local checkpoint {ckpt_file.name} after confirmed upload.", flush=True)

        if args.keep_remote_checkpoints > 0 and len(history) > args.keep_remote_checkpoints:
            to_prune = history[: -args.keep_remote_checkpoints]
            for old_remote in to_prune:
                if old_remote != remote_ckpt:
                    hub.delete_file(old_remote)
            relay_state["history"] = history[-args.keep_remote_checkpoints :]
            hub.upload_json(relay_state, baton_remote_path, stage_dir)

    save_every_seconds = args.save_every_minutes * 60.0 if args.save_every_minutes > 0 else None
    max_wall_seconds = args.max_wall_minutes * 60.0 if args.max_wall_minutes > 0 else None

    print("=" * 76, flush=True)
    print(
        f"  TELOS ALPHA 1 RELAY — run_id={args.run_id} | session=#{relay_state['session_index']} | "
        f"start_step={resume_step} -> target={t_cfg['max_steps']}",
        flush=True,
    )
    print(
        f"  Periodic upload: every {args.save_every_minutes:.1f}m | "
        f"Clean wall exit: {args.max_wall_minutes:.1f}m | Stage dir: {stage_dir}",
        flush=True,
    )
    print("=" * 76, flush=True)

    trainer.train(
        resume_step=resume_step,
        save_every_seconds=save_every_seconds,
        max_wall_seconds=max_wall_seconds,
        on_checkpoint_saved=_on_checkpoint_saved,
    )
    return relay_state


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Two-account Kaggle TPU v5e-8 HF Hub relay trainer.")
    p.add_argument("--config", default="configs/ar/bitmoe_2.25b_a377m_v5e8.yaml")
    p.add_argument("--data-dir", default=None, help="Directory of uint16 .bin shards (e.g. /kaggle/input/telos-alpha1-shards).")
    p.add_argument("--tokenizer", default=None, help="Optional path to tokenizer_alpha1_32k.json.")
    p.add_argument("--synthetic", action="store_true", help="Use synthetic stream (smoke tests only).")
    p.add_argument("--hf-repo", default=None, help="Hugging Face model repo id (e.g. Kazenowoko/telos-alpha1).")
    p.add_argument("--local-hub-dir", default=None, help="Local directory mock hub (for offline tests).")
    p.add_argument("--run-id", default="alpha1-2.25b", help="Run namespace inside the HF repo.")
    p.add_argument("--continue", dest="continue_run", action="store_true", help="Resume from HF relay_state.json baton.")
    p.add_argument("--device", default="xla", choices=["xla", "tpu", "cpu", "cuda", "mps"])
    p.add_argument("--no-spmd", dest="spmd", action="store_false", help="Disable xr.use_spmd() (only for single-chip debugging).")
    p.set_defaults(spmd=True)
    p.add_argument("--stage-dir", default="/tmp/telos_relay", help="Ephemeral staging dir (keep in /tmp on Kaggle).")
    p.add_argument("--save-every-minutes", type=float, default=75.0, help="Periodic checkpoint + HF upload interval (minutes).")
    p.add_argument("--save-every-steps", type=int, default=None, help="Optional step-based save interval override.")
    p.add_argument("--max-wall-minutes", type=float, default=510.0, help="Clean session exit limit (default 510m = 8h30m).")
    p.add_argument("--max-steps", type=int, default=None, help="Optional total max_steps override.")
    p.add_argument("--keep-remote-checkpoints", type=int, default=3, help="Number of remote checkpoints to retain (0 = keep all).")
    p.add_argument("--keep-local", dest="delete_local_after_upload", action="store_false", help="Do not delete /tmp checkpoints after upload.")
    p.set_defaults(delete_local_after_upload=True)
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_relay(args)


if __name__ == "__main__":
    main()
