"""Unit tests for multi-shard `.bin` streaming, exact `(shard_idx, row_idx)` resume, and two-account HF relay."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from telos.models.transformer import TelosTransformer
from telos.training.dataloader import ShardedMemmapDataset
from telos.training.trainer_pytorch import UnifiedPyTorchTrainer
import run_bitmoe_v5e8_relay as relay


def _write_uint16_shard(path: Path, tokens: np.ndarray) -> None:
    arr = np.asarray(tokens, dtype=np.uint16)
    arr.tofile(path)
    meta = {"dtype": "uint16", "tokens": int(arr.size)}
    Path(str(path) + ".json").write_text(json.dumps(meta), encoding="utf-8")


def _make_shard_dir(root: Path, seq_len: int = 16) -> Path:
    shard_dir = root / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    # 3 shards with distinct row counts (5, 7, 4 -> 16 total rows) and unique deterministic values
    row_counts = [5, 7, 4]
    val = 10
    for idx, n_rows in enumerate(row_counts):
        mat = np.zeros((n_rows, seq_len), dtype=np.uint16)
        for r in range(n_rows):
            mat[r, :] = (val + r) % 250 + 4
        val += n_rows
        _write_uint16_shard(shard_dir / f"shard_{idx:04d}.bin", mat.reshape(-1))
    return shard_dir


def test_sharded_memmap_crosses_boundaries_and_wraps(tmp_path: Path):
    seq_len = 16
    shard_dir = _make_shard_dir(tmp_path, seq_len=seq_len)
    ds = ShardedMemmapDataset(shard_dir, seq_len=seq_len, vocab_size=256)

    assert ds.num_shards == 3
    assert ds.shard_rows == [5, 7, 4]
    assert len(ds) == 16

    # Start near end of shard 0 (row 3) and read 6 rows -> spans shard 0 (2 rows) + shard 1 (4 rows)
    batch, next_s, next_r = ds.get_batch_numpy(0, 3, total_batch=6, seq_len=seq_len)
    assert batch.shape == (6, seq_len)
    assert (next_s, next_r) == (1, 4)
    # Verify exact row values match continuous sequence 13..18 (+4 offset)
    expected_first = [((10 + i) % 250) + 4 for i in range(3, 9)]
    assert batch[:, 0].tolist() == expected_first

    # Wrap around from shard 2 row 2 across end back to shard 0
    batch2, next_s2, next_r2 = ds.get_batch_numpy(2, 2, total_batch=5, seq_len=seq_len)
    assert (next_s2, next_r2) == (0, 3)
    expected_wrap = [((10 + (i % 16)) % 250) + 4 for i in range(14, 19)]
    assert batch2[:, 0].tolist() == expected_wrap

    # Round-trip cursor <-> global_row
    for g in range(16):
        s, r = ds.cursor_from_global_row(g)
        assert ds.global_row_from_cursor(s, r) == g

    sig1 = ds.signature()
    sig2 = ds.signature()
    assert sig1 == sig2
    assert sig1["kind"] == "sharded_memmap_v1"
    ds.close()


def _tiny_bitmoe_cfg(data_dir: Path, ckpt_dir: Path, max_steps: int = 6, save_every: int = 3) -> dict:
    return {
        "paradigm": "ar",
        "model": {
            "vocab_size": 256,
            "d_model": 64,
            "n_layers": 2,
            "n_heads": 4,
            "n_kv_heads": 2,
            "max_seq_len": 16,
            "dropout": 0.0,
            "tied_embeddings": True,
            "is_causal": True,
            "use_bitlinear": True,
            "use_subln": True,
            "hybrid_swa_every": 2,
            "sliding_window": 8,
            "gdn_n_heads": 2,
            "gdn_head_dim": 32,
            "gdn_conv_kernel": 4,
            "gdn_chunk_size": 8,
            "ffn_type": "moe_relu2",
            "moe_num_experts": 4,
            "moe_top_k": 2,
            "moe_d_ff": 64,
            "moe_shared_d_ff": 64,
            "moe_capacity_factor": 1.5,
            "moe_z_loss_coef": 0.001,
            "mtp_num_heads": 1,
            "mtp_d_ff": 64,
            "mtp_loss_weight": 0.3,
        },
        "data": {
            "train_path": str(data_dir),
            "synthetic": False,
        },
        "training": {
            "precision": "fp32",
            "batch_size": 2,
            "gradient_accumulation": 2,
            "max_steps": max_steps,
            "warmup_steps": 2,
            "max_lr": 1e-3,
            "min_lr": 1e-4,
            "weight_decay": 0.01,
            "grad_clip": 1.0,
            "seed": 123,
            "fail_on_nonfinite_loss": True,
        },
        "checkpoint": {
            "checkpoint_dir": str(ckpt_dir),
            "save_every_steps": save_every,
            "optimizer_state_dtype": "bf16",
        },
    }


def test_trainer_multi_shard_exact_cursor_resume(tmp_path: Path):
    shard_dir = _make_shard_dir(tmp_path, seq_len=16)

    # Full uninterrupted run 1..6 (each step consumes 4 rows: 6 steps = 24 rows, crossing all 3 shards!)
    cfg_full = _tiny_bitmoe_cfg(shard_dir, tmp_path / "ckpt_full", max_steps=6, save_every=3)
    m_full = TelosTransformer(**cfg_full["model"])
    t_full = UnifiedPyTorchTrainer("ar", m_full, cfg_full, device_type="cpu")
    t_full.train()
    full_loss_step6 = t_full.last_logged_loss
    full_cursor_step6 = dict(t_full._shard_cursor)

    # Resume fresh trainer from step-3 checkpoint -> step 6
    ckpt3 = tmp_path / "ckpt_full" / "checkpoint_step_3.pt"
    assert ckpt3.exists()
    raw_ckpt = torch.load(ckpt3, map_location="cpu", weights_only=True)
    # 3 steps * 4 rows = 12 rows -> shard 0 (5) + shard 1 (7) = shard 2, row 0!
    assert raw_ckpt["shard_cursor"] == {"shard_idx": 2, "row_idx": 0}

    cfg_res = _tiny_bitmoe_cfg(shard_dir, tmp_path / "ckpt_res", max_steps=6, save_every=3)
    m_res = TelosTransformer(**cfg_res["model"])
    t_res = UnifiedPyTorchTrainer("ar", m_res, cfg_res, device_type="cpu")
    t_res.load_checkpoint(ckpt3)
    t_res.train(resume_step=3)

    assert t_res.global_step == 6
    assert t_res._shard_cursor == full_cursor_step6
    assert abs(t_res.last_logged_loss - full_loss_step6) < 1e-2


def test_two_account_relay_baton_and_continue(tmp_path: Path):
    shard_dir = _make_shard_dir(tmp_path, seq_len=16)
    cfg = _tiny_bitmoe_cfg(shard_dir, tmp_path / "unused_ckpt", max_steps=6, save_every=2)
    cfg_path = tmp_path / "tiny_relay.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    hub_dir = tmp_path / "mock_hf_hub"

    # Session 1 (Account A): forced early wall exit after first step (~0.0001 min)
    parser = relay.build_parser()
    args1 = parser.parse_args([
        "--config", str(cfg_path),
        "--data-dir", str(shard_dir),
        "--local-hub-dir", str(hub_dir),
        "--run-id", "alpha1-test",
        "--device", "cpu",
        "--no-spmd",
        "--stage-dir", str(tmp_path / "stage_a"),
        "--save-every-minutes", "0",
        "--max-wall-minutes", "0.00001",
    ])
    state1 = relay.run_relay(args1)
    assert state1["status"] == "in_progress"
    assert state1["session_index"] == 1
    assert 1 <= state1["global_step"] < 6
    assert state1["shard_cursor"] is not None
    # Verify local /tmp checkpoint was deleted after confirmed upload
    local_ckpts_a = list((tmp_path / "stage_a" / "alpha1-test" / "checkpoints").glob("*.pt"))
    assert local_ckpts_a == []

    # Session 2 (Account B): --continue from fresh staging dir finishes remaining steps to step 6
    args2 = parser.parse_args([
        "--config", str(cfg_path),
        "--data-dir", str(shard_dir),
        "--local-hub-dir", str(hub_dir),
        "--run-id", "alpha1-test",
        "--continue",
        "--device", "cpu",
        "--no-spmd",
        "--stage-dir", str(tmp_path / "stage_b"),
        "--save-every-minutes", "0",
        "--max-wall-minutes", "60",
    ])
    state2 = relay.run_relay(args2)
    assert state2["status"] == "completed"
    assert state2["session_index"] == 2
    assert state2["global_step"] == 6
    assert (hub_dir / "alpha1-test" / "relay_state.json").exists()
