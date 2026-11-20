"""Mocked XLA tests for persistent cache, fused all-reduce, RNG gating, and checkpointing."""
from __future__ import annotations

import sys
import types
import torch

from telos.models.transformer import TelosTransformer
from telos.training.xla_utils import configure_xla_persistent_cache


def test_configure_xla_persistent_cache_calls_initialize_cache(tmp_path, monkeypatch):
    calls = []
    fake_xr = types.SimpleNamespace(initialize_cache=lambda path, readonly=False: calls.append((path, readonly)))
    fake_xla = types.ModuleType("torch_xla")
    fake_xla.runtime = fake_xr
    monkeypatch.setitem(sys.modules, "torch_xla", fake_xla)
    monkeypatch.setitem(sys.modules, "torch_xla.runtime", fake_xr)

    cache_dir = tmp_path / "xla_cache"
    configured = configure_xla_persistent_cache(cache_dir)
    assert configured == str(cache_dir)
    assert cache_dir.is_dir()
    assert calls == [(str(cache_dir), False)]


def test_configure_xla_persistent_cache_noop_when_unset(monkeypatch):
    monkeypatch.delenv("XLA_PERSISTENT_CACHE_PATH", raising=False)
    assert configure_xla_persistent_cache(None) is None


def test_xla_trainer_skips_per_step_rng_on_deterministic_ar_and_fuses_all_reduce(monkeypatch, tmp_path):
    from telos.training.trainer_pytorch import UnifiedPyTorchTrainer

    events = []
    fake_xm = types.SimpleNamespace(
        xla_device=lambda: torch.device("cpu"),
        set_rng_state=lambda seed, *args: events.append(("set_rng_state", seed)),
        get_rng_state=lambda *args: 0,
        reduce_gradients=lambda opt: events.append(("reduce_gradients",)),
        mark_step=lambda: events.append(("mark_step",)),
        rendezvous=lambda tag: events.append(("rendezvous", tag)),
        all_reduce=lambda op, tensors, scale=1.0: (events.append(("all_reduce", len(tensors))), [t * scale for t in tensors])[1],
        xworld_size=lambda: 2,
        REDUCE_SUM="sum",
    )
    fake_xr = types.SimpleNamespace(
        global_ordinal=lambda: 0,
        world_size=lambda: 2,
        process_index=lambda: 0,
        is_spmd=lambda: False,
        initialize_cache=lambda path, readonly=False: events.append(("init_cache", path)),
    )
    fake_xla = types.ModuleType("torch_xla")
    fake_xla_core = types.ModuleType("torch_xla.core")
    fake_xla_core.xla_model = fake_xm
    fake_xla.core = fake_xla_core
    fake_xla.runtime = fake_xr

    monkeypatch.setitem(sys.modules, "torch_xla", fake_xla)
    monkeypatch.setitem(sys.modules, "torch_xla.core", fake_xla_core)
    monkeypatch.setitem(sys.modules, "torch_xla.core.xla_model", fake_xm)
    monkeypatch.setitem(sys.modules, "torch_xla.runtime", fake_xr)

    import telos.training.xla_utils as xu
    monkeypatch.setattr(xu, "_CACHED_XLA_DEVICE", None)
    monkeypatch.setattr(xu, "_CACHED_XLA_WORLD_SIZE", None)
    monkeypatch.setattr(xu, "_CACHED_IS_MASTER", None)
    monkeypatch.setattr(xu, "_CACHED_SPMD_MESH", None)

    # 1. Deterministic AR (dropout=0): per-step set_rng_state should NOT run during train loop.
    ar_model = TelosTransformer(vocab_size=64, d_model=32, n_layers=1, n_heads=2, max_seq_len=16, is_causal=True, dropout=0.0)
    ar_cfg = {
        "model": {"vocab_size": 64, "d_model": 32, "n_layers": 1, "n_heads": 2, "seq_len": 16, "dropout": 0.0},
        "training": {"batch_size": 2, "gradient_accumulation": 2, "max_steps": 2, "precision": "fp32",
                     "xla_cache_dir": str(tmp_path / "cache"), "xla_mark_step_per_microbatch": True},
        "data": {"synthetic": True},
        "checkpoint": {"save_every_steps": 0, "dir": str(tmp_path / "ckpt_ar")},
    }
    ar_trainer = UnifiedPyTorchTrainer("ar", ar_model, ar_cfg, device_type="xla")
    events.clear()
    ar_trainer.train()
    assert not any(e[0] == "set_rng_state" for e in events)
    # 2 steps * (1 microbatch mark_step + 1 step mark_step) + 1 final mark_step = 5
    assert sum(1 for e in events if e[0] == "mark_step") >= 4

    # 2. COROSred unified: all_reduce must happen BEFORE mark_step on cadence step 1.
    crsr_model = TelosTransformer(
        vocab_size=64, d_model=32, n_layers=1, n_heads=2, max_seq_len=16,
        is_causal=True, use_reliability_head=True,
    )
    crsr_cfg = {
        "model": {"vocab_size": 64, "d_model": 32, "n_layers": 1, "n_heads": 2, "seq_len": 16},
        "training": {"batch_size": 2, "gradient_accumulation": 1, "max_steps": 1, "precision": "fp32"},
        "corosred": {"unified": True, "cadence": 1},
        "data": {"synthetic": True},
        "checkpoint": {"save_every_steps": 0, "dir": str(tmp_path / "ckpt_crsr")},
    }
    crsr_trainer = UnifiedPyTorchTrainer("corosred", crsr_model, crsr_cfg, device_type="xla")
    # Simulate accelerator 0D tensor metrics (xla_device is mocked to CPU in this unit test).
    orig_exec = crsr_trainer._execute_microbatch
    def _exec_with_tensor_metrics(batch_seqs, compute_metrics=False):
        loss, metrics = orig_exec(batch_seqs, compute_metrics=compute_metrics)
        metrics["lrh_bal_acc"] = torch.tensor(0.6)
        metrics["lrh_acc"] = torch.tensor(0.7)
        return loss, metrics
    monkeypatch.setattr(crsr_trainer, "_execute_microbatch", _exec_with_tensor_metrics)
    events.clear()
    crsr_trainer.train()
    ar_idx = next(i for i, e in enumerate(events) if e[0] == "all_reduce")
    ms_idx = next(i for i, e in enumerate(events) if e[0] == "mark_step")
    assert ar_idx < ms_idx, f"Expected all_reduce before mark_step, got {events}"
