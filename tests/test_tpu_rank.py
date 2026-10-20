"""Regression coverage for replica identity without requiring TPU hardware."""
import sys
import types

import numpy as np
import pytest
import torch

from telos.training import xla_utils
from telos.training.dataloader import get_global_targets_contiguous_pytorch
from telos.training.trainer_pytorch import UnifiedPyTorchTrainer


@pytest.fixture
def xla_runtime(monkeypatch):
    xla = types.ModuleType("torch_xla")
    core = types.ModuleType("torch_xla.core")
    xm = types.ModuleType("torch_xla.core.xla_model")
    runtime = types.ModuleType("torch_xla.runtime")
    xm.set_rng_state = lambda seed: None
    # Deliberately omit the removed xm.get_ordinal API.
    xla.core = core
    xla.runtime = runtime
    core.xla_model = xm
    for name, module in {
        "torch_xla": xla,
        "torch_xla.core": core,
        "torch_xla.core.xla_model": xm,
        "torch_xla.runtime": runtime,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(xla_utils, "get_xla_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(xla_utils, "get_xla_world_size", lambda: 8)
    monkeypatch.setattr(xla_utils, "get_xla_spmd_mesh", lambda: None)
    monkeypatch.setattr(xla_utils, "is_xla_master", lambda: runtime.global_ordinal() == 0)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    return runtime


def make_trainer():
    return UnifiedPyTorchTrainer(
        "ar", torch.nn.Linear(2, 2),
        {"model": {"vocab_size": 16, "seq_len": 2},
         "training": {"batch_size": 48, "gradient_accumulation": 1, "max_steps": 2}},
        device_type="xla",
    )


def test_eight_workers_keep_distinct_ranks_and_input_slices(xla_runtime):
    dataset = np.repeat(np.arange(768, dtype=np.int64)[:, None], 2, axis=1)
    covered = [set(), set()]
    for rank in range(8):
        xla_runtime.global_ordinal = lambda rank=rank: rank
        trainer = make_trainer()
        assert trainer.rank == rank
        for step in range(2):
            start = step * 384 + trainer.rank * 48
            batch, _ = get_global_targets_contiguous_pytorch(
                dataset, start, 48, 2, trainer.device, non_blocking=False,
            )
            rows = set(batch[:, 0].tolist())
            assert not covered[step].intersection(rows)
            covered[step].update(rows)
    assert covered == [set(range(384)), set(range(384, 768))]


@pytest.mark.parametrize("error", [AttributeError("rank API missing"), RuntimeError("runtime unavailable")])
def test_rank_lookup_failure_aborts_instead_of_becoming_zero(xla_runtime, monkeypatch, error):
    def fail():
        raise error
    xla_runtime.global_ordinal = fail
    # Let the constructor reach the separate rank lookup rather than failing master detection.
    monkeypatch.setattr(xla_utils, "is_xla_master", lambda: True)
    with pytest.raises(RuntimeError, match="Unable to determine TPU global rank") as raised:
        make_trainer()
    assert raised.value.__cause__ is error


@pytest.mark.parametrize("rank", [-1, 8])
def test_invalid_replica_rank_is_rejected(xla_runtime, rank):
    xla_runtime.global_ordinal = lambda: rank
    with pytest.raises(RuntimeError, match="outside world size"):
        make_trainer()
