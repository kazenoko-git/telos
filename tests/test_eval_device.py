import pytest
import torch
from telos.eval.runner import _place_native_model
from telos.training import xla_utils


def test_xla_uses_runtime_device(monkeypatch):
    target = object()
    class Model:
        def to(self, device):
            assert device is target
            return self
    monkeypatch.setattr(xla_utils, 'get_xla_device', lambda: target)
    monkeypatch.setattr(xla_utils, 'is_tpu_environment', lambda: True)
    model = Model()
    assert _place_native_model(model, 'pytorch', 'xla') is model
    assert _place_native_model(model, 'pytorch', 'auto') is model


def test_explicit_device_failure_propagates():
    class Model:
        def to(self, device):
            raise RuntimeError('placement failed')
    with pytest.raises(RuntimeError, match='placement failed'):
        _place_native_model(Model(), 'pytorch', 'cpu')


def test_mlx_rejects_incompatible_device():
    with pytest.raises(ValueError, match='MLX'):
        _place_native_model(object(), 'mlx', 'xla')
