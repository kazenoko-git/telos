"""Regression coverage for trainer precision and optimizer safeguards."""
import pytest
import torch
from telos.models.transformer import TelosTransformer
from telos.training.trainer_pytorch import UnifiedPyTorchTrainer


def make_trainer(tmp_path, **training):
    model = TelosTransformer(vocab_size=32, d_model=16, n_layers=1, n_heads=2,
                             max_seq_len=8, is_causal=True)
    cfg = {
        'model': {'vocab_size': 32, 'seq_len': 8, 'd_model': 16},
        'training': {'precision': 'float32', 'max_steps': 2, 'warmup_steps': 0,
                     'batch_size': 2, 'max_lr': .001, **training},
        'data': {'synthetic': True},
        'checkpoint': {'checkpoint_dir': str(tmp_path)},
    }
    return model, cfg


@pytest.mark.parametrize('clip', [0, -1])
def test_disabled_clipping_keeps_nonzero_updates(tmp_path, monkeypatch, clip):
    model, cfg = make_trainer(tmp_path, grad_clip=clip)
    trainer = UnifiedPyTorchTrainer('ar', model, cfg, 'cpu')
    before = model.tok_embeddings.weight.detach().clone()
    def forbidden(*args, **kwargs):
        raise AssertionError('Disabled clipping must not touch gradients')
    monkeypatch.setattr(torch.nn.utils, 'clip_grad_norm_', forbidden)
    trainer.train()
    assert not torch.equal(before, model.tok_embeddings.weight)


@pytest.mark.parametrize('precision', ['bf16', 'bfloat16'])
def test_cuda_trainer_uses_native_bf16_guard(tmp_path, monkeypatch, precision):
    from telos.training import hardware
    model, cfg = make_trainer(tmp_path, precision=precision)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    calls = []
    monkeypatch.setattr(hardware, 'cuda_has_native_bf16', lambda device: calls.append(device) or False)
    monkeypatch.setattr(torch.cuda, 'is_bf16_supported', lambda: pytest.fail('Emulation query bypassed guard'))
    trainer = UnifiedPyTorchTrainer('ar', model, cfg, 'cuda')
    assert calls == [trainer.device]
    assert trainer.precision == 'fp16'
