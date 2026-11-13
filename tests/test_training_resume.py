"""Interrupted runs must replay optimizer updates, data and randomness."""

from copy import deepcopy
import random

import numpy as np
import pytest
import torch

from telos.models import TelosTransformer
from telos.training.trainer_pytorch import UnifiedPyTorchTrainer
from telos.training.checkpoint import atomic_write, read_resume


def config(tmp_path, paradigm):
    data = tmp_path / 'tokens.bin'
    (np.arange(17 * 8, dtype=np.uint16) % 27 + 4).tofile(data)
    return {
        'paradigm': paradigm, '_backend': 'pytorch', '_device': 'cpu', '_device_count': 1,
        'model': {'vocab_size': 32, 'seq_len': 8, 'd_model': 32, 'n_layers': 1,
                  'n_heads': 2, 'n_kv_heads': 2},
        'training': {'max_steps': 6, 'batch_size': 2, 'gradient_accumulation': 2,
                     'max_lr': .003, 'min_lr': .0001, 'warmup_steps': 2, 'seed': 17,
                     'precision': 'float32'},
        'data': {'train_path': str(data)},
        'checkpoint': {'checkpoint_dir': str(tmp_path / 'out'), 'save_every_steps': 2},
    }


def trainer(cfg, paradigm):
    torch.manual_seed(19)
    return UnifiedPyTorchTrainer(paradigm, TelosTransformer(
        vocab_size=32, d_model=32, n_layers=1, n_heads=2, n_kv_heads=2,
        is_causal=paradigm in ('ar', 'corosred'), use_reliability_head=paradigm == 'corosred'),
        deepcopy(cfg), device_type='cpu')


def assert_tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_tree_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            assert_tree_equal(left, right)
    else:
        assert a == b


@pytest.mark.parametrize('paradigm', ['ar', 'mdlm', 'undlm', 'corosred'])
@pytest.mark.parametrize('synthetic', [False, True])
def test_interrupted_training_matches_uninterrupted(tmp_path, paradigm, synthetic):
    cfg = config(tmp_path, paradigm)
    cfg['data']['synthetic'] = synthetic
    reference = trainer(cfg, paradigm)
    reference.train()
    expected = read_resume(tmp_path / 'out/checkpoint_final.pt')

    # Simulate process loss after the checkpoint was safely committed.
    interrupted = trainer(cfg, paradigm)
    save = interrupted.save_checkpoint
    def stop_at_checkpoint(path, **kwargs):
        save(path, **kwargs)
        if interrupted.global_step == 2:
            raise KeyboardInterrupt
    interrupted.save_checkpoint = stop_at_checkpoint
    with pytest.raises(KeyboardInterrupt):
        interrupted.train()

    resumed = trainer(cfg, paradigm)
    resumed.load_checkpoint(tmp_path / 'out/checkpoint_step_2.pt')
    random.random(); np.random.random(7); torch.rand(7)
    resumed.train()  # No numeric step needed: the checkpoint is authoritative.
    actual = read_resume(tmp_path / 'out/checkpoint_final.pt')
    for key in ['model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict',
                'host_rng', 'torch_rng', 'next_row', 'global_step', 'beta_idx',
                'beta_counter', 'metric_tracker', 'routing_masks', 'routing_refresh']:
        assert_tree_equal(expected[key], actual[key])


def test_resume_rejects_changed_data_and_step_only(tmp_path):
    cfg = config(tmp_path, 'ar')
    original = trainer(cfg, 'ar')
    original.train()
    fresh = trainer(cfg, 'ar')
    fresh.load_checkpoint(tmp_path / 'out/checkpoint_step_2.pt')
    values = np.fromfile(cfg['data']['train_path'], dtype=np.uint16)
    values[5] += 1
    values.tofile(cfg['data']['train_path'])
    with pytest.raises(ValueError, match='Training data differs'):
        fresh.train()
    with pytest.raises(ValueError, match='step number alone'):
        trainer(cfg, 'ar').train(resume_step=2)


def test_atomic_write_preserves_previous_checkpoint(tmp_path):
    path = tmp_path / 'checkpoint.pt'
    path.write_bytes(b'last good checkpoint')
    def broken(temporary):
        from pathlib import Path
        Path(temporary).write_bytes(b'torn checkpoint')
        raise OSError('disk full')
    with pytest.raises(OSError):
        atomic_write(path, broken)
    assert path.read_bytes() == b'last good checkpoint'
    assert list(tmp_path.iterdir()) == [path]


def test_rejects_legacy_weights_only_checkpoint(tmp_path):
    path = tmp_path / 'legacy.pt'
    torch.save({'model_state_dict': {}}, path)
    with pytest.raises(ValueError, match='no complete training state'):
        read_resume(path)


@pytest.mark.parametrize('backend', ['cpu'])
def test_public_train_api_restores_saved_configuration(tmp_path, backend):
    from telos.train.cli import train
    result = train(paradigm='ar', d_model=128, n_layers=1, seq_len=8,
                   vocab_size=32, max_steps=4, batch_size=2, grad_accum=1,
                   warmup_steps=1, hardware=backend, devices=1, synthetic=True,
                   checkpoint_dir=str(tmp_path / 'first'), save_every=2)
    expected = read_resume(tmp_path / 'first/checkpoint_final.pt')
    # Resume without restating any architecture, data, objective or LR options.
    train(resume_checkpoint=tmp_path / 'first/checkpoint_step_2.pt',
          checkpoint_dir=str(tmp_path / 'resumed'), hardware=backend)
    actual = read_resume(tmp_path / 'resumed/checkpoint_final.pt')
    assert_tree_equal(expected['model_state_dict'], actual['model_state_dict'])
    assert_tree_equal(expected['optimizer_state_dict'], actual['optimizer_state_dict'])
    assert_tree_equal(expected['scheduler_state_dict'], actual['scheduler_state_dict'])


def test_resume_rejects_changed_schedule_and_distributed_state(tmp_path):
    cfg = config(tmp_path, 'ar')
    original = trainer(cfg, 'ar')
    original.train()
    changed = deepcopy(cfg)
    changed['training']['max_steps'] = 100
    with pytest.raises(ValueError, match='configuration mismatch'):
        trainer(changed, 'ar').load_checkpoint(tmp_path / 'out/checkpoint_step_2.pt')
    saved = read_resume(tmp_path / 'out/checkpoint_step_2.pt')
    saved['world_size'] = 8
    torch.save(saved, tmp_path / 'distributed.pt')
    with pytest.raises(ValueError, match='single training'):
        trainer(cfg, 'ar').load_checkpoint(tmp_path / 'distributed.pt')
