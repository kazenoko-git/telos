"""Native Metal interruption/replay coverage for full MLX training state."""
from copy import deepcopy
import random

import numpy as np
import pytest

mx = pytest.importorskip('mlx.core')
if not mx.metal.is_available():
    pytest.skip('Metal is required', allow_module_level=True)

from mlx.utils import tree_flatten
from telos.models import MLXTelosTransformer
from telos.training.trainer_mlx import UnifiedMLXTrainer
from telos.training.checkpoint import load_mlx_state


@pytest.mark.parametrize('paradigm,precision', [(p, 'float32') for p in ('ar', 'mdlm', 'undlm', 'corosred')] + [('ar', 'bfloat16')])
@pytest.mark.parametrize('accumulation', [1, 2])
def test_native_mlx_interrupted_training_replays(tmp_path, paradigm, precision, accumulation):
    data = tmp_path / 'tokens.bin'
    (np.arange(17 * 8, dtype=np.uint16) % 27 + 4).tofile(data)
    cfg = {
        'paradigm': paradigm, '_backend': 'mlx', '_device': 'gpu', '_device_count': 1,
        'model': {'vocab_size': 32, 'seq_len': 8, 'd_model': 64, 'n_layers': 1,
                  'n_heads': 2, 'n_kv_heads': 2},
        'training': {'max_steps': 6, 'batch_size': 2, 'gradient_accumulation': accumulation,
                     'max_lr': .003, 'min_lr': .0001, 'warmup_steps': 2, 'seed': 17,
                     'precision': precision, 'async_eval': True},
        'data': {'train_path': str(data)},
        'checkpoint': {'checkpoint_dir': str(tmp_path / 'out'), 'save_every_steps': 2},
    }
    def make():
        mx.random.seed(19)
        model = MLXTelosTransformer(vocab_size=32, d_model=64, n_layers=1,
                                    n_heads=2, n_kv_heads=2, precision=precision,
                                    is_causal=paradigm in ('ar', 'corosred'),
                                    use_reliability_head=paradigm == 'corosred')
        mx.eval(model.parameters())
        return UnifiedMLXTrainer(paradigm, model, deepcopy(cfg))
    reference = make()
    reference.train()
    expected = load_mlx_state(tmp_path / 'out/checkpoint_final.resume.npz')
    resumed = make()
    resumed.load_checkpoint(tmp_path / 'out/checkpoint_step_2.resume.npz')
    saved = load_mlx_state(tmp_path / 'out/checkpoint_step_2.resume.npz')
    for (_, a), (_, b) in zip(tree_flatten(saved['model']), tree_flatten(resumed.model.parameters())):
        assert mx.array_equal(a, b).item()
    random.random(); np.random.random(7); mx.eval(mx.random.uniform(shape=(7,)))
    resumed.train()
    actual = load_mlx_state(tmp_path / 'out/checkpoint_final.resume.npz')
    # Stochastic embedding gradients use parallel GPU scatter reductions;
    # float32 continuation allows their rounding error while RNG is exact.
    # BF16 AR additionally exercises fp32 master-weight restoration.
    for key in ('mlx_rng', 'beta_buffer', 'master_params', 'optimizer', 'model'):
        left, right = tree_flatten(expected[key]), tree_flatten(actual[key])
        assert [k for k, _ in left] == [k for k, _ in right]
        for (_, a), (_, b) in zip(left, right):
            if isinstance(a, mx.array):
                assert a.dtype == b.dtype
                if key in ('mlx_rng', 'beta_buffer'):
                    assert mx.array_equal(a, b).item()
                else:
                    left_np, right_np = np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32))
                    delta = np.linalg.norm(left_np - right_np)
                    scale = max(float(np.linalg.norm(left_np)), 1e-8)
                    assert delta / scale < 1e-5, (key, delta / scale, np.max(np.abs(left_np-right_np)))
            else:
                assert a == b
    assert expected['host_rng'] == actual['host_rng']
    assert expected['next_row'] == actual['next_row']
    assert expected['beta_idx'] == actual['beta_idx']
