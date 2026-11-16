from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from telos.validation.state import assert_state_equal, compare_checkpoints
from telos.training.hardware import cuda_has_native_bf16, detect_cuda_profile
from telos.training.xla_utils import normalize_single_chip_tpu_environment


@pytest.mark.parametrize('change', ['nan', 'inf', 'missing', 'dtype', 'shape', 'value', 'container', 'scalar'])
def test_state_comparison_rejects_false_passes(change):
    a = {'model': {'w': torch.ones(2)}, 'optimizer': [1.0]}
    b = {'model': {'w': torch.ones(2)}, 'optimizer': [1.0]}
    if change in ('nan', 'inf'):
        b['model']['w'][0] = float(change)
    elif change == 'missing':
        b['model']['extra'] = torch.ones(1)
    elif change == 'dtype':
        b['model']['w'] = b['model']['w'].double()
    elif change == 'shape':
        b['model']['w'] = b['model']['w'].reshape(1, 2)
    elif change == 'value':
        b['model']['w'][0] = 2
    elif change == 'container':
        b['optimizer'] = (1.0,)
    else:
        a['optimizer'] = b['optimizer'] = [float('nan')]
    with pytest.raises(AssertionError):
        assert_state_equal(a, b)


def test_checkpoint_comparison_checks_full_state(tmp_path):
    a, b = tmp_path/'a.pt', tmp_path/'b.pt'
    state = {'model': {'w': torch.ones(2)}, 'optimizer': {'moment': torch.ones(1)}, 'rng': torch.arange(3)}
    torch.save(state, a); torch.save(state, b)
    assert compare_checkpoints(a, b)['tensor_leaves'] == 3
    state['optimizer']['moment'] += 1
    torch.save(state, b)
    with pytest.raises(AssertionError, match='moment'):
        compare_checkpoints(a, b)


@pytest.mark.parametrize('native', [False, True])
def test_cuda_auto_precision_excludes_emulation(monkeypatch, native):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 0)
    monkeypatch.setattr(torch.cuda, 'device', lambda index: nullcontext())
    monkeypatch.setattr(torch.cuda, 'is_bf16_supported', lambda including_emulation=True: including_emulation or native)
    assert cuda_has_native_bf16() == native
    from telos.posttrain.trainer import _torch_precision
    assert _torch_precision(torch.device('cuda'), 'auto') == (torch.bfloat16 if native else torch.float16)
    assert _torch_precision(torch.device('cuda'), 'bf16') == torch.bfloat16


def test_cuda_profile_checks_each_gpu_and_older_api(monkeypatch):
    selected = []
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 2)
    monkeypatch.setattr(torch.cuda, 'device', lambda index: selected.append(index) or nullcontext())
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda index: 'mock')
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda index: SimpleNamespace(total_memory=1024**3))
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda index: (8 if index == 0 else 7, 5))
    monkeypatch.setattr(torch.cuda, 'is_bf16_supported', lambda: True)
    assert detect_cuda_profile().precision == 'fp16'
    assert selected == [0, 1]


@pytest.fixture
def tpu_env(monkeypatch):
    for key in ('TPU_ACCELERATOR_TYPE', 'TPU_NAME', 'TPU_NUM_DEVICES', 'WORLD_SIZE',
                'PJRT_LOCAL_PROCESS_COUNT', 'TPU_WORKER_HOSTNAMES', 'TPU_PROCESS_BOUNDS',
                'TPU_CHIPS_PER_HOST_BOUNDS', 'TPU_CHIPS_PER_PROCESS_BOUNDS'):
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


def test_single_chip_topology_normalizes_before_detection(tpu_env):
    tpu_env.setenv('TPU_ACCELERATOR_TYPE', 'v5e-1')
    tpu_env.setenv('TPU_CHIPS_PER_HOST_BOUNDS', '2,2,1')
    assert normalize_single_chip_tpu_environment()
    import os
    assert os.environ['TPU_CHIPS_PER_HOST_BOUNDS'] == '1,1,1'
    from telos.training import xla_utils
    tpu_env.setattr(xla_utils, '_CACHED_XLA_DEVICE', None)
    assert xla_utils.get_xla_world_size() == 1


@pytest.mark.parametrize('marker', ['v5e-16', 'v5e-8', '', 'my-v5e-1-vm'])
def test_multi_or_unknown_topology_is_unchanged(tpu_env, marker):
    tpu_env.setenv('TPU_ACCELERATOR_TYPE', marker)
    tpu_env.setenv('TPU_CHIPS_PER_HOST_BOUNDS', '2,2,1')
    assert not normalize_single_chip_tpu_environment()
    import os
    assert os.environ['TPU_CHIPS_PER_HOST_BOUNDS'] == '2,2,1'


@pytest.mark.parametrize('key,value', [('WORLD_SIZE', '8'), ('TPU_NUM_DEVICES', '8'),
    ('PJRT_LOCAL_PROCESS_COUNT', '2'), ('TPU_PROCESS_BOUNDS', '2,1,1'),
    ('TPU_WORKER_HOSTNAMES', 'a,b'), ('TPU_CHIPS_PER_HOST_BOUNDS', '4,4,1')])
def test_single_chip_conflicting_metadata_fails(tpu_env, key, value):
    tpu_env.setenv('TPU_ACCELERATOR_TYPE', 'v5e-1')
    tpu_env.setenv(key, value)
    with pytest.raises(RuntimeError, match='refusing topology override'):
        normalize_single_chip_tpu_environment()


def test_nf4_recompute_avoids_retained_dense_weight_with_exact_gradient():
    from telos.posttrain.adapters import NF4Linear
    torch.manual_seed(19)
    layer = NF4Linear(torch.nn.Linear(64, 64))
    source = torch.randn(2, 8, 64)
    results = []
    for recompute in (False, True):
        x = source.clone().requires_grad_()
        saved = []
        with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
            y = layer(x) if recompute else F.linear(x, layer.dequantize(), layer.bias)
            loss = y.square().sum()
        loss.backward()
        results.append((loss.detach(), x.grad, sum(t.numel()*t.element_size() for t in saved if t.shape == (64, 64))))
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    torch.testing.assert_close(results[0][1], results[1][1], rtol=0, atol=0)
    assert results[0][2] > 0 and results[1][2] == 0
    assert all(not p.requires_grad for p in layer.parameters())


def test_xla_parent_normalizes_before_launch(tpu_env):
    import os
    import sys
    import types
    from telos.posttrain.trainer import _posttrain_xla
    tpu_env.setenv('TPU_ACCELERATOR_TYPE', 'v5e-1')
    tpu_env.setenv('TPU_CHIPS_PER_HOST_BOUNDS', '2,2,1')
    launched = []
    def launch(worker, args):
        assert os.environ['TPU_CHIPS_PER_HOST_BOUNDS'] == '1,1,1'
        assert os.environ['TPU_CHIPS_PER_PROCESS_BOUNDS'] == '1,1,1'
        launched.append(True)
    tpu_env.setitem(sys.modules, 'torch_xla', types.SimpleNamespace(launch=launch))
    _posttrain_xla({'output': '/unused'})
    assert launched == [True]


def test_validation_benchmark_advances_schedule_and_weights(tmp_path):
    from telos.validation.hardware import benchmark
    result = benchmark(SimpleNamespace(hardware='cpu', output=tmp_path, steps=8,
                                      compiled=False, seq_len=8))
    assert result['global_step'] == result['scheduler_step'] == 8
    assert result['max_parameter_change'] > 0
    assert result['measured_updates'] == 3
    assert result['input_tokens'] == 3*2*8
    assert result['tokens_per_second'] == result['input_tokens']/result['elapsed_seconds']


def test_nf4_uses_xla_checkpoint_wrapper(monkeypatch):
    import sys
    import types
    from telos.posttrain.adapters import _checkpoint_nf4
    calls = []
    def checkpoint(function, inputs, **options):
        calls.append(options)
        return 'xla result'
    monkeypatch.setitem(sys.modules, 'torch_xla.utils.checkpoint', types.SimpleNamespace(checkpoint=checkpoint))
    assert _checkpoint_nf4(lambda x: x, SimpleNamespace(device=SimpleNamespace(type='xla'))) == 'xla result'
    assert calls == [{'use_reentrant': True}]


@pytest.mark.parametrize('mode', ['full', 'lora', 'qlora'])
def test_dpo_frozen_reference_and_parameter_invariants(tmp_path, mode):
    from telos.validation.hardware import prepare, training, adapter_invariants
    prepare(tmp_path)
    args = SimpleNamespace(output=tmp_path, case='dpo-'+mode, hardware='cpu',
                           steps=8, restart=False)
    training(args)
    assert adapter_invariants(tmp_path, args.case)['status'] == 'PASS'
    if mode == 'qlora':
        path = tmp_path/'train/checkpoint_final.resume.pt'
        state = torch.load(path, weights_only=True)
        key = next(k for k in state['model'] if k.endswith('packed_weight'))
        state['model'][key][0] ^= 1
        torch.save(state, path)
        with pytest.raises(AssertionError, match='packed_weight'):
            adapter_invariants(tmp_path, args.case)
