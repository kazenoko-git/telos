"""Projection/CE chunking must preserve target rounding and small gradients."""
import numpy as np
import pytest

mx = pytest.importorskip('mlx.core')
if not mx.metal.is_available():
    pytest.skip('Metal required', allow_module_level=True)

from telos.training.mlx_losses import cross_entropy
from telos.training.mlx_vocab_loss import (
    tiled_projection_cross_entropy,
    token_chunked_projection_cross_entropy,
)


@pytest.mark.parametrize('function', [tiled_projection_cross_entropy, token_chunked_projection_cross_entropy])
@pytest.mark.parametrize('dtype', [mx.float32, mx.bfloat16])
@pytest.mark.parametrize('chunk_size', [7, 64])
def test_loss_and_gradients_match_full_head(function, dtype, chunk_size):
    mx.random.seed(71)
    hidden = mx.random.normal((2, 5, 64)).astype(dtype)
    weight = (mx.random.normal((257, 64)) * .125).astype(dtype)
    targets = mx.array([[0, 1, 63, 64, 256], [31, 128, 200, 255, 7]])
    # Arbitrary cotangents, including zero and negative weights.
    cotangent = mx.array([[0., .1, -.2, .3, 1.], [.5, -1., .7, .9, 0.]])
    def full(x, w):
        return mx.sum(cross_entropy(x @ w.T, targets, reduction='none') * cotangent)
    def chunked(x, w):
        return mx.sum(function(x, w, targets, chunk_size=chunk_size) * cotangent)
    expected, reference = mx.value_and_grad(full, argnums=(0, 1))(hidden, weight)
    actual, gradients = mx.compile(mx.value_and_grad(chunked, argnums=(0, 1)))(hidden, weight)
    mx.eval(expected, actual, reference, gradients)
    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=2e-6, atol=2e-6)
    for a, b in zip(reference, gradients):
        a, b = a.astype(mx.float32), b.astype(mx.float32)
        relative = mx.sqrt(mx.sum((a - b) ** 2) / mx.maximum(mx.sum(a * a), 1e-30)).item()
        assert relative < (2e-5 if dtype == mx.float32 else .015)
        assert mx.all(mx.isfinite(b)).item()


@pytest.mark.parametrize('function', [tiled_projection_cross_entropy, token_chunked_projection_cross_entropy])
def test_bf16_target_logit_uses_same_rounding_as_partition(function):
    hidden = mx.array([[1.0625, 0.]], dtype=mx.bfloat16)
    weight = mx.zeros((257, 2), dtype=mx.bfloat16).at[0, 0].add(37.75)
    targets = mx.array([0])
    expected = cross_entropy(hidden @ weight.T, targets, reduction='none')
    actual = function(hidden, weight, targets, chunk_size=32)
    assert mx.all(actual >= 0).item()
    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize('function', [tiled_projection_cross_entropy, token_chunked_projection_cross_entropy])
def test_confident_bf16_target_gradient_does_not_round_to_zero(function):
    hidden = mx.array([[1., 0.]], dtype=mx.bfloat16)
    weight = mx.zeros((257, 2), dtype=mx.bfloat16).at[0, 0].add(14.)
    targets = mx.array([0])
    reference = mx.grad(lambda w: mx.sum(cross_entropy(hidden @ w.T, targets, reduction='none')))(weight)
    actual = mx.grad(lambda w: mx.sum(function(hidden, w, targets, chunk_size=32)))(weight)
    assert reference[0, 0].item() < 0
    assert actual[0, 0].item() < 0
    np.testing.assert_allclose(np.array(actual.astype(mx.float32)), np.array(reference.astype(mx.float32)), rtol=.015, atol=2e-7)


@pytest.mark.parametrize('function', [tiled_projection_cross_entropy, token_chunked_projection_cross_entropy])
def test_chunk_contract_validation(function):
    h, w, t = mx.ones((2, 4)), mx.ones((7, 4)), mx.zeros((2,), dtype=mx.int32)
    with pytest.raises(ValueError, match='positive'):
        function(h, w, t, chunk_size=0)
    with pytest.raises(ValueError, match='token dimensions'):
        function(h, w, t[:1])
    with pytest.raises(ValueError, match='integer'):
        function(h, w, t.astype(mx.float32))


def test_token_chunk_axis_is_rejected_on_torch():
    from telos.train.cli import train
    with pytest.raises(ValueError, match='only on MLX'):
        train(hardware='cpu', params='1M', tokens=32, d_model=64, n_layers=1,
              fused_vocab_loss=True, fused_vocab_chunk_axis='token')


@pytest.mark.parametrize('axis', ['vocab', 'token'])
@pytest.mark.parametrize('dtype', ['float32', 'bf16'])
@pytest.mark.parametrize('checkpoint', [False, True])
def test_ar_chunking_preserves_context_masks_and_parameter_gradients(axis, dtype, checkpoint):
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from telos.models.mlx_transformer import MLXTelosTransformer
    from telos.diffusion.ar import ar_loss_fn_mlx
    mx.random.seed(29)
    kwargs = dict(vocab_size=97, d_model=64, n_layers=2, n_heads=1,
                  precision=dtype, is_causal=True, use_grad_checkpoint=checkpoint)
    reference = MLXTelosTransformer(**kwargs)
    candidate = MLXTelosTransformer(**kwargs)
    candidate.load_weights(tree_flatten(reference.parameters()))
    candidate.fused_vocab_loss_enabled = True
    candidate.fused_vocab_chunk_axis = axis
    candidate.fused_vocab_chunk_size = 7
    tokens = mx.array([[0, 1, 2, 3, 0, 1, 2, 3, 0],
                       [3, 4, 8, 20, 1, 7, 3, 9, 96],
                       [3, 7, 6, 11, 19, 4, 8, 12, 24]])
    special = mx.array([True] * 4 + [False] * 93)
    def calculate(model):
        return nn.value_and_grad(model, lambda m: ar_loss_fn_mlx(m, tokens, 97, special)[0])(model)
    expected, expected_grad = calculate(reference)
    actual, actual_grad = calculate(candidate)
    mx.eval(expected, actual, expected_grad, actual_grad)
    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=2e-6, atol=3e-6)
    assert [k for k, _ in tree_flatten(expected_grad)] == [k for k, _ in tree_flatten(actual_grad)]
    for (name, a), (_, b) in zip(tree_flatten(expected_grad), tree_flatten(actual_grad)):
        assert a.dtype == b.dtype and a.shape == b.shape
        a, b = a.astype(mx.float32), b.astype(mx.float32)
        error = mx.sqrt(mx.sum((a-b)**2) / mx.maximum(mx.sum(a*a), 1e-30)).item()
        assert error < (2e-5 if dtype == 'float32' else .015), (name, error)


@pytest.mark.parametrize('axis', ['vocab', 'token'])
def test_chunked_pretraining_resumes_exactly(tmp_path, axis):
    import copy
    from mlx.utils import tree_flatten
    from telos.models.mlx_transformer import MLXTelosTransformer
    from telos.training.trainer_mlx import UnifiedMLXTrainer
    from telos.training.checkpoint import read_resume
    data = tmp_path / 'tokens.bin'
    (np.arange(80 * 16, dtype=np.uint16) % 93 + 4).tofile(data)
    cfg = dict(paradigm='ar', model=dict(vocab_size=97, d_model=64, n_layers=1,
               n_heads=1, n_kv_heads=1, seq_len=16, is_causal=True),
               training=dict(max_steps=4, warmup_steps=1, batch_size=2, gradient_accumulation=2,
                   precision='bf16', max_lr=.001, min_lr=.0001, seed=17,
                   fused_vocab_loss=True, fused_vocab_chunk_axis=axis, fused_vocab_chunk_size=7),
               data=dict(train_path=str(data)),
               checkpoint=dict(checkpoint_dir=str(tmp_path / 'control'), save_every_steps=2))
    def trainer(config):
        mx.random.seed(17)
        model = MLXTelosTransformer(vocab_size=97, d_model=64, n_layers=1, n_heads=1, is_causal=True)
        return UnifiedMLXTrainer('ar', model, copy.deepcopy(config))
    trainer(cfg).train()
    expected = read_resume(tmp_path / 'control/checkpoint_final.resume.npz')
    cfg['checkpoint']['checkpoint_dir'] = str(tmp_path / 'resumed')
    resumed = trainer(cfg)
    resumed.load_checkpoint(tmp_path / 'control/checkpoint_step_2.resume.npz')
    resumed.train()
    actual = read_resume(tmp_path / 'resumed/checkpoint_final.resume.npz')
    for group in ('model', 'optimizer', 'master_params', 'mlx_rng'):
        a, b = dict(tree_flatten(expected[group])), dict(tree_flatten(actual[group]))
        assert a.keys() == b.keys()
        for key in a:
            assert a[key].dtype == b[key].dtype and a[key].shape == b[key].shape
            assert mx.array_equal(a[key], b[key]).item(), (group, key)
    assert expected['global_step'] == actual['global_step'] == 4
    assert expected['next_row'] == actual['next_row']
