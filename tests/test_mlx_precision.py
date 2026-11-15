"""Numerical regressions for FP32 GEMMs and repeated-token lookup gradients."""
import os
import subprocess
import sys

import numpy as np
import pytest

mx = pytest.importorskip('mlx.core')
if not mx.metal.is_available():
    pytest.skip('Metal required', allow_module_level=True)

from telos.models.mlx_components import MLXEmbedding


def test_default_fp32_matmul_matches_float64_reference():
    rng = np.random.default_rng(73)
    # Nontrivial mantissas distinguish FP32 accumulation from TF32 inputs.
    a = rng.uniform(.75, 1.25, (32, 64)).astype(np.float32)
    b = rng.uniform(.75, 1.25, (64, 192)).astype(np.float32)
    expected = a.astype(np.float64) @ b.astype(np.float64)
    actual = mx.array(a) @ mx.array(b)
    np.testing.assert_allclose(np.array(actual), expected, rtol=1e-6, atol=2e-6)


@pytest.mark.parametrize('dtype', [mx.bfloat16, mx.float16])
@pytest.mark.parametrize('compiled', [False, True])
def test_repeated_token_gradients_accumulate_in_fp32(dtype, compiled):
    embedding = MLXEmbedding(4, 8)
    embedding.set_dtype(dtype)
    indices = mx.ones((257,), dtype=mx.int32)
    # Every term is representable, but accumulating small terms into BF16's
    # 1.0 loses updates. FP32 sum is exactly 1 + 256/256 = 2.
    cotangent = mx.concatenate((mx.ones((1, 8)), mx.full((256, 8), 1/256))).astype(dtype)
    def differentiate(weight, cotangent):
        def lookup(weight):
            embedding.weight = weight
            return embedding(indices)
        return mx.vjp(lookup, [weight], [cotangent])[1][0]
    fn = mx.compile(differentiate) if compiled else differentiate
    gradient = fn(embedding.weight, cotangent)
    expected = mx.zeros((4, 8), dtype=dtype).at[1].add(mx.full((8,), 2, dtype=dtype))
    assert gradient.dtype == dtype
    np.testing.assert_array_equal(np.array(gradient.astype(mx.float32)),
                                  np.array(expected.astype(mx.float32)))


def test_explicit_tf32_opt_in_is_not_overwritten():
    environment = dict(os.environ, MLX_ENABLE_TF32='1')
    subprocess.run([sys.executable, '-c',
                    'import os; import telos.training.core; assert os.environ["MLX_ENABLE_TF32"] == "1"'],
                   env=environment, check=True, timeout=30)


def test_fp32_transformer_gradients_match_independent_float64_reference():
    import torch
    import torch.nn.functional as functional
    from mlx.utils import tree_flatten
    from telos.models.mlx_transformer import MLXTelosTransformer
    from telos.training.mlx_losses import cross_entropy

    mx.random.seed(42)
    model = MLXTelosTransformer(vocab_size=128, d_model=64, n_layers=2,
                                n_heads=2, precision='float32', is_causal=True)
    tokens = mx.random.randint(4, 128, (2, 16))
    weights = {name: torch.tensor(np.array(value), dtype=torch.float64, requires_grad=True)
               for name, value in tree_flatten(model.parameters())}
    ids = torch.tensor(np.array(tokens).astype(np.int64))

    def norm(inputs, key):
        return inputs * torch.rsqrt(inputs.square().mean(-1, keepdim=True) + 1e-5) * weights[key]

    # Independent CPU/Float64 implementation of this small traditional-RoPE
    # transformer, including both paths into its tied embedding parameter.
    hidden = functional.embedding(ids, weights['emb.weight'])
    angles = torch.arange(16, dtype=torch.float64)[:, None] * torch.pow(
        10000., -torch.arange(16, dtype=torch.float64) / 16)[None, :]
    def rotate(inputs):
        pairs = inputs.reshape(2, 2, 16, 16, 2)
        a, b = pairs[..., 0], pairs[..., 1]
        return torch.stack((a * angles.cos() - b * angles.sin(),
                            a * angles.sin() + b * angles.cos()), -1).flatten(-2)

    for index in range(2):
        prefix = f'layers.{index}.'
        projected = functional.linear(norm(hidden, prefix + 'norm1.weight'), weights[prefix + 'qkv_proj.weight'])
        query, key, value = [part.reshape(2, 16, 2, 32).transpose(1, 2) for part in projected.chunk(3, -1)]
        scores = (rotate(query) * 32**-.5) @ rotate(key).transpose(-1, -2)
        scores = scores.masked_fill(torch.triu(torch.ones(16, 16, dtype=torch.bool), 1), float('-inf'))
        attention = (scores.softmax(-1) @ value).transpose(1, 2).reshape(2, 16, 64)
        hidden = hidden + functional.linear(attention, weights[prefix + 'out.weight'])
        gate, up = functional.linear(norm(hidden, prefix + 'norm2.weight'),
                                     weights[prefix + 'mlp.gate_up.weight']).chunk(2, -1)
        hidden = hidden + functional.linear(functional.silu(gate) * up, weights[prefix + 'mlp.w3.weight'])
    logits = norm(hidden, 'norm.weight') @ weights['emb.weight'].T
    expected_loss = functional.cross_entropy(logits[:, :-1].reshape(-1, 128), ids[:, 1:].reshape(-1))
    expected_loss.backward()

    import mlx.nn as nn
    def loss(active):
        return cross_entropy(active(tokens)[:, :-1], tokens[:, 1:], reduction='mean')
    for compiled in (False, True):
        def candidate():
            return nn.value_and_grad(model, loss)(model)
        fn = mx.compile(candidate, inputs=[model.state]) if compiled else candidate
        actual_loss, gradients = fn()
        np.testing.assert_allclose(np.array(actual_loss), expected_loss.item(), rtol=1e-4, atol=2e-6)
        for name, gradient in tree_flatten(gradients):
            np.testing.assert_allclose(np.array(gradient), weights[name].grad.numpy(),
                                       rtol=1e-4, atol=2e-6, err_msg=name)
