"""Exercise Metal projection/CE tail handling, derivatives and AR integration."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("Metal required", allow_module_level=True)

from telos.training.mlx_fused_projection import fused_projection_cross_entropy
from telos.training.mlx_losses import cross_entropy


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("length,width,vocab", [(5, 33, 1), (5, 17, 41), (5, 64, 257), (43, 64, 259)])
def test_fused_projection_matches_full_compiled_head(dtype, length, width, vocab):
    mx.random.seed(411)
    hidden = mx.random.normal((3, length, width)).astype(dtype)
    weight = (mx.random.normal((vocab, width)) * 0.125).astype(dtype)
    targets = (mx.arange(3 * length) * 313 % vocab).reshape(3, length)
    cotangent = mx.linspace(-0.7, 1.0, 3 * length).reshape(3, length)

    def step(function):
        def scalar(h, w):
            losses = function(h, w, targets)
            return mx.sum(losses * cotangent), losses
        return mx.compile(mx.value_and_grad(scalar, argnums=(0, 1)))(hidden, weight)

    (_, expected), expected_grads = step(lambda h, w, t: cross_entropy(h @ w.T, t))
    (_, actual), actual_grads = step(fused_projection_cross_entropy)
    np.testing.assert_allclose(np.array(actual), np.array(expected), atol=5e-6, rtol=2e-6)
    for ref, found in zip(expected_grads, actual_grads):
        ref, found = ref.astype(mx.float32), found.astype(mx.float32)
        relative = mx.sqrt(mx.sum((ref - found) ** 2) / mx.maximum(mx.sum(ref ** 2), 1e-30)).item()
        assert relative < (2e-5 if dtype == mx.float32 else 0.015)


def test_fused_projection_preserves_very_small_confident_target_gradient():
    h = mx.array([[1., 0.]], dtype=mx.bfloat16)
    w = mx.zeros((257, 2), dtype=mx.bfloat16).at[0, 0].add(30.)
    t = mx.array([0])
    loss, gradient = mx.value_and_grad(lambda weight: fused_projection_cross_entropy(h, weight, t).sum())(w)
    expected = np.log1p(256 * np.exp(-30))
    assert loss.item() == pytest.approx(expected, rel=5e-6)
    assert gradient[0, 0].item() == pytest.approx(-expected, rel=0.005)


def test_fused_projection_ar_pretraining_step(tmp_path):
    from telos.models.mlx_transformer import MLXTelosTransformer
    from telos.training.trainer_mlx import UnifiedMLXTrainer
    from telos.training.checkpoint import read_resume

    data = tmp_path / "tokens.bin"
    (np.arange(96, dtype=np.uint16) % 33 + 4).tofile(data)
    config = dict(
        paradigm="ar",
        model=dict(vocab_size=37, d_model=32, n_layers=1, n_heads=2, n_kv_heads=2, seq_len=8),
        training=dict(max_steps=2, warmup_steps=1, batch_size=2, gradient_accumulation=2,
                      max_lr=0.001, min_lr=0.0001, seed=17, precision="bf16",
                      fused_vocab_loss=True, fused_vocab_chunk_axis="metal"),
        data=dict(train_path=str(data)),
        checkpoint=dict(checkpoint_dir=str(tmp_path / "run"), save_every_steps=1),
    )
    model = MLXTelosTransformer(37, 32, 1, 2, is_causal=True)
    trainer = UnifiedMLXTrainer("ar", model, config)
    trainer.train()
    state = read_resume(tmp_path / "run/checkpoint_final.resume.npz")
    assert state["global_step"] == 2
    assert state["model"]["emb"]["weight"].shape == (37, 32)
