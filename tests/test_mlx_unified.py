"""
Unit tests for Apple Silicon MLX Unified Continuous COROSred training engine.
Verifies:
1. corosred_unified_loss_fn_mlx forward pass and gradient computation.
2. mx.compile compatibility with dynamic schedule weights (alpha, beta, gamma).
3. Parameter update verification on model weights.
"""

import pytest


def test_mlx_unified_corosred_step():
    """Verify that unified COROSred step compiles and optimizes weights in MLX."""
    try:
        from telos.training.core import metal_usable
        if not metal_usable():
            pytest.skip("MLX requires a Metal-capable session")
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
    except ImportError:
        pytest.skip("MLX not installed in this environment")

    from telos.models import MLXTelosTransformer
    from telos.diffusion.corosred import corosred_unified_loss_fn_mlx

    mx.random.seed(42)

    vocab_size = 500
    seq_len = 32
    bs = 4
    d_model = 64
    n_layers = 2
    n_heads = 2

    # Instantiate model with reliability head enabled
    model = MLXTelosTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        is_causal=False,
        use_reliability_head=True,
        precision="float32"
    )

    optimizer = optim.AdamW(learning_rate=1e-3, betas=[0.9, 0.95])
    loss_and_grad_fn = nn.value_and_grad(model, corosred_unified_loss_fn_mlx)

    def microbatch_step_uncompiled(batch_seqs, alpha, beta, gamma):
        (loss, ce), grads = loss_and_grad_fn(
            model,
            batch_seqs,
            vocab_size,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            causal_ratio=0.75,
            mask_prob=0.15,
            k_amb=5
        )
        return loss, ce, grads

    compiled_step = mx.compile(microbatch_step_uncompiled, inputs=[model.state], outputs=[model.state])

    # Generate synthetic batch
    batch = mx.random.randint(0, vocab_size, (bs, seq_len))
    alpha_mx = mx.array(0.85, dtype=mx.float32)
    beta_mx = mx.array(0.15, dtype=mx.float32)
    gamma_mx = mx.array(0.05, dtype=mx.float32)

    # Initial weights snapshot
    initial_weights = mx.array(model.emb.weight)

    # Execute compiled training step
    loss, ce, grads = compiled_step(batch, alpha_mx, beta_mx, gamma_mx)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), loss, ce)

    # Assert losses are finite and positive
    assert not mx.isnan(loss).item(), "Unified MLX loss is NaN"
    assert loss.item() > 0.0, "Unified MLX loss must be strictly positive"
    assert ce.item() > 0.0, "Unified MLX cross-entropy must be strictly positive"

    # Assert weights updated
    updated_weights = mx.array(model.emb.weight)
    weight_diff = mx.sum(mx.abs(updated_weights - initial_weights)).item()
    assert weight_diff > 1e-6, "Weights did not update after compiled optimizer step"


def test_mlx_unified_loss_weighting_contract():
    """Verify that MLX unified loss strictly obeys per-task per-token mean weighting."""
    try:
        from telos.training.core import metal_usable
        if not metal_usable():
            pytest.skip("MLX requires a Metal-capable session")
        import mlx.core as mx
    except ImportError:
        pytest.skip("MLX not installed in this environment")

    from telos.models import MLXTelosTransformer
    from telos.diffusion.corosred import corosred_unified_loss_fn_mlx

    model = MLXTelosTransformer(
        vocab_size=128,
        d_model=64,
        n_layers=2,
        n_heads=2,
        is_causal=False,
        use_reliability_head=True,
        precision="float32"
    )
    batch = mx.random.randint(0, 128, (4, 32))
    alpha = 0.85
    beta = 0.15
    gamma = 0.0

    total_loss, pooled_ce = corosred_unified_loss_fn_mlx(
        model,
        batch,
        vocab_size=128,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        causal_ratio=0.75,
        mask_prob=0.15,
        k_amb=5
    )
    assert not mx.isnan(total_loss).item()
    assert abs(total_loss.item() - pooled_ce.item()) < 1e-6


@pytest.mark.parametrize('cadence,expected', [(1, 6), (3, 3)])
def test_reliability_probe_cadence_survives_resume(tmp_path, monkeypatch, cadence, expected):
    import numpy as np
    mx = pytest.importorskip('mlx.core')
    if not mx.metal.is_available():
        pytest.skip('Metal required')
    from copy import deepcopy
    from telos.models import MLXTelosTransformer
    from telos.training.trainer_mlx import UnifiedMLXTrainer
    from telos.training import mlx_reliability
    calls = []
    real_probe = mlx_reliability.reliability_metrics
    def probe(*args):
        calls.append(1)
        return real_probe(*args)
    monkeypatch.setattr(mlx_reliability, 'reliability_metrics', probe)
    cfg = {
        'model': dict(vocab_size=32, d_model=64, n_layers=1, n_heads=2, seq_len=8),
        'training': dict(max_steps=6, batch_size=2, gradient_accumulation=2, seed=17,
                         precision='bfloat16', max_lr=.001, warmup_steps=1),
        'corosred': dict(reliability_cadence=cadence),
        'data': dict(synthetic=True),
        'checkpoint': dict(checkpoint_dir=str(tmp_path), save_every_steps=2),
    }
    def make():
        mx.random.seed(19)
        model = MLXTelosTransformer(vocab_size=32, d_model=64, n_layers=1, n_heads=2,
                                    use_reliability_head=True, is_causal=True)
        return UnifiedMLXTrainer('corosred', model, deepcopy(cfg))
    original = make()
    original.train()
    assert len(calls) == expected
    from telos.training.checkpoint import load_mlx_state
    from mlx.utils import tree_flatten
    saved = load_mlx_state(tmp_path / 'checkpoint_final.resume.npz')
    resumed = make()
    resumed.load_checkpoint(tmp_path / 'checkpoint_step_2.resume.npz')
    resumed.train()
    actual = load_mlx_state(tmp_path / 'checkpoint_final.resume.npz')
    assert actual['reliability_metrics'] == saved['reliability_metrics']
    for key in ('model', 'optimizer', 'master_params', 'mlx_rng'):
        for (_, a), (_, b) in zip(tree_flatten(saved[key]), tree_flatten(actual[key])):
            np.testing.assert_array_equal(np.asarray(a.astype(mx.float32)), np.asarray(b.astype(mx.float32)))
