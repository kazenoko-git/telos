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
