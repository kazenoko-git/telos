"""Parity test for MLX reliability_metrics against explicit FP32 logsumexp reference."""
from __future__ import annotations

import numpy as np
import pytest

from telos.training.core import metal_usable


@pytest.mark.skipif(not metal_usable(), reason="MLX Metal GPU unavailable in this session")
def test_mlx_reliability_metrics_matches_fp32_logsumexp_reference():
    import mlx.core as mx
    from telos.models.mlx_transformer import MLXTelosTransformer
    from telos.training.mlx_reliability import reliability_metrics

    mx.random.seed(42)
    model = MLXTelosTransformer(
        vocab_size=256, d_model=64, n_layers=2, n_heads=2,
        is_causal=True, use_reliability_head=True,
    )
    tokens = mx.random.randint(4, 256, (4, 32)).astype(mx.int32)
    mx.eval(model.parameters(), tokens)

    acc, auc, causal_ce, infill_ce = reliability_metrics(model, tokens)

    # Explicit FP32 logsumexp reference matching the previous implementation.
    logits, scores = model(tokens, mask_override=True, return_reliability=True)
    targets = tokens[:, 1:]
    labels = mx.argmax(logits[:, :-1], axis=-1) == targets
    scores = scores[:, :-1]
    mx.eval(labels, scores)
    labels_np = np.array(labels).reshape(-1)
    scores_np = np.array(scores.astype(mx.float32)).reshape(-1)
    ref_acc = float(np.mean((scores_np >= 0) == labels_np))
    ref_causal_ce = float(
        mx.mean(
            mx.logsumexp(logits[:, :-1].astype(mx.float32), axis=-1)
            - mx.take_along_axis(logits[:, :-1].astype(mx.float32), targets[..., None], axis=-1)[..., 0]
        ).item()
    )
    positions = mx.broadcast_to((mx.arange(tokens.shape[1]) % 7 == 1)[None, :], tokens.shape)
    masked = mx.where(positions, mx.ones_like(tokens), tokens)
    infill = model(masked, mask_override=False).astype(mx.float32)
    per_token = mx.logsumexp(infill, axis=-1) - mx.take_along_axis(infill, tokens[..., None], axis=-1)[..., 0]
    ref_infill_ce = float((mx.sum(per_token * positions) / mx.maximum(mx.sum(positions), 1)).item())

    assert acc == ref_acc
    assert abs(causal_ce - ref_causal_ce) <= 1e-5
    assert abs(infill_ce - ref_infill_ce) <= 1e-5
    assert 0.0 <= auc <= 1.0
