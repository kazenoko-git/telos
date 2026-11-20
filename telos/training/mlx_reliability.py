"""Reliability measurements for MLX COROSred scheduling."""
import numpy as np


def reliability_metrics(model, tokens):
    import mlx.core as mx
    from telos.training.mlx_losses import cross_entropy

    logits, scores = model(
        tokens,
        mask_override=True,
        return_reliability=True,
        next_token_only=True,
    )
    targets = tokens[:, 1:]
    labels = mx.argmax(logits, axis=-1) == targets
    scores = scores[:, :-1]
    causal_ce = cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="mean",
    )
    # Materialize causal outputs and drop causal logits before running the infill pass.
    mx.eval(labels, scores, causal_ce)
    del logits

    positions = mx.broadcast_to((mx.arange(tokens.shape[1]) % 7 == 1)[None, :], tokens.shape)
    masked = mx.where(positions, mx.ones_like(tokens), tokens)
    infill = model(masked, mask_override=False)
    per_token = cross_entropy(
        infill.reshape(-1, infill.shape[-1]),
        tokens.reshape(-1),
        reduction="none",
    ).reshape(tokens.shape)
    infill_ce = mx.sum(per_token * positions) / mx.maximum(mx.sum(positions), 1)
    mx.eval(infill_ce)
    del infill

    labels_np = np.array(labels).reshape(-1)
    scores_np = np.array(scores.astype(mx.float32)).reshape(-1)
    accuracy = float(np.mean((scores_np >= 0) == labels_np))
    positives = scores_np[labels_np]
    negatives = scores_np[~labels_np]
    if not len(positives) or not len(negatives):
        auc = 0.5
    else:
        values, inverse = np.unique(scores_np, return_inverse=True)
        negative_counts = np.bincount(inverse[~labels_np], minlength=len(values))
        below = np.cumsum(negative_counts) - negative_counts
        positive_bins = inverse[labels_np]
        auc = float(
            np.sum(below[positive_bins] + 0.5 * negative_counts[positive_bins])
            / (len(positives) * len(negatives))
        )
    return accuracy, auc, float(causal_ce.item()), float(infill_ce.item())
