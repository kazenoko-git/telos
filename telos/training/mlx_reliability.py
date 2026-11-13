"""Reliability measurements for MLX COROSred scheduling."""
import numpy as np


def reliability_metrics(model, tokens):
    import mlx.core as mx
    logits, scores = model(tokens, mask_override=True, return_reliability=True)
    labels = mx.argmax(logits[:, :-1], axis=-1) == tokens[:, 1:]
    scores = scores[:, :-1]
    mx.eval(labels, scores)
    labels = np.array(labels).reshape(-1)
    scores = np.array(scores.astype(mx.float32)).reshape(-1)
    accuracy = float(np.mean((scores >= 0) == labels))
    positives = scores[labels]
    negatives = scores[~labels]
    if not len(positives) or not len(negatives):
        auc = .5
    else:
        values, inverse = np.unique(scores, return_inverse=True)
        negative_counts = np.bincount(inverse[~labels], minlength=len(values))
        below = np.cumsum(negative_counts) - negative_counts
        positive_bins = inverse[labels]
        auc = float(np.sum(below[positive_bins] + .5 * negative_counts[positive_bins])
                    / (len(positives) * len(negatives)))
    return accuracy, auc
