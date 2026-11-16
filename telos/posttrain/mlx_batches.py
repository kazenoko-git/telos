"""CPU staging and bounded frozen-reference scores for MLX post-training."""


def collate_mlx(rows, max_length, *, fixed=False):
    """Stage encoded rows directly as int32, without intermediate torch tensors."""
    import numpy as np
    import mlx.core as mx

    encoded = [row["_encoded"] for row in rows]
    width = max_length if fixed else max(len(tokens) for tokens, _, _ in encoded)
    ids = np.full((len(rows), width), encoded[0][2], dtype=np.int32)
    labels = np.full_like(ids, -100)
    for index, (tokens, targets, _) in enumerate(encoded):
        ids[index, :len(tokens)] = tokens
        labels[index, :len(targets)] = targets
    return mx.array(ids), mx.array(labels)


class ReferenceScores:
    """Cache individual response scores, retaining at most a batch-based budget.

    A miss evaluates the original complete batch, preserving reference GEMM
    shapes. Width is part of the key because batch padding can affect numerical
    execution. Scores are host FP32 scalars; no reference graphs are retained.
    The cache belongs to one frozen reference and one training invocation.
    """

    def __init__(self, max_batches, batch_size):
        self.capacity = max_batches * 2 * batch_size
        self.scores = {}

    def get(self, rows, width, evaluate):
        import mlx.core as mx

        if not self.capacity:
            return evaluate()
        keys = [(width, row["_reference_key"]) for row in rows]
        if all(key in self.scores for key in keys):
            return mx.array([self.scores[key] for key in keys], dtype=mx.float32)
        result = evaluate()
        mx.eval(result)
        for key, score in zip(keys, result.tolist()):
            if len(self.scores) >= self.capacity:
                break
            self.scores.setdefault(key, score)
        return result
