"""Checkpointed vocabulary projection + CE; opt-in memory/compute tradeoff."""
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def token_chunked_cross_entropy(hidden, weight, targets, chunk_size=512):
    """Checkpointed token-chunked projection + CE using native F.cross_entropy."""
    if chunk_size < 1 or targets.shape != hidden.shape[:-1]:
        raise ValueError('Positive token chunk size and matching target shape required')
    shape = targets.shape
    hidden = hidden.reshape(-1, hidden.shape[-1])
    targets = targets.reshape(-1)
    losses = []
    for start in range(0, hidden.shape[0], chunk_size):
        h_chunk = hidden[start:start + chunk_size]
        t_chunk = targets[start:start + chunk_size]

        def reduce_chunk(inputs, weights, labels):
            return F.cross_entropy(F.linear(inputs, weights).float(), labels, reduction='none')

        if torch.is_grad_enabled():
            chunk_loss = checkpoint(reduce_chunk, h_chunk, weight, t_chunk, use_reentrant=False)
        else:
            chunk_loss = reduce_chunk(h_chunk, weight, t_chunk)
        losses.append(chunk_loss)
    return torch.cat(losses, dim=0).reshape(shape)


def tiled_cross_entropy(hidden, weight, targets, chunk_size=256, *, chunk_axis="vocab"):
    if chunk_axis == "token":
        return token_chunked_cross_entropy(hidden, weight, targets, chunk_size=chunk_size)
    if chunk_axis != "vocab":
        raise ValueError("chunk_axis must be 'vocab' or 'token'")
    if chunk_size < 1 or targets.shape != hidden.shape[:-1]:
        raise ValueError('Positive vocabulary chunk size and matching target shape required')
    # Class IDs must be valid; callers replace ignored labels before this function.
    shape = targets.shape
    hidden = hidden.reshape(-1, hidden.shape[-1])
    targets = targets.reshape(-1)
    partition = selected = None
    for start in range(0, weight.shape[0], chunk_size):
        tile = weight[start:start + chunk_size]
        def reduce_tile(inputs, weights, labels, start=start):
            logits = F.linear(inputs, weights).float()
            within = (labels >= start) & (labels < start + weights.shape[0])
            index = (labels - start).clamp(0, weights.shape[0] - 1)
            chosen = logits.gather(1, index[:, None]).squeeze(1)
            return torch.logsumexp(logits, -1), torch.where(within, chosen, 0.0)
        if torch.is_grad_enabled():
            lse, target = checkpoint(reduce_tile, hidden, tile, targets, use_reentrant=False)
        else:
            lse, target = reduce_tile(hidden, tile, targets)
        partition = lse if partition is None else torch.logaddexp(partition, lse)
        selected = target if selected is None else selected + target
    return (partition - selected).reshape(shape)
