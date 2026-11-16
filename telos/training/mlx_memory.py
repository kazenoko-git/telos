"""Conservative MLX training-memory planning without allocating device arrays."""

from dataclasses import dataclass
import math

GIB = 1024**3


@dataclass(frozen=True)
class MemoryBudget:
    total_bytes: int
    ceiling_bytes: int
    headroom_bytes: int

    @property
    def allocation_bytes(self):
        return self.ceiling_bytes - self.headroom_bytes


def training_memory_budget(total_bytes, *, recommended_bytes=None, limit_bytes=None):
    """Training allocation plus at least one GiB must fit in 60% of RAM."""
    if total_bytes <= 0:
        raise ValueError('System memory must be positive')
    ceiling = int(total_bytes * 0.60)
    for cap in (recommended_bytes, limit_bytes):
        if cap is not None:
            if cap <= 0:
                raise ValueError('Memory caps must be positive')
            ceiling = min(ceiling, int(cap))
    budget = MemoryBudget(int(total_bytes), ceiling, GIB)
    if budget.allocation_bytes <= 0:
        raise ValueError('Memory budget cannot provide 1 GiB training headroom')
    return budget


def estimate_training_bytes(parameter_count, model, batch_size, *, checkpointing=False):
    """Estimate peak working storage, not just weights or a guaranteed maximum.

    Includes FP32 Adam/master/update storage, activations, vocabulary logits and
    fallback attention matrices. A 1.8 safety factor covers allocator/workspace
    uncertainty; the separate fixed headroom must still be reserved.
    """
    seq = int(model.get('seq_len', 512))
    width = int(model.get('d_model', 512))
    layers = int(model.get('n_layers', 12))
    heads = int(model.get('n_heads', max(1, width // 64)))
    vocab = int(model.get('vocab_size', 8192))
    if min(parameter_count, batch_size, seq, width, layers, heads, vocab) <= 0:
        raise ValueError('Model dimensions, parameter count and batch must be positive')
    # Static allowance includes optimizer-update temporaries as well as live state.
    static = parameter_count * 24
    active_layers = 1 if checkpointing else layers
    activations = batch_size * seq * width * (layers + 8 if checkpointing else layers * 16)
    logits = batch_size * seq * vocab * 6
    attention = batch_size * heads * seq * seq * 8 * active_layers
    return math.ceil((static + activations + logits + attention) * 1.8)


def effective_batch_divisors(effective_batch):
    if effective_batch < 1:
        raise ValueError('Effective batch must be positive')
    divisors = set()
    for n in range(1, math.isqrt(effective_batch) + 1):
        if effective_batch % n == 0:
            divisors.update((n, effective_batch // n))
    return sorted(divisors)


def plan_training_batch(parameter_count, model, effective_batch, budget, *, checkpointing=False, allow_checkpointing=True):
    """Pick the largest fitting exact microbatch; checkpoint only if necessary."""
    sizes = effective_batch_divisors(effective_batch)
    modes = [True] if checkpointing else ([False, True] if allow_checkpointing else [False])
    for use_checkpoint in modes:
        for size in reversed(sizes):
            estimate = estimate_training_bytes(parameter_count, model, size, checkpointing=use_checkpoint)
            if estimate <= budget.allocation_bytes:
                return dict(batch_size=size, grad_accum=effective_batch // size,
                            checkpointing=use_checkpoint, estimated_peak_bytes=estimate,
                            effective_batch=effective_batch,
                            memory_ceiling_bytes=budget.ceiling_bytes,
                            allocation_budget_bytes=budget.allocation_bytes,
                            headroom_bytes=budget.headroom_bytes)
    raise RuntimeError('Model training does not fit the memory budget even at microbatch 1 '
                       'with checkpointing and accumulation; reduce model size or context length')
