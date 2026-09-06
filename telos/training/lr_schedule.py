"""
Unified LR Schedule for PyTorch.
(MLX has its own functional schedule in the trainer loop).
"""

import math
try:
    from torch.optim.lr_scheduler import LRScheduler as _LRSchedulerBase
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler as _LRSchedulerBase


class WarmupCosineLR(_LRSchedulerBase):
    """Linear Warmup followed by Cosine Annealing decay with optional step quantization for XLA graph stability."""

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr: float = 3e-5,
        update_cadence: int = 1,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.update_cadence = max(1, update_cadence)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        step = self.last_epoch

        if step < self.warmup_steps:
            alpha = step / max(1, self.warmup_steps)
            return [base_lr * alpha for base_lr in self.base_lrs]

        if step >= self.max_steps:
            return [self.min_lr for _ in self.base_lrs]

        # Quantize step to update_cadence to prevent XLA from tracing distinct float constants every step
        effective_step = step if self.update_cadence == 1 else (step - (step % self.update_cadence))
        progress = (effective_step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))

        return [self.min_lr + (base_lr - self.min_lr) * cosine_factor for base_lr in self.base_lrs]
