"""
learning rate schedule with linear warmup and cosine Decay.

i believe this is standard schedule
1. linear warmup from 0.0 to max_lr over first warmup_steps.
2. cosine decay from max_lr down to min_lr over remaining steps.
"""

import math
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineLR(_LRScheduler):
    """Linear Warmup followed by Cosine Annealing decay."""

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr: float = 3e-5,
        last_epoch: int = -1
    ):
        """
        Args:
            optimizer: PyTorch optimizer instance.
            warmup_steps: number of steps for linear warmup phase.
            max_steps: total training steps.
            min_lr: floor learning rate at end of cosine decay.
        """
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        step = self.last_epoch

        # linear scaling from 0 to base_lr
        if step < self.warmup_steps:
            alpha = step / max(1, self.warmup_steps)
            return [base_lr * alpha for base_lr in self.base_lrs]

        # clamp to min_lr if step exceeds max_steps
        if step >= self.max_steps:
            return [self.min_lr for _ in self.base_lrs]

        # calculate cosine factor between 0.0 and 1.0
        progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))

        return [self.min_lr + (base_lr - self.min_lr) * cosine_factor for base_lr in self.base_lrs]
