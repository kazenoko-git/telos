"""Compiled MLX training steps, independent of model architecture and objective."""

import mlx.core as mx
from mlx.utils import tree_flatten, tree_map

from .core import clip_grad_norm_mlx, execute_mlx_training_step


class MLXTrainingStep:
    """Fuse forward, backward, clipping and Adam update for one microbatch.

    The loss/gradient callable has the existing ``(loss, metric, gradients)``
    contract. Accumulated runs retain the reference bounded-memory path. The
    optional asynchronous path holds at most one previous step on the host;
    call ``synchronize`` before measuring elapsed time or saving checkpoints.
    """

    def __init__(self, model, optimizer, microbatch_step, *, grad_clip=1.0, fused=True):
        self.model = model
        self.optimizer = optimizer
        self.microbatch_step = microbatch_step
        self.grad_clip = grad_clip
        self.fused = fused
        self._initialized = False
        self._full_step = None
        self._pending = None

    def _build_full_step(self):
        model, optimizer = self.model, self.optimizer
        master = getattr(optimizer, "master_params", None)
        state = [model.state, optimizer.state, mx.random.state]
        if master is not None:
            state.append(master)

        def full_step(*batch):
            loss, metric, grads = self.microbatch_step(*batch)
            if self.grad_clip > 0:
                grads, _ = clip_grad_norm_mlx(grads, self.grad_clip)
            if master is not None:
                master.update(optimizer.apply_gradients(grads, master))
                model.update(tree_map(lambda p: p.astype(mx.bfloat16), master))
            else:
                optimizer.update(model, grads)
            return loss, metric

        self._full_step = mx.compile(full_step, inputs=state, outputs=state)

    def __call__(self, batches, *, grad_accum=1, eval_every_microbatch=False, asynchronous=False):
        if grad_accum < 1:
            raise ValueError("grad_accum must be positive")
        batches = iter(batches)
        if not self._initialized or grad_accum != 1 or not self.fused:
            self.synchronize()
            result = execute_mlx_training_step(
                self.model, self.optimizer, self.microbatch_step, batches,
                grad_accum, self.grad_clip, not self._initialized, eval_every_microbatch,
                stage_profile=getattr(self, "stage_profile", None),
            )
            self._initialized = True
            return result

        if self._full_step is None:
            self._build_full_step()
        batch = next(batches)
        result = self._full_step(*(batch if isinstance(batch, tuple) else (batch,)))
        targets = [self.model.parameters(), self.optimizer.state, result]
        if hasattr(self.optimizer, "master_params"):
            targets.append(self.optimizer.master_params)
        # Store array values, never mutable dictionaries captured by compilation.
        arrays = [v for _, v in tree_flatten(targets)]
        if asynchronous:
            mx.async_eval(*arrays)
            if self._pending is not None:
                mx.eval(*self._pending)
            self._pending = arrays
        else:
            mx.eval(*arrays)
            self._pending = None
        return result

    def synchronize(self):
        if self._pending is not None:
            mx.eval(*self._pending)
            self._pending = None
