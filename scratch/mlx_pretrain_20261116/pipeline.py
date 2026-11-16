"""Diagnostic one-microbatch-ahead dispatch; no production source changes.

The next graph is constructed while the previous evaluation is in flight, but
its evaluation is not submitted until the previous one completes. Snapshot array
values, preserve left-to-right sums, and drain on errors. This module patches a
single-threaded training experiment only; it is not a public concurrent API.
"""
import mlx.core as mx
from mlx.utils import tree_flatten
from telos.training import core, mlx_step


class Evaluator:
    def __init__(self):
        self.pending = None

    def __getattr__(self, name):
        return getattr(mx, name)

    def drain(self):
        if self.pending is not None:
            mx.eval(*self.pending)
            self.pending = None

    def eval(self, *targets):
        if len(targets) == 3 and isinstance(targets[0], dict):
            if self.pending is not None:
                raise RuntimeError('An accumulation evaluation was submitted before draining its predecessor')
            arrays = [value for _, value in tree_flatten(targets)]
            mx.async_eval(*arrays)
            self.pending = arrays
        else:
            self.drain()
            mx.eval(*targets)


original_execute = core.execute_mlx_training_step


def execute(model, optimizer, compiled_step_fn, batch_iterator, grad_accum,
            grad_clip, is_first_step, eval_every_microbatch=False,
            stage_profile=None, accumulation_step_fn=None):
    arguments = (model, optimizer, compiled_step_fn, batch_iterator, grad_accum,
                 grad_clip, is_first_step, eval_every_microbatch)
    if grad_accum < 2 or eval_every_microbatch or stage_profile is not None:
        return original_execute(*arguments, stage_profile=stage_profile,
                                accumulation_step_fn=accumulation_step_fn)
    evaluator = Evaluator()

    def wrap(function):
        def prepared(*args):
            result = function(*args)
            evaluator.drain()
            return result
        return prepared

    saved_mx = core.mx
    core.mx = evaluator
    try:
        return original_execute(model, optimizer, wrap(compiled_step_fn), batch_iterator,
                                grad_accum, grad_clip, is_first_step, eval_every_microbatch,
                                accumulation_step_fn=wrap(accumulation_step_fn) if accumulation_step_fn else None)
    finally:
        try:
            evaluator.drain()
        finally:
            core.mx = saved_mx


def install():
    mlx_step.execute_mlx_training_step = execute


def uninstall():
    mlx_step.execute_mlx_training_step = original_execute
