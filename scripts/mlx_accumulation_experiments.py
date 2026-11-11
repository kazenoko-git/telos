"""Accumulation experiments isolated from the production training path.

None of these variants established a throughput improvement on 50M MBS8.
Retained for reproducible short benchmarks and numerical parity checks.
"""
from unittest.mock import patch

import mlx.core as mx
from mlx.utils import tree_map

from telos.training.core import clip_grad_norm_mlx, cast_optimizer_moments_bf16
from telos.training.mlx_step import MLXTrainingStep

def execute_mlx_training_step(
    model,
    optimizer,
    compiled_step_fn,
    batch_iterator,
    grad_accum: int,
    grad_clip: float,
    is_first_step: bool,
    eval_every_microbatch: bool = False,
    pipeline_accumulation: bool = False,
    accumulate_fn=None,
    defer_last_microbatch: bool = False,
):
    accum_grads = None
    accum_loss = mx.array(0.0, dtype=mx.float32)
    accum_ce = mx.array(0.0, dtype=mx.float32)
    pending = None

    for i in range(grad_accum):
        batch_data = next(batch_iterator)
        if isinstance(batch_data, tuple):
            loss, ce, grads = compiled_step_fn(*batch_data)
        else:
            loss, ce, grads = compiled_step_fn(batch_data)

        if accum_grads is None:
            accum_grads = grads
            accum_loss = accum_loss + loss
            accum_ce = accum_ce + ce
        elif accumulate_fn is not None:
            accum_grads, accum_loss, accum_ce = accumulate_fn(
                (accum_grads, accum_loss, accum_ce), (grads, loss, ce)
            )
        else:
            accum_grads = tree_map(lambda a, b: a + b, accum_grads, grads)
            accum_loss = accum_loss + loss
            accum_ce = accum_ce + ce

        # Never retain an accumulation window of activation graphs. RAM-tier
        # heuristics cannot predict that window's size. Materialize each
        # microbatch's accumulated gradients before building the next graph.
        if eval_every_microbatch or grad_accum > 1:
            if defer_last_microbatch and i == grad_accum - 1 and not eval_every_microbatch:
                continue  # The optimizer eval materializes this final graph.
            if pipeline_accumulation and grad_accum > 1:
                # Build one lookahead graph while the previous microbatch runs.
                # Complete that evaluation before submitting the next: no window
                # of unevaluated microbatches or concurrent activation sets.
                if pending is not None:
                    mx.eval(*pending)
                pending = (accum_grads, accum_loss, accum_ce)
                mx.async_eval(*pending)
            else:
                mx.eval(accum_grads, accum_loss, accum_ce)

    # The optimizer's final eval below also waits for the last microbatch.
    # Prepare its graph while that microbatch is still executing.

    use_master_weights = getattr(model, "use_master_weights", False)
    if use_master_weights and not hasattr(optimizer, "master_params"):
        # Initialize fp32 master weights on optimizer to prevent polluting model.parameters()
        optimizer.master_params = tree_map(lambda p: p.astype(mx.float32), model.parameters())

    if is_first_step:
        # Run eager update on first step before casting optimizer moments to bfloat16
        if grad_clip > 0.0:
            accum_grads, _ = clip_grad_norm_mlx(accum_grads, max_norm=grad_clip, scale=float(grad_accum))
        else:
            accum_grads = tree_map(lambda g: g / float(grad_accum), accum_grads)

        if use_master_weights:
            optimizer.master_params = optimizer.apply_gradients(accum_grads, optimizer.master_params)
            model.update(tree_map(lambda p: p.astype(mx.bfloat16), optimizer.master_params))
        else:
            optimizer.update(model, accum_grads)

        optimizer.state = cast_optimizer_moments_bf16(
            optimizer.state,
            keep_v_fp32=getattr(optimizer, "keep_v_fp32", False)
        )
        eval_targets = [model.parameters(), optimizer.state, accum_loss, accum_ce]
        if use_master_weights:
            eval_targets.append(optimizer.master_params)
        mx.eval(*eval_targets)
        return accum_loss, accum_ce

    # Fused accumulation scaling and norm clipping
    if grad_clip > 0.0:
        accum_grads, _ = clip_grad_norm_mlx(accum_grads, max_norm=grad_clip, scale=float(grad_accum))
    else:
        accum_grads = tree_map(lambda g: g / float(grad_accum), accum_grads)

    compiled_opt = getattr(optimizer, "_compiled_opt_update", None)
    if compiled_opt is None:
        if use_master_weights:
            # Compiled update applying float32 gradients to master weights and projecting to bfloat16 model
            def update_fn(master_dict, g):
                new_master = optimizer.apply_gradients(g, master_dict)
                new_model = tree_map(lambda p: p.astype(mx.bfloat16), new_master)
                return new_master, new_model

            compiled_opt = mx.compile(update_fn, inputs=[optimizer.state], outputs=[optimizer.state])
        else:
            def update_fn(grads_inner):
                optimizer.update(model, grads_inner)
                return model.parameters(), optimizer.state

            compiled_opt = mx.compile(update_fn, inputs=[model.state, optimizer.state], outputs=[model.state, optimizer.state])
        optimizer._compiled_opt_update = compiled_opt

    if use_master_weights:
        optimizer.master_params, new_model = compiled_opt(optimizer.master_params, accum_grads)
        model.update(new_model)
        mx.eval(optimizer.master_params, model.parameters(), optimizer.state, accum_loss, accum_ce)
    else:
        compiled_opt(accum_grads)
        mx.eval(model.parameters(), optimizer.state, accum_loss, accum_ce)

    return accum_loss, accum_ce


class ExperimentalAccumulationStep(MLXTrainingStep):
    """Select an experiment only for calls through this benchmark wrapper."""

    def __init__(self, *args, pipeline_accumulation=False, compiled_accumulation=False,
                 defer_last_microbatch=False, **kwargs):
        super().__init__(*args, **kwargs)

        def accumulate(previous, current):
            return tree_map(lambda a, b: a + b, previous, current)

        compiled = mx.compile(accumulate) if compiled_accumulation else None

        def execute(*positional, **keywords):
            return execute_mlx_training_step(
                *positional, **keywords, pipeline_accumulation=pipeline_accumulation,
                accumulate_fn=compiled, defer_last_microbatch=defer_last_microbatch,
            )

        self._execute_experiment = execute

    def __call__(self, *args, **kwargs):
        # Trials run in separate processes; do not share this wrapper across threads.
        with patch('telos.training.mlx_step.execute_mlx_training_step', self._execute_experiment):
            return super().__call__(*args, **kwargs)
