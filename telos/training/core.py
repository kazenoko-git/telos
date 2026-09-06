import math
import subprocess
try:
    import mlx.core as mx
    from mlx.utils import tree_map, tree_flatten
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

_swap_cache = {"value": "0M", "counter": 0}

def get_sys_mem_str() -> str:
    """Returns Apple Silicon Metal unified memory and swap usage string."""
    global _swap_cache
    try:
        active_gb = mx.get_active_memory() / 1e9
        peak_gb = mx.get_peak_memory() / 1e9
        _swap_cache["counter"] += 1
        # Refresh sysctl swap usage only every 5th log call to avoid subprocess overhead and die
        if _swap_cache["counter"] % 5 == 1:
            swap_res = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True)
            swap_parts = swap_res.stdout.strip().split()
            _swap_cache["value"] = swap_parts[6] if len(swap_parts) >= 7 else "0M"
        return f"Metal Unified GPU: {active_gb:.2f}GB (Peak: {peak_gb:.2f}GB) | Swap: {_swap_cache['value']}"
    except Exception:
        return ""

def clip_grad_norm_mlx(grads, max_norm: float = 1.0, scale: float = 1.0): # FIX: Fuse accum scaling + clipping into single tree traversal
    """Clips global gradient L2 norm to max_norm in float32 for numerical stability."""
    total_norm_sq = mx.array(0.0, dtype=mx.float32)
    for _, g in tree_flatten(grads): total_norm_sq = total_norm_sq + mx.sum(g.astype(mx.float32) ** 2)
    total_norm = mx.sqrt(total_norm_sq)
    # Compute effective norm after scaling: ||g / scale||₂ = ||g||₂ / scale
    effective_norm = total_norm / scale if scale != 1.0 else total_norm
    clip_coef = max_norm / (effective_norm + 1e-6)
    # Fused coefficient: (1 / scale) * min(1, clip_coef)
    combined = mx.minimum(mx.array(1.0, dtype=mx.float32), clip_coef) / scale
    clipped = tree_map(lambda g: g * combined.astype(g.dtype), grads)
    return clipped, effective_norm

def build_special_token_lut(vocab_size: int, special_tokens=(0, 1, 2, 3)):
    """Precomputes 1D boolean array for constant-time special token lookup."""
    lut = [False] * vocab_size
    for token_id in special_tokens:
        if token_id < vocab_size:
            lut[token_id] = True
    if MLX_AVAILABLE:
        return mx.array(lut, dtype=mx.bool_)
    try:
        import numpy as np
        return np.array(lut, dtype=bool)
    except ImportError:
        return lut

def cast_optimizer_moments_bf16(state_dict: dict, keep_v_fp32: bool = False) -> dict:
    """Casts AdamW moment tensors m (and optionally v) to bfloat16 to reduce memory footprint by ~50%.
    
    Args:
        state_dict: Optimizer state dictionary containing 'm' and 'v' moment arrays.
        keep_v_fp32: If True, retains the second moment 'v' in float32 for late-training numerical stability.
    """
    new_state = {}
    for k, v in state_dict.items():
        if isinstance(v, dict):
            new_state[k] = cast_optimizer_moments_bf16(v, keep_v_fp32=keep_v_fp32)
        elif isinstance(v, mx.array) and v.dtype == mx.float32:
            if k == "m":
                new_state[k] = v.astype(mx.bfloat16)
            elif k == "v" and not keep_v_fp32:
                new_state[k] = v.astype(mx.bfloat16)
            else:
                new_state[k] = v
        else:
            new_state[k] = v
    return new_state

def execute_mlx_training_step(
    model,
    optimizer,
    compiled_step_fn,
    batch_iterator,
    grad_accum: int,
    grad_clip: float,
    is_first_step: bool,
    eval_every_microbatch: bool = False
):
    accum_grads = None
    accum_loss = mx.array(0.0, dtype=mx.float32)
    accum_ce = mx.array(0.0, dtype=mx.float32)

    # Double-buffering state: tracks previous microbatch targets to overlap CPU eval with GPU compute
    prev_accum_grads = None
    prev_accum_loss = None
    prev_accum_ce = None

    for i in range(grad_accum):
        batch_data = next(batch_iterator)
        if isinstance(batch_data, tuple):
            loss, ce, grads = compiled_step_fn(*batch_data)
        else:
            loss, ce, grads = compiled_step_fn(batch_data)
        
        if accum_grads is None:
            accum_grads = grads
        else:
            accum_grads = tree_map(lambda a, b: a + b, accum_grads, grads)
        
        accum_loss = accum_loss + loss
        accum_ce = accum_ce + ce
        
        # On memory-constrained devices (<24GB RAM), evaluate intermediate graph
        # using a 1-microbatch lagged double-buffer:
        # submit microbatch i to the Metal stream before waiting on microbatch (i-1)'s accumulated grads.
        # This keeps the GPU queue constantly saturated while CPU handles evaluation.
        if eval_every_microbatch:
            if prev_accum_grads is not None:
                mx.eval(prev_accum_grads, prev_accum_loss, prev_accum_ce)
            prev_accum_grads = accum_grads
            prev_accum_loss = accum_loss
            prev_accum_ce = accum_ce

    # Evaluate the final microbatch if lagged evaluation was active
    if eval_every_microbatch and prev_accum_grads is not None:
        mx.eval(accum_grads, accum_loss, accum_ce)

    use_master_weights = getattr(model, "use_master_weights", False)
    if use_master_weights and not hasattr(optimizer, "master_params"):
        # Initialize fp32 master weights on optimizer to prevent polluting model.parameters()
        optimizer.master_params = tree_map(lambda p: p.astype(mx.float32), model.parameters())

    if is_first_step:
        # Fallback to eager update on the first step because we cast AdamW moments
        # to bfloat16 afterwards.
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


