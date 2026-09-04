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
    return mx.array(lut, dtype=mx.bool_)

def cast_optimizer_moments_bf16(state_dict: dict) -> dict:
    """Casts AdamW moment tensors m and v to bfloat16 to reduce memory footprint by 50%."""
    new_state = {}
    for k, v in state_dict.items():
        if isinstance(v, dict):
            new_state[k] = cast_optimizer_moments_bf16(v)
        elif isinstance(v, mx.array) and k in ("m", "v") and v.dtype == mx.float32:
            new_state[k] = v.astype(mx.bfloat16)
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
        # to prevent activation memory leak during gradient accumulation.
        # On high-RAM devices (>=36GB RAM), skip intermediate eval to keep Metal pipeline saturated.
        if eval_every_microbatch:
            mx.eval(accum_grads, accum_loss, accum_ce)

    if is_first_step:
        # Fallback to eager update on the first step because we cast AdamW moments
        # to bfloat16 afterwards.
        if grad_clip > 0.0:
            accum_grads, _ = clip_grad_norm_mlx(accum_grads, max_norm=grad_clip, scale=float(grad_accum))
        else:
            accum_grads = tree_map(lambda g: g / float(grad_accum), accum_grads)
        optimizer.update(model, accum_grads)
        optimizer.state = cast_optimizer_moments_bf16(optimizer.state)
        mx.eval(model.parameters(), optimizer.state, accum_loss, accum_ce)
        return accum_loss, accum_ce

    compiled_opt = getattr(optimizer, "_compiled_opt_update", None)
    if compiled_opt is None:
        def update_fn(grads_inner):
            optimizer.update(model, grads_inner)
            return model.parameters(), optimizer.state

        compiled_opt = mx.compile(update_fn, inputs=[model.state, optimizer.state], outputs=[model.state, optimizer.state])
        optimizer._compiled_opt_update = compiled_opt

    # Fused accumulation scaling and norm clipping
    if grad_clip > 0.0:
        accum_grads, _ = clip_grad_norm_mlx(accum_grads, max_norm=grad_clip, scale=float(grad_accum))
    else:
        accum_grads = tree_map(lambda g: g / float(grad_accum), accum_grads)

    compiled_opt(accum_grads)
    mx.eval(model.parameters(), optimizer.state, accum_loss, accum_ce)

    return accum_loss, accum_ce


