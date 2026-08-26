import math
import subprocess
try:
    import mlx.core as mx
    from mlx.utils import tree_map, tree_flatten
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

def get_sys_mem_str() -> str:
    """Returns Apple Silicon Metal unified memory and swap usage string."""
    try:
        active_gb = mx.get_active_memory() / 1e9
        peak_gb = mx.get_peak_memory() / 1e9
        swap_res = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True)
        swap_parts = swap_res.stdout.strip().split()
        used_swap = swap_parts[6] if len(swap_parts) >= 7 else "0M"
        return f"Metal Unified GPU: {active_gb:.2f}GB (Peak: {peak_gb:.2f}GB) | Swap: {used_swap}"
    except Exception:
        return ""

def clip_grad_norm_mlx(grads, max_norm: float = 1.0):
    """Clips global gradient L2 norm to max_norm in float32 for numerical stability."""
    total_norm_sq = mx.array(0.0, dtype=mx.float32)
    for _, g in tree_flatten(grads):
        total_norm_sq = total_norm_sq + mx.sum(g.astype(mx.float32) ** 2)
    total_norm = mx.sqrt(total_norm_sq)
    clip_coef = max_norm / (total_norm + 1e-6)
    scale = mx.minimum(mx.array(1.0, dtype=mx.float32), clip_coef)
    clipped = tree_map(lambda g: g * scale.astype(g.dtype), grads)
    return clipped, total_norm

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
    is_first_step: bool
):
    """
    Executes a single unified training step including microbatch accumulation, 
    memory evaluation, gradient clipping, and optimizer step.
    
    Args:
        model: MLX model.
        optimizer: MLX optimizer (e.g. AdamW).
        compiled_step_fn: function taking a batch and returning (loss, ce, grads).
        batch_iterator: Iterator or generator that yields `batch_seqs` for `grad_accum` times.
        grad_accum: Number of micro-batches to accumulate over.
        grad_clip: Maximum L2 norm for gradients.
        is_first_step: True if this is the first training step (used to trigger bfloat16 moment casting).
        
    Returns:
        accum_loss (mx.array): Summed loss across microbatches.
        accum_ce (mx.array): Summed cross entropy across microbatches.
    """
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
        
        # Evaluate intermediate graph to prevent memory leak during accumulation
        mx.eval(accum_grads, accum_loss, accum_ce)

    # Average gradients over accumulation steps
    accum_grads = tree_map(lambda g: g / grad_accum, accum_grads)
    
    # Apply gradient clipping (L2 norm <= grad_clip)
    if grad_clip > 0.0:
        accum_grads, _ = clip_grad_norm_mlx(accum_grads, max_norm=grad_clip)
    
    optimizer.update(model, accum_grads)
    
    # Cast AdamW moments to bf16 on the first step to save 50% optimizer memory
    if is_first_step:
        optimizer.state = cast_optimizer_moments_bf16(optimizer.state)
        
    # Single evaluation of parameters and optimizer state per step (no inner loop sync stalls)
    mx.eval(model.parameters(), optimizer.state, accum_loss, accum_ce)
    
    return accum_loss, accum_ce
