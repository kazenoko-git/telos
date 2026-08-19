import re

with open('scripts/shared/run_optimization_suite.py', 'r') as f:
    content = f.read()

# Replace imports
old_import = """from mdiff.training.trainer import (
    apply_masking_mlx, loss_fn_mlx, build_special_token_lut,
    cast_optimizer_moments_bf16, clip_grad_norm_mlx, get_sys_mem_str
)"""
new_import = """from telos.training.core import (
    clip_grad_norm_mlx, build_special_token_lut, get_sys_mem_str,
    cast_optimizer_moments_bf16, execute_mlx_training_step
)
from mdiff.training.trainer import apply_masking_mlx, loss_fn_mlx"""
content = content.replace(old_import, new_import)

# Warmup loop replacement
warmup_pattern = r'    for _ in range\(warmup_steps\):\n        accum_grads = None\n.*?mx.eval\(model.parameters\(\), optimizer.state\)\n'

warmup_replacement = """    def random_batch_gen():
        while True:
            yield mx.random.randint(0, vocab_size, (micro_batch, seq_len))
            
    batch_gen = random_batch_gen()

    for _ in range(warmup_steps):
        execute_mlx_training_step(
            model=model,
            optimizer=optimizer,
            compiled_step_fn=compiled_step,
            batch_iterator=batch_gen,
            grad_accum=grad_accum,
            grad_clip=1.0,
            is_first_step=False
        )\n"""

content = re.sub(warmup_pattern, warmup_replacement, content, flags=re.DOTALL)

# Bench loop replacement
bench_pattern = r'    start_time = time.perf_counter\(\)\n    for step in range\(bench_steps\):\n        accum_grads = None\n.*?mx.eval\(model.parameters\(\), optimizer.state\)\n'

bench_replacement = """    start_time = time.perf_counter()
    for step in range(bench_steps):
        execute_mlx_training_step(
            model=model,
            optimizer=optimizer,
            compiled_step_fn=compiled_step,
            batch_iterator=batch_gen,
            grad_accum=grad_accum,
            grad_clip=1.0,
            is_first_step=False
        )\n"""

content = re.sub(bench_pattern, bench_replacement, content, flags=re.DOTALL)

with open('scripts/shared/run_optimization_suite.py', 'w') as f:
    f.write(content)
