import json
import re

with open("notebooks/shared/Optimization_Test_Suite.ipynb", "r") as f:
    nb = json.load(f)

# Cell 1 (Imports)
for cell in nb["cells"]:
    if cell["cell_type"] == "code" and "from mdiff.training.trainer" in "".join(cell["source"]):
        new_source = []
        for line in cell["source"]:
            if "apply_masking_mlx, loss_fn_mlx, build_special_token_lut" in line:
                new_source.append("from telos.training.core import clip_grad_norm_mlx, build_special_token_lut, get_sys_mem_str, cast_optimizer_moments_bf16, execute_mlx_training_step\n")
                new_source.append("from mdiff.training.trainer import apply_masking_mlx, loss_fn_mlx\n")
            else:
                new_source.append(line)
        cell["source"] = new_source

# Cell 2 (Benchmark runner)
for cell in nb["cells"]:
    if cell["cell_type"] == "code" and "def benchmark_patch(" in "".join(cell["source"]):
        source_str = "".join(cell["source"])
        
        # Remove clip_grad_norm_mlx
        source_str = re.sub(r'def clip_grad_norm_mlx\(.*?\n    return clipped, total_norm\n\n', '', source_str, flags=re.DOTALL)
        
        # Replace warmup loop
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
        source_str = re.sub(warmup_pattern, warmup_replacement, source_str, flags=re.DOTALL)

        # Replace bench loop
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
        source_str = re.sub(bench_pattern, bench_replacement, source_str, flags=re.DOTALL)

        # Re-split into lines
        cell["source"] = [line + "\n" for line in source_str.split("\n")][:-1]

with open("notebooks/shared/Optimization_Test_Suite.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
