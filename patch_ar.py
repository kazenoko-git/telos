import re

with open('ar/training/trainer.py', 'r') as f:
    content = f.read()

# Remove clip_grad_norm_mlx
content = re.sub(r'def clip_grad_norm_mlx\(.*?\n    return clipped, total_norm\n\n\n', '', content, flags=re.DOTALL)

# Remove build_special_token_lut
content = re.sub(r'def build_special_token_lut\(.*?\n    return mx.array\(lut, dtype=mx.bool_\)\n\n\n', '', content, flags=re.DOTALL)

# Remove get_sys_mem_str
content = re.sub(r'def get_sys_mem_str\(.*?return ""\n\n\n', '', content, flags=re.DOTALL)

# Remove cast_optimizer_moments_bf16
content = re.sub(r'def cast_optimizer_moments_bf16\(.*?\n    return new_state\n\n\n', '', content, flags=re.DOTALL)

# Add imports
import_str = """
from telos.training.core import (
    clip_grad_norm_mlx, build_special_token_lut, get_sys_mem_str,
    cast_optimizer_moments_bf16, execute_mlx_training_step
)
"""
if "from telos.training.core" not in content:
    content = content.replace("MLX_AVAILABLE = False\n", "MLX_AVAILABLE = False\n" + import_str)

# Replace the inner loop with execute_mlx_training_step
loop_pattern = r'            accum_grads = None\n            accum_loss = mx.array\(0.0, dtype=mx.float32\)\n.*?\n            mx.eval\(self.model.parameters\(\), optimizer.state, accum_loss, accum_ce\)'

replacement = """            def batch_gen():
                for i in range(grad_accum):
                    yield global_targets[i * bs : (i + 1) * bs]

            accum_loss, accum_ce = execute_mlx_training_step(
                model=self.model,
                optimizer=optimizer,
                compiled_step_fn=microbatch_step,
                batch_iterator=batch_gen(),
                grad_accum=grad_accum,
                grad_clip=self.grad_clip,
                is_first_step=(step == resume_step + 1)
            )"""

content = re.sub(loop_pattern, replacement, content, flags=re.DOTALL)

with open('ar/training/trainer.py', 'w') as f:
    f.write(content)
