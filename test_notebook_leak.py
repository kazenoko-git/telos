import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import gc

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = mx.random.normal((1000, 1000))
    def __call__(self, x):
        return x @ self.w

def benchmark_patch():
    model = SimpleModel()
    def loss_fn(model, x):
        return mx.sum(model(x))
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    def step(x):
        l, g = loss_and_grad(model, x)
        return l, l, g
        
    compiled_step = mx.compile(step, inputs=[model.state], outputs=[model.state])
    optimizer = optim.AdamW(learning_rate=3e-4)
    
    def batch_gen():
        while True:
            yield mx.random.normal((100, 1000))
    
    bg = batch_gen()
    
    from telos.training.core import execute_mlx_training_step
    
    for _ in range(5):
        execute_mlx_training_step(model, optimizer, compiled_step, bg, grad_accum=64, grad_clip=1.0, is_first_step=False)
        
    del model, optimizer, compiled_step, bg, loss_fn, loss_and_grad, step
    gc.collect()
    mx.clear_cache()
    return mx.get_peak_memory() / 1e9

for i in range(10):
    peak = benchmark_patch()
    print(f"Iter {i}, Peak Mem: {peak} GB, Active: {mx.get_active_memory() / 1e9} GB")
    
