import mlx.core as mx
import mlx.nn as nn
import gc

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = mx.random.normal((1000, 1000))
    def __call__(self, x):
        return x @ self.w

print("Testing dynamic compilation leak...")
for i in range(100):
    model = SimpleModel()
    def step(x):
        return mx.sum(model(x))
    
    compiled = mx.compile(step, inputs=[model.state], outputs=[model.state])
    x = mx.random.normal((100, 1000))
    out = compiled(x)
    mx.eval(out)
    
    del model, step, compiled, out, x
    mx.clear_cache()
    gc.collect()
    
    if i % 10 == 0:
        print(f"Iter {i}, Peak Mem: {mx.get_peak_memory() / 1e6} MB, Active: {mx.get_active_memory() / 1e6} MB")

