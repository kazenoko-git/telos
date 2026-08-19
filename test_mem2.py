import mlx.core as mx
import mlx.nn as nn
import gc
from mlx.utils import tree_map

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = mx.random.normal((1000, 1000))
    def __call__(self, x):
        return x @ self.w

model = SimpleModel()
def loss_fn(model, x):
    return mx.sum(model(x))

loss_and_grad = nn.value_and_grad(model, loss_fn)

def step(x):
    return loss_and_grad(model, x)

state = [model.state]
compiled_step = mx.compile(step, inputs=state, outputs=state)

print("Starting memory test with accumulation...")
for i in range(10):
    accum_grads = None
    accum_loss = mx.array(0.0)
    
    for j in range(64): # Simulate grad_accum=64
        x = mx.random.normal((100, 1000))
        loss, grads = compiled_step(x)
        
        if accum_grads is None:
            accum_grads = grads
        else:
            accum_grads = tree_map(lambda a, b: a + b, accum_grads, grads)
        accum_loss = accum_loss + loss
        mx.eval(accum_grads, accum_loss)
        
    print(f"Iter {i}, Peak Mem: {mx.get_peak_memory() / 1e6} MB, Active: {mx.get_active_memory() / 1e6} MB")

