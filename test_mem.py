import mlx.core as mx
import mlx.nn as nn
import gc

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

print("Starting memory test...")
for i in range(100):
    x = mx.random.normal((1000, 1000))
    loss, grads = compiled_step(x)
    mx.eval(grads, loss) # evaluate outputs
    if i % 10 == 0:
        mx.clear_cache()
        gc.collect()
        print(f"Iter {i}, Peak Mem: {mx.get_peak_memory() / 1e6} MB, Active: {mx.get_active_memory() / 1e6} MB")

