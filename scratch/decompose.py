"""Decompose the master-weights divergence: compile effect vs bf16-moments effect."""
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map


def cast_moments_bf16(state):
    def cast_state(v):
        if isinstance(v, dict):
            return {k: (val.astype(mx.bfloat16) if k in ("m", "v") and isinstance(val, mx.array) and val.dtype == mx.float32
                        else cast_state(val) if isinstance(val, dict) else val)
                    for k, val in v.items()}
        return v
    return cast_state(state)


def run(bf16_moments: bool, use_compile: bool, lrs):
    mx.random.seed(0)
    model = nn.Linear(4, 4)
    model.update(tree_map(lambda p: p.astype(mx.bfloat16), model.parameters()))
    opt = optim.AdamW(learning_rate=lrs[0], betas=[0.9, 0.95], bias_correction=True)
    master = None
    compiled_opt = None
    for step in range(3):
        mx.random.seed(100 + step)
        xs = mx.random.normal((2, 4))

        def loss_fn(m):
            return (m(xs) ** 2).sum()
        _, g = nn.value_and_grad(model, loss_fn)(model)
        g = tree_map(lambda t: t.astype(mx.float32), g)
        if master is None:
            master = tree_map(lambda p: p.astype(mx.float32), model.parameters())
        opt.learning_rate = lrs[step]

        if step == 0 or not use_compile:
            master = opt.apply_gradients(g, master)
            model.update(tree_map(lambda p: p.astype(mx.bfloat16), master))
        else:
            master, nmod = compiled_opt(master, g)
            model.update(nmod)

        if step == 0 and bf16_moments:
            opt.state = cast_moments_bf16(opt.state)
        if step == 0 and use_compile:
            def update_fn(md, gg):
                nm = opt.apply_gradients(gg, md)
                nmod = tree_map(lambda p: p.astype(mx.bfloat16), nm)
                return nm, nmod
            compiled_opt = mx.compile(update_fn, inputs=[opt.state], outputs=[opt.state])
        mx.eval(model.parameters(), master, opt.state)
    return master['weight']


LRS = (1e-3, 5e-4, 2e-4)
wE = run(False, False, LRS)   # eager, fp32 moments
wD = run(True,  False, LRS)   # eager, bf16 moments
wC = run(True,  True,  LRS)   # compiled, bf16 moments (core.py exact)
print("BF16 MOMENTS effect (eager vs eager):  ", mx.abs(wE - wD).max().item())
print("COMPILE effect (eager vs compiled):    ", mx.abs(wD - wC).max().item())
print("TOTAL (fp32 eager vs core.py):         ", mx.abs(wE - wC).max().item())
