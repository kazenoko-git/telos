"""Hypothesis: the failing test diverges because the two runs use DIFFERENT model
initializations (no mx.random.seed(0) before each run), unlike decompose.py which seeds."""
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map


def cast_moments(state):
    def cs(v):
        if isinstance(v, dict):
            return {k: (val.astype(mx.bfloat16) if k in ("m", "v") and isinstance(val, mx.array) and val.dtype == mx.float32
                        else cs(val) if isinstance(val, dict) else val)
                    for k, val in v.items()}
        return v
    return cs(state)


LRS = (1e-3, 5e-4, 2e-4)


def core_path(seed_init):
    if seed_init:
        mx.random.seed(0)
    model = nn.Linear(4, 4)
    model.update(tree_map(lambda p: p.astype(mx.bfloat16), model.parameters()))
    opt = optim.AdamW(learning_rate=LRS[0], betas=[0.9, 0.95], bias_correction=True)
    master = None
    comp = None
    for step in range(3):
        mx.random.seed(100 + step)
        xs = mx.random.normal((2, 4))

        def lf(m):
            return (m(xs) ** 2).sum()
        _, g = nn.value_and_grad(model, lf)(model)
        g = tree_map(lambda t: t.astype(mx.float32), g)
        if master is None:
            master = tree_map(lambda p: p.astype(mx.float32), model.parameters())
        opt.learning_rate = LRS[step]
        if step == 0:
            master = opt.apply_gradients(g, master)
            model.update(tree_map(lambda p: p.astype(mx.bfloat16), master))
            opt.state = cast_moments(opt.state)

            def uf(md, gg):
                nm = opt.apply_gradients(gg, md)
                return nm, tree_map(lambda p: p.astype(mx.bfloat16), nm)
            comp = mx.compile(uf, inputs=[opt.state], outputs=[opt.state])
        else:
            master, nmod = comp(master, g)
            model.update(nmod)
        mx.eval(model.parameters(), master, opt.state)
    return master['weight']


def eager_fp32(seed_init):
    if seed_init:
        mx.random.seed(0)
    model = nn.Linear(4, 4)
    model.update(tree_map(lambda p: p.astype(mx.bfloat16), model.parameters()))
    opt = optim.AdamW(learning_rate=LRS[0], betas=[0.9, 0.95], bias_correction=True)
    master = None
    for step in range(3):
        mx.random.seed(100 + step)
        xs = mx.random.normal((2, 4))

        def lf(m):
            return (m(xs) ** 2).sum()
        _, g = nn.value_and_grad(model, lf)(model)
        g = tree_map(lambda t: t.astype(mx.float32), g)
        if master is None:
            master = tree_map(lambda p: p.astype(mx.float32), model.parameters())
        opt.learning_rate = LRS[step]
        master = opt.apply_gradients(g, master)
        model.update(tree_map(lambda p: p.astype(mx.bfloat16), master))
        mx.eval(model.parameters(), master, opt.state)
    return master['weight']


wc_s = core_path(seed_init=True)
we_s = eager_fp32(seed_init=True)
print("WITH initial seed(0)    -> diff:", mx.abs(wc_s - we_s).max().item())

wc_n = core_path(seed_init=False)
we_n = eager_fp32(seed_init=False)
print("WITHOUT initial seed    -> diff:", mx.abs(wc_n - we_n).max().item())
