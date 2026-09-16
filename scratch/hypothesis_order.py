"""Instrument: print initial weights of eager vs core run in the exact order the
failing test calls them, to find where they diverge."""
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map

LRS = (1e-3, 5e-4, 2e-4)


def cast_moments_bf16(state):
    def cast_state(v):
        if isinstance(v, dict):
            return {k: (val.astype(mx.bfloat16) if k in ("m", "v") and isinstance(val, mx.array) and val.dtype == mx.float32
                        else cast_state(val) if isinstance(val, dict) else val)
                    for k, val in v.items()}
        return v
    return cast_state(state)


def run_core_path(lrs):
    model = nn.Linear(4, 4)
    model.update(tree_map(lambda p: p.astype(mx.bfloat16), model.parameters()))
    w_init = mx.array(model.parameters()["weight"])
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
        if step == 0:
            master = opt.apply_gradients(g, master)
            model.update(tree_map(lambda p: p.astype(mx.bfloat16), master))
            opt.state = cast_moments_bf16(opt.state)

            def update_fn(md, gg):
                nm = opt.apply_gradients(gg, md)
                nmod = tree_map(lambda p: p.astype(mx.bfloat16), nm)
                return nm, nmod
            compiled_opt = mx.compile(update_fn, inputs=[opt.state], outputs=[opt.state])
        else:
            master, nmod = compiled_opt(master, g)
            model.update(nmod)
        mx.eval(model.parameters(), master, opt.state)
    return w_init, master["weight"]


def run_eager_fp32(lrs):
    model = nn.Linear(4, 4)
    model.update(tree_map(lambda p: p.astype(mx.bfloat16), model.parameters()))
    w_init = mx.array(model.parameters()["weight"])
    opt = optim.AdamW(learning_rate=lrs[0], betas=[0.9, 0.95], bias_correction=True)
    master = None
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
        master = opt.apply_gradients(g, master)
        model.update(tree_map(lambda p: p.astype(mx.bfloat16), master))
        mx.eval(model.parameters(), master, opt.state)
    return w_init, master["weight"]


# exact call order of the failing test
w_e_init, w_e = run_eager_fp32(LRS)
w_c_init, w_c = run_core_path(LRS)

print("init diff:", mx.abs(w_e_init - w_c_init).max().item())
print("final diff:", mx.abs(w_e - w_c).max().item())

# And the reverse order
w_c2_init, w_c2 = run_core_path(LRS)
w_e2_init, w_e2 = run_eager_fp32(LRS)
print("reverse order init diff:", mx.abs(w_e2_init - w_c2_init).max().item())
print("reverse order final diff:", mx.abs(w_e2 - w_c2).max().item())
