"""Empirical verification of the master-weights optimizer path in telos/training/core.py.

Result established by scratch/decompose.py:
  - COMPILE effect: 0.0 (mx.compile with pytree signature is numerically exact)
  - BF16-moments effect: ~2.5e-6 (intentional memory optimization drift)
"""
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map

LRS = (1e-3, 5e-4, 2e-4)
FROZEN = (1e-3, 1e-3, 1e-3)


def cast_moments_bf16(state):
    def cast_state(v):
        if isinstance(v, dict):
            return {k: (val.astype(mx.bfloat16) if k in ("m", "v") and isinstance(val, mx.array) and val.dtype == mx.float32
                        else cast_state(val) if isinstance(val, dict) else val)
                    for k, val in v.items()}
        return v
    return cast_state(state)


def run_core_path(lrs):
    """Exact core.py master-weights branch: eager step 0, bf16 moments cast after,
    compiled updates (pytree signature) thereafter."""
    mx.random.seed(0)  # deterministic model init (test hygiene, not in core.py)
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
    return model, master, opt


def run_eager_fp32(lrs):
    """Reference: eager updates, fp32 moments, same data and lr schedule."""
    mx.random.seed(0)  # identical model init to run_core_path
    model = nn.Linear(4, 4)
    model.update(tree_map(lambda p: p.astype(mx.bfloat16), model.parameters()))
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
    return model, master, opt


m_e, mp_e, _ = run_eager_fp32(LRS)
m_c, mp_c, _ = run_core_path(LRS)
_, mp_f, _ = run_core_path(FROZEN)

w_c = m_c.parameters()["weight"]
mpc, mpf, mpe = mp_c["weight"], mp_f["weight"], mp_e["weight"]

print(f"model weight dtype:   {w_c.dtype}  (expect bfloat16)")
print(f"master weight dtype:  {mpc.dtype}  (expect float32)")
diff = mx.abs(mpe - mpc).max().item()
print(f"eager-fp32 vs core.py path max diff: {diff:.6e}  (bf16-moment drift, expect < 1e-4)")
diff_lr = mx.abs(mpc - mpf).max().item()
print(f"scheduled vs frozen-lr max diff:     {diff_lr:.6e}  (expect > 1e-5 → lr propagates)")

ok = (
    w_c.dtype == mx.bfloat16
    and mpc.dtype == mx.float32
    and diff < 1e-4
    and diff_lr > 1e-5
)
print(f"\nALL CHECKS {'PASSED' if ok else 'FAILED'}")
assert ok
