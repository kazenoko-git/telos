"""
Regression tests for:
1. Dynamic LR updates inside mx.compile (ensuring scheduler changes are picked up)
2. fp32 master weights precision & checkpoint resume synchronization
"""

import pytest
import numpy as np


def test_mlx_compile_dynamic_learning_rate():
    """Verify that mx.compile dynamically observes changes to optimizer.learning_rate."""
    try:
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
    except ImportError:
        pytest.skip("MLX not installed in this environment")

    mx.random.seed(42)

    def make():
        model = nn.Linear(4, 4)
        opt = optim.AdamW(learning_rate=1e-3, betas=[0.9, 0.95], bias_correction=True)
        return model, opt

    def grad_of(model, x):
        def loss_fn(m):
            return (m(x) ** 2).sum()
        return nn.value_and_grad(model, loss_fn)(model)

    x = mx.random.normal((2, 4))

    # Eager reference
    model_eager, opt_eager = make()
    w0_eager = mx.array(model_eager.parameters()["weight"])
    _, g = grad_of(model_eager, x)
    opt_eager.update(model_eager, g)
    mx.eval(model_eager.parameters())
    w1_eager = mx.array(model_eager.parameters()["weight"])
    opt_eager.learning_rate = 100.0
    _, g = grad_of(model_eager, x)
    opt_eager.update(model_eager, g)
    mx.eval(model_eager.parameters())
    w2_eager = mx.array(model_eager.parameters()["weight"])

    eager_d1 = mx.abs(w1_eager - w0_eager).max().item()
    eager_d2 = mx.abs(w2_eager - w1_eager).max().item()

    # Compiled execution
    model_comp, opt_comp = make()
    w0_comp = mx.array(model_comp.parameters()["weight"])
    _, g = grad_of(model_comp, x)

    def update_fn(grads_inner):
        opt_comp.update(model_comp, grads_inner)
        return model_comp.parameters(), opt_comp.state

    compiled_opt = mx.compile(update_fn, inputs=[model_comp.state, opt_comp.state], outputs=[model_comp.state, opt_comp.state])

    compiled_opt(g)
    mx.eval(model_comp.parameters())
    w1_comp = mx.array(model_comp.parameters()["weight"])

    # Change LR dynamically on optimizer
    opt_comp.learning_rate = 100.0
    _, g = grad_of(model_comp, x)
    compiled_opt(g)
    mx.eval(model_comp.parameters())
    w2_comp = mx.array(model_comp.parameters()["weight"])

    comp_d1 = mx.abs(w1_comp - w0_comp).max().item()
    comp_d2 = mx.abs(w2_comp - w1_comp).max().item()

    # Step 1 delta should closely match eager
    assert abs(comp_d1 - eager_d1) / max(eager_d1, 1e-12) < 1e-3, "Compiled step 1 does not match eager"
    # Step 2 delta should scale up ~1e5x when lr changes from 1e-3 to 100.0
    assert comp_d2 > 1000 * comp_d1, f"Compiled optimizer failed to pick up dynamic learning rate update: {comp_d2} vs {comp_d1}"


def test_pytorch_master_weights_precision_and_resume():
    """Verify that master weights prevent precision underflow and synchronize on resume."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        pytest.skip("PyTorch not installed in this environment")

    # 1. Test precision underflow prevention with master weights
    p_bf16 = nn.Parameter(torch.ones(10, dtype=torch.bfloat16))
    p_master = nn.Parameter(p_bf16.detach().clone().float())

    opt = torch.optim.AdamW([p_master], lr=1e-5, eps=1e-8)
    
    # Tiny gradient: ~1e-4. In bf16, an update of 1e-5 * 1e-4 = 1e-9 would completely underflow
    g = torch.full_like(p_bf16, 1e-4)
    p_master.grad = g.float()
    opt.step()

    # Master weights have moved by a small amount in float32
    delta_fp32 = (p_master.data - 1.0).abs().max().item()
    assert delta_fp32 > 0, "Master weights failed to register float32 update"

    # Copy to bf16
    p_bf16.data.copy_(p_master.data.to(torch.bfloat16))

    # 2. Test checkpoint resume synchronization
    from telos.models import TelosTransformer
    from telos.training import UnifiedPyTorchTrainer

    cfg = {
        "model": {"vocab_size": 256, "d_model": 64, "n_layers": 2, "n_heads": 2, "seq_len": 32},
        "training": {"max_steps": 10, "precision": "bf16", "batch_size": 2, "gradient_accumulation": 1},
        "checkpoint": {"save_every_steps": 10}
    }
    model = TelosTransformer(vocab_size=256, d_model=64, n_layers=2, n_heads=2, seq_len=32)
    trainer = UnifiedPyTorchTrainer(paradigm="ar", model=model, cfg=cfg, device_type="cpu")
    trainer.use_master_weights = True
    trainer.param_to_master = [(p, p.detach().clone().float().requires_grad_(True)) for p in trainer.model.parameters() if p.requires_grad]

    # Stale master weight test: mutate model parameter to a known distinct value
    with torch.no_grad():
        for p in trainer.model.parameters():
            p.fill_(42.0)

    # Simulate checkpoint dict
    ckpt = {
        "global_step": 5,
        "model_state_dict": trainer.model.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "scheduler_state_dict": trainer.scheduler.state_dict(),
    }
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
        torch.save(ckpt, tmp.name)

        # Mutate master params to stale random values
        for _, mp in trainer.param_to_master:
            mp.data.fill_(-999.0)

        # Load checkpoint
        trainer.load_checkpoint(tmp.name)

        # Master weights MUST have been synced from the loaded model weights (42.0)
        for _, mp in trainer.param_to_master:
            assert torch.allclose(mp.data, torch.tensor(42.0)), "Master weights were not updated from loaded checkpoint!"
