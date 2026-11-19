"""CPU numerical/replay checks; TPU compilation requires physical hardware."""
from copy import deepcopy
import pytest
import torch
from telos.training.lr_schedule import WarmupCosineLR
from telos.training.xla_optimizer import XLAAdamW


@pytest.mark.parametrize('amsgrad', [False, True])
@pytest.mark.parametrize('weight_decay', [0., .1])
def test_dynamic_controls_match_native_adamw_and_resume(amsgrad, weight_decay):
    torch.manual_seed(81)
    initial = torch.randn(4, 8)
    a, b = [torch.nn.Parameter(initial.clone()) for _ in range(2)]
    opts = [cls([p], lr=.003, betas=(.9, .95), amsgrad=amsgrad, weight_decay=weight_decay)
            for cls, p in [(torch.optim.AdamW, a), (XLAAdamW, b)]]
    schedulers = [WarmupCosineLR(o, 5, 20, update_cadence=3) for o in opts]
    for step in range(20):
        grad = torch.randn_like(a) if step != 3 else torch.zeros_like(a)
        for p, opt, sched in zip((a, b), opts, schedulers):
            p.grad = grad.clone()
            opt.step()
            sched.step()
        assert schedulers[0].get_last_lr() == schedulers[1].get_last_lr()
        torch.testing.assert_close(b, a, rtol=1e-6, atol=2e-7)
        if step == 8:
            # Native checkpoints must load, and our own replay must be exact.
            restored = torch.nn.Parameter(b.detach().clone())
            opt = XLAAdamW([restored])
            opt.load_state_dict(deepcopy(opts[1].state_dict()))
            a.grad = b.grad = restored.grad = grad.clone()
            before_extra = b.detach().clone()
            replay = deepcopy(opts[1].state_dict())
            opts[1].step(); opt.step()
            torch.testing.assert_close(restored, b, rtol=0, atol=0)
            # Restore pre-extra-step state and weight for the schedule comparison.
            opts[1].load_state_dict(replay)
            with torch.no_grad(): b.copy_(before_extra)
            # Resume a native AdamW checkpoint through the dynamic path.
            opt.load_state_dict(deepcopy(opts[0].state_dict()))
            assert opt.state[restored]['step'].device.type == 'cpu'
