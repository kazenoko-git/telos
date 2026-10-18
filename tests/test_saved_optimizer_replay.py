"""Prevent a false optimizer-parity result from aliased replay states."""
import copy
import torch
from scripts.replay_saved_optimizer import build_optimizer,compare_tensors


def source_checkpoint():
    parameter=torch.nn.Parameter(torch.tensor([.5,-.2]))
    optimizer=torch.optim.AdamW([parameter],lr=.001,betas=(.9,.95))
    parameter.grad=torch.tensor([.2,-.3]);optimizer.step()
    return {'model_state_dict':{'weight':parameter.detach().clone()},
            'optimizer_state_dict':copy.deepcopy(optimizer.state_dict())}


def test_replay_optimizers_do_not_share_parameters_or_moments():
    checkpoint=source_checkpoint()
    a,oa=build_optimizer(checkpoint,[['weight']],False)
    b,ob=build_optimizer(checkpoint,[['weight']],True)
    before=b['weight'].detach().clone()
    moment=ob.state[b['weight']]['exp_avg'].clone()
    a['weight'].grad=torch.ones(2);oa.step()
    assert torch.equal(b['weight'],before)
    assert torch.equal(ob.state[b['weight']]['exp_avg'],moment)
    assert torch.equal(checkpoint['model_state_dict']['weight'],before)
    assert compare_tensors(a,b)['max_absolute']>0


def test_comparison_detects_different_prescribed_updates():
    checkpoint=source_checkpoint()
    a,oa=build_optimizer(checkpoint,[['weight']],False)
    b,ob=build_optimizer(checkpoint,[['weight']],True)
    a['weight'].grad=torch.ones(2)
    b['weight'].grad=-torch.ones(2)
    oa.step();ob.step()
    assert compare_tensors(a,b,checkpoint['model_state_dict'])['relative_l2_vs_plain_cumulative_update']>0
