"""Causal reachability matters more than observing different scheduler values."""
import torch
from scripts.audit_cadence_paths import gradient_probe
from telos.training.schedule import COROSredSchedule


def test_metric_feedback_does_not_change_nominal_backbone_weights():
    schedule=COROSredSchedule(max_steps=12716,alpha_min=.5,beta_max=.35)
    for step in [0,2543,4000,8000,12716]:
        below=schedule.get_weights(step,.5,.5)
        above=schedule.get_weights(step,.8,.8)
        assert below['alpha']==above['alpha']
        assert below['beta']==above['beta']
        assert below['gamma']==0.
        assert above['gamma']==.1


def test_routing_refresh_without_metrics_remains_disabled():
    schedule=COROSredSchedule(max_steps=12716)
    for step in [1,4000,8000,12716]:
        assert schedule.get_weights(step)['mask_blend']==0.
        assert schedule.get_weights(step,.8,.8)['mask_blend']==1.


def test_detached_head_can_affect_backbone_only_after_shared_clipping_here():
    torch.set_num_threads(1)
    result=gradient_probe()
    assert result['max_backbone_gradient_difference_before_clip']==0.
    assert result['head_gradient_norm_gamma0']==0.
    assert result['head_gradient_norm_gamma_point1']>0.
    assert result['max_backbone_gradient_difference_with_forced_active_clip_point25']>0.

