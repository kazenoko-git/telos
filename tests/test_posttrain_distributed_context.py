import pytest
import torch
import telos.posttrain.trainer as trainer


@pytest.mark.parametrize('existing',[False,True])
def test_torchrun_group_is_cleaned_up_only_when_owned(monkeypatch,existing):
    monkeypatch.setenv('WORLD_SIZE','2')
    monkeypatch.setenv('LOCAL_RANK','1')
    calls=[]
    monkeypatch.setattr(torch.cuda,'set_device',lambda device:calls.append(('device',device)))
    monkeypatch.setattr(torch.distributed,'is_initialized',lambda:existing)
    monkeypatch.setattr(torch.distributed,'init_process_group',lambda backend:calls.append(('init',backend)))
    monkeypatch.setattr(torch.distributed,'destroy_process_group',lambda:calls.append(('destroy',)))
    def fail(*args,**kwargs):
        raise RuntimeError('training failed')
    monkeypatch.setattr(trainer,'_posttrain',fail)
    with pytest.raises(RuntimeError,match='training failed'):
        trainer.posttrain('source','data','out',hardware='cuda')
    assert (('init','nccl') in calls) is not existing
    assert (('destroy',) in calls) is not existing
