"""Real CUDA integration checks, runnable on the target hardware."""
import numpy as np
import pytest
import torch
from telos.training.prefetch import CUDAPrefetcher


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA hardware required')
def test_prefetch_wraparound_and_early_cancel():
    data=np.arange(11*8,dtype=np.uint16).reshape(11,8)
    loader=CUDAPrefetcher(data,9,4,8,4,torch.device('cuda:0'),100)
    try:
        for start in (9,2,6):
            batch=loader.get()
            expected=data[(np.arange(4)+start)%11].astype(np.int64)
            np.testing.assert_array_equal(batch.cpu().numpy(),expected)
    finally:
        loader.close()
    assert not loader.thread.is_alive()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA hardware required')
def test_native_cuda_gqa_backward():
    from telos.models import TelosTransformer
    model=TelosTransformer(vocab_size=64,d_model=128,n_layers=1,n_heads=2,n_kv_heads=1,is_causal=True).cuda()
    with torch.autocast('cuda',dtype=torch.float16):
        loss=model(torch.randint(0,64,(2,16),device='cuda')).float().square().mean()
    loss.backward()
    assert model.layers[0].attn.k_proj.weight.grad.isfinite().all()


def test_prefetch_staging_order_and_cancellation_on_cpu(monkeypatch):
    """Model copy/event behavior locally; the hardware test above remains separate."""
    class Stream:
        def synchronize(self): pass
        def wait_event(self,event): pass
    class Event:
        def synchronize(self): pass
        def record(self,stream): pass
    from contextlib import nullcontext
    original_empty,original_to=torch.empty,torch.Tensor.to
    monkeypatch.setattr(torch.cuda,'Stream',lambda **kwargs:Stream())
    monkeypatch.setattr(torch.cuda,'Event',Event)
    monkeypatch.setattr(torch.cuda,'set_device',lambda device:None)
    monkeypatch.setattr(torch.cuda,'stream',lambda stream:nullcontext())
    monkeypatch.setattr(torch.cuda,'current_stream',lambda device:Stream())
    def empty(*args,**kwargs):
        kwargs.pop('pin_memory',None)
        return original_empty(*args,**kwargs)
    def transfer(self,*args,**kwargs):
        return original_to(self,*args,**kwargs).clone()
    monkeypatch.setattr(torch,'empty',empty)
    monkeypatch.setattr(torch.Tensor,'to',transfer)
    monkeypatch.setattr(torch.Tensor,'record_stream',lambda self,stream:None)
    data=np.arange(11*8,dtype=np.uint16).reshape(11,8)
    loader=CUDAPrefetcher(data,9,4,8,4,torch.device('cpu'),100)
    try:
        for start in (9,2,6,10,3):
            expected=data[(np.arange(4)+start)%11].astype(np.int64)
            np.testing.assert_array_equal(loader.get().numpy(),expected)
    finally:
        loader.close()
    assert not loader.thread.is_alive()
