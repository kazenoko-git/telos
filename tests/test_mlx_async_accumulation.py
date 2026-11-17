"""Bounded asynchronous accumulation must preserve update state and RNG order."""
import numpy as np
import pytest

mx=pytest.importorskip('mlx.core')
nn=pytest.importorskip('mlx.nn')
optim=pytest.importorskip('mlx.optimizers')
from mlx.utils import tree_flatten,tree_map
from telos.training.mlx_step import MLXTrainingStep,build_mlx_accumulation_step

if not mx.metal.is_available():pytest.skip('Metal required',allow_module_level=True)


def snapshot(tree):
    mx.eval(tree)
    return {key:np.asarray(value.astype(mx.float32) if value.dtype==mx.bfloat16 else value).copy()
        for key,value in tree_flatten(tree)}


def exercise(asynchronous,accumulation,compiled,master,eager=False):
    mx.random.seed(73)
    model=nn.Linear(8,4)
    model.set_dtype(mx.bfloat16 if master else mx.float32)
    model.use_master_weights=master
    def loss_fn(module,x):
        y=module(x)*mx.random.uniform(shape=(x.shape[0],4))
        loss=mx.mean((y.astype(mx.float32)-.25)**2)
        return loss,loss
    grad=nn.value_and_grad(model,loss_fn)
    def micro(x):
        (loss,metric),grads=grad(model,x)
        return loss,metric,grads
    state=[model.state,mx.random.state]
    opt=optim.AdamW(1e-3,betas=[.9,.95],bias_correction=True)
    step=MLXTrainingStep(model,opt,mx.compile(micro,inputs=state,outputs=state),
        accumulation_step=build_mlx_accumulation_step(micro,state) if compiled else None,
        async_accumulation=asynchronous,grad_clip=.1)
    losses=[]
    for lr in (1e-3,0.0,2e-3):
        opt.learning_rate=lr
        batches=(mx.full((3,8),(i+1)/accumulation) for i in range(accumulation))
        losses.append(snapshot(step(batches,grad_accum=accumulation,eval_every_microbatch=eager)))
    final=snapshot([model.parameters(),opt.state,getattr(opt,'master_params',{}),mx.random.uniform(shape=(16,))])
    return losses,final


@pytest.mark.parametrize('accumulation',[2,8])
@pytest.mark.parametrize('compiled',[False,True])
@pytest.mark.parametrize('master',[False,True])
def test_async_accumulation_preserves_losses_update_state_and_rng(accumulation,compiled,master):
    losses,expected=exercise(False,accumulation,compiled,master)
    actual_losses,actual=exercise(True,accumulation,compiled,master)
    assert actual.keys()==expected.keys()
    for want,got in zip(losses,actual_losses):
        for key in want:np.testing.assert_allclose(got[key],want[key],rtol=3e-5,atol=3e-6)
    for key in expected:np.testing.assert_allclose(actual[key],expected[key],rtol=3e-5,atol=3e-6)


def test_eager_policy_disables_async_dispatch(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('Eager policy must not dispatch asynchronously')
    monkeypatch.setattr(mx,'async_eval',forbidden)
    exercise(True,8,True,True,eager=True)


def test_async_dispatch_waits_before_queue_can_grow(monkeypatch):
    original_async,original_eval=mx.async_eval,mx.eval
    pending=[]
    largest=[0]
    def dispatch(*arrays):
        pending.append(tuple(id(value) for value in arrays))
        largest[0]=max(largest[0],len(pending))
        return original_async(*arrays)
    def evaluate(*arrays,**kwargs):
        handles=tuple(id(value) for value in arrays)
        if handles in pending:pending.remove(handles)
        result=original_eval(*arrays,**kwargs)
        # An optimizer-step or snapshot fence drains the dependency chain.
        if arrays and isinstance(arrays[0],(dict,list,tuple)):
            pending.clear()
        return result
    monkeypatch.setattr(mx,'async_eval',dispatch)
    monkeypatch.setattr(mx,'eval',evaluate)
    exercise(True,8,True,True)
    assert largest[0]==2


@pytest.mark.parametrize('asynchronous',[False,True])
def test_clipped_accumulated_gradients_match_explicit_reference(asynchronous):
    from telos.training.core import execute_mlx_training_step,clip_grad_norm_mlx
    mx.random.seed(17)
    model=nn.Linear(8,4)
    class RecordingAdamW(optim.AdamW):
        def apply_gradients(self,grads,params):
            self.observed=grads
            return super().apply_gradients(grads,params)
    opt=RecordingAdamW(1e-3)
    def objective(m,x):
        val=mx.mean(m(x).astype(mx.float32)**2)
        return val,val
    grad=nn.value_and_grad(model,objective)
    def micro(x):
        (loss,metric),grads=grad(model,x)
        return loss,metric,grads
    batches=[mx.full((3,8),(i+1)/8) for i in range(8)]
    summed=None
    for x in batches:
        _,_,g=micro(x)
        summed=g if summed is None else tree_map(lambda a,b:a+b,summed,g)
    expected=snapshot(clip_grad_norm_mlx(summed,.1,scale=8)[0])
    execute_mlx_training_step(model,opt,micro,iter(batches),8,.1,True,async_accumulation=asynchronous)
    actual=snapshot(opt.observed)
    for key in expected:np.testing.assert_allclose(actual[key],expected[key],rtol=1e-6,atol=1e-7)
