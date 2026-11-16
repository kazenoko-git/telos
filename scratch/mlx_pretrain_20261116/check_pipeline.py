"""Exact state/RNG and error-drain gate for the dispatch experiment."""
import json
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from telos.training.mlx_step import MLXTrainingStep, build_mlx_accumulation_step
import pipeline
import sys
if "--final" in sys.argv:
    import final_update as probe
else:
    probe = pipeline


def snapshot(tree):
    values=tree_flatten(tree)
    mx.eval(*[v for _,v in values])
    return [(k,str(v.dtype),np.array(v.astype(mx.float32) if v.dtype==mx.bfloat16 else v)) for k,v in values]


def run(pipelined, master, accumulated, compiled, grad_clip=1.0):
    mx.random.seed(117)
    model=nn.Linear(8,4)
    if master:model.set_dtype(mx.bfloat16)
    model.use_master_weights=master
    optimizer=optim.AdamW(1e-3,betas=[.9,.95],bias_correction=True)
    def loss(module,x):
        v=mx.mean((module(x).astype(mx.float32)*mx.random.uniform(shape=(3,4))-.2)**2)
        return v,v
    gradient=nn.value_and_grad(model,loss)
    def micro(x):
        (value,metric),grads=gradient(model,x)
        return value,metric,grads
    state=[model.state,mx.random.state]
    step=MLXTrainingStep(model,optimizer,mx.compile(micro,inputs=state,outputs=state),grad_clip=grad_clip,
        accumulation_step=build_mlx_accumulation_step(micro,state) if compiled else None)
    if pipelined:probe.install()
    else:probe.uninstall()
    losses=[]
    try:
        for lr in [1e-3,2e-3,0.,1e-4]:
            optimizer.learning_rate=lr
            batches=[mx.full((3,8),(i+1)/accumulated,dtype=mx.bfloat16 if master else mx.float32) for i in range(accumulated)]
            losses.append(step(batches,grad_accum=accumulated)[0])
        return snapshot([losses,model.parameters(),optimizer.state,getattr(optimizer,'master_params',{}),mx.random.uniform(shape=(16,))])
    finally:probe.uninstall()


def main():
    results=[]
    for master in (False,True):
        for accumulated in (2,8):
            for compiled in (False,True):
                expected=run(False,master,accumulated,compiled)
                actual=run(True,master,accumulated,compiled)
                assert len(expected)==len(actual)
                for (ka,da,a),(kb,db,b) in zip(expected,actual):
                    assert (ka,da)==(kb,db)
                    np.testing.assert_array_equal(a,b,err_msg=ka)
                results.append(dict(master=master,accumulation=accumulated,compiled_accumulation=compiled,exact=True))
    # A failed graph-preparation callback must drain already scheduled work and
    # restore the real mx module before the error escapes.
    evaluators=[]
    original_evaluator=pipeline.Evaluator
    class TrackedEvaluator(original_evaluator):
        def __init__(self):
            super().__init__()
            evaluators.append(self)
    pipeline.Evaluator=TrackedEvaluator
    calls=[0]
    def failing_micro(x):
        calls[0]+=1
        if calls[0]==2:raise RuntimeError("injected preparation failure")
        return mx.array(1.),mx.array(1.),{'x':x*2}
    try:
        try:
            pipeline.execute(None,None,failing_micro,iter([mx.ones((4,))]*2),2,1.,True)
        except RuntimeError as error:
            assert str(error)=="injected preparation failure"
        else:raise AssertionError("Expected the injected failure")
        assert evaluators and evaluators[0].pending is None
        assert pipeline.core.mx is mx
    finally:
        pipeline.Evaluator=original_evaluator
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
