"""Probe final accumulated backward + clipping + Adam in one compiled callable."""
import mlx.core as mx
from mlx.utils import tree_map
from telos.training.core import clip_grad_norm_mlx
from telos.training.mlx_step import MLXTrainingStep

original_call=MLXTrainingStep.__call__


def _build(step, count):
    model, optimizer = step.model, step.optimizer
    master=getattr(optimizer,'master_params',None)
    state=[model.state,optimizer.state,mx.random.state]
    if master is not None:state.append(master)
    def update(prior_grads,prior_loss,prior_metric,*batch):
        if step.accumulation_step is not None:
            grads,loss,metric=step.accumulation_step(prior_grads,prior_loss,prior_metric,*batch)
        else:
            loss,metric,current=step.microbatch_step(*batch)
            grads=tree_map(lambda a,b:a+b,prior_grads,current)
            loss,metric=prior_loss+loss,prior_metric+metric
        if step.grad_clip>0:
            grads,_=clip_grad_norm_mlx(grads,step.grad_clip,scale=float(count))
        else:
            grads=tree_map(lambda g:g/float(count),grads)
        if master is not None:
            master.update(optimizer.apply_gradients(grads,master))
            model.update(tree_map(lambda p:p.astype(mx.bfloat16),master))
        else:
            optimizer.update(model,grads)
        return loss,metric
    return mx.compile(update,inputs=state,outputs=state)


def call(step,batches,*,grad_accum=1,eval_every_microbatch=False,asynchronous=False):
    if (grad_accum<2 or not step._initialized or eval_every_microbatch
            or getattr(step,'stage_profile',None) is not None):
        return original_call(step,batches,grad_accum=grad_accum,
                             eval_every_microbatch=eval_every_microbatch,asynchronous=asynchronous)
    step.synchronize()
    batches=iter(batches)
    grads=None;loss=mx.array(0.,dtype=mx.float32);metric=mx.array(0.,dtype=mx.float32)
    for i in range(grad_accum-1):
        batch=next(batches);batch=batch if isinstance(batch,tuple) else (batch,)
        if i and step.accumulation_step is not None:
            grads,loss,metric=step.accumulation_step(grads,loss,metric,*batch)
        else:
            value,ce,current=step.microbatch_step(*batch)
            grads=current if grads is None else tree_map(lambda a,b:a+b,grads,current)
            loss,metric=loss+value,metric+ce
        mx.eval(grads,loss,metric)
    key=(grad_accum,step.grad_clip,id(getattr(step.optimizer,'master_params',None)))
    if getattr(step,'_probe_final_key',None)!=key:
        step._probe_final_step=_build(step,grad_accum)
        step._probe_final_key=key
    batch=next(batches);batch=batch if isinstance(batch,tuple) else (batch,)
    loss,metric=step._probe_final_step(grads,loss,metric,*batch)
    mx.eval(step.model.parameters(),step.optimizer.state,loss,metric,
            getattr(step.optimizer,'master_params',{}))
    return loss,metric


def install():MLXTrainingStep.__call__=call

def uninstall():MLXTrainingStep.__call__=original_call
