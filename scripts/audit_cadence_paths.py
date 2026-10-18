"""CPU schedule replay and gradient-reachability audit; not a training-cause proof."""
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from telos.training.schedule import COROSredSchedule,DynamicMetricTracker
from telos.training.lr_schedule import WarmupCosineLR
from telos.diffusion.corosred_unified import RoutingMaskCache,corosred_unified_step_pytorch
from telos.models import TelosConfig,TelosTransformer

HISTORICAL='a2f09a0'

def historical(path):
    source=subprocess.check_output(['git','show',f'{HISTORICAL}:{path}'],cwd=ROOT,text=True)
    ns={'__name__':'cadence_historical_probe'}
    exec(compile(source,f'{HISTORICAL}:{path}','exec'),ns)
    return ns,hashlib.sha256(source.encode()).hexdigest()


def schedule_replay(schedule_class,tracker_class,*,cadence,corrected,steps=12716):
    schedule=schedule_class(max_steps=steps,alpha_min=.5,beta_max=.35)
    tracker=tracker_class(cadence=cadence) if corrected else tracker_class()
    trace=[]
    for step in range(1,steps+1):
        # Match the trainer: weights use the previous step's metric state.
        weights=schedule.get_weights(step-1,tracker.lrh_acc_ema,tracker.lrh_auc_ema)
        trace.append({**weights,'auc_ema':tracker.lrh_auc_ema})
        # Synthetic step response, deliberately crossing the gamma threshold.
        observation=.50 if step<1000 else .60
        if step==1 or step%cadence==0:
            if corrected: tracker.update_lrh(observation,observation,cadence=cadence)
            else: tracker.update_lrh(observation,observation)
    active=[i+1 for i,x in enumerate(trace) if x['gamma']>0]
    return trace,{'first_gamma_step':active[0] if active else None,'gamma_active_steps':len(active),'final_auc_ema':tracker.lrh_auc_ema}


def gradient_probe(step_fn=corosred_unified_step_pytorch,cache_class=RoutingMaskCache):
    torch.manual_seed(42)
    template=TelosTransformer(TelosConfig(vocab_size=32,d_model=16,n_layers=1,n_heads=2,n_kv_heads=2,max_seq_len=16,use_reliability_head=True,dropout=0.)).eval()
    tokens=torch.randint(4,32,(6,16))
    outcomes=[]
    for gamma,blend in [(0.,0.),(.1,.8)]:
        model=copy.deepcopy(template)
        cache=cache_class()
        torch.manual_seed(123)
        loss,metrics=step_fn(model,tokens,32,{'alpha':.85,'beta':.15,'gamma':gamma,'mask_blend':blend},routing_cache=cache,causal_ratio=.65,k_amb=5)
        loss.backward()
        raw={k:p.grad.clone() for k,p in model.named_parameters() if p.grad is not None}
        norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
        clipped={k:p.grad.clone() for k,p in model.named_parameters() if p.grad is not None}
        for k,p in model.named_parameters():
            if k in raw: p.grad=raw[k].clone()
        torch.nn.utils.clip_grad_norm_(model.parameters(),.25)
        forced_clipped={k:p.grad.clone() for k,p in model.named_parameters() if p.grad is not None}
        outcomes.append((raw,clipped,norm,forced_clipped))
    a,b=outcomes
    backbone=[k for k in a[0] if not k.startswith('reliability_head.')]
    head=[k for k in a[0] if k.startswith('reliability_head.')]
    return {'max_backbone_gradient_difference_before_clip':max(float((a[0][k]-b[0][k]).abs().max()) for k in backbone),
            'max_backbone_gradient_difference_after_clip':max(float((a[1][k]-b[1][k]).abs().max()) for k in backbone),
            'max_backbone_gradient_difference_with_forced_active_clip_point25':max(float((a[3][k]-b[3][k]).abs().max()) for k in backbone),
            'global_norm_gamma0':a[2],'global_norm_gamma_point1':b[2],
            'head_gradient_norm_gamma0':math.sqrt(sum(float(a[0][k].double().square().sum()) for k in head)),
            'head_gradient_norm_gamma_point1':math.sqrt(sum(float(b[0][k].double().square().sum()) for k in head))}


def main():
    torch.set_num_threads(1)
    old,oldhash=historical('telos/training/schedule.py')
    loss_old,losshash=historical('telos/diffusion/corosred_unified.py')
    report={'purpose':'Synthetic CPU replay and source reachability; does not identify cause of saved CUDA/TPU deficit',
            'historical_revision':subprocess.check_output(['git','rev-parse',HISTORICAL],text=True).strip(),
            'historical_source_sha256':{'schedule.py':oldhash,'corosred_unified.py':losshash},
            'current_source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['telos/training/schedule.py','telos/training/lr_schedule.py','telos/training/trainer_pytorch.py','telos/diffusion/corosred_unified.py','telos/models/transformer.py']},
            'metric_input':'Synthetic identical per-step .50 until step 999, .60 thereafter; not recovered training observations',
            'metric_replays':{},'lr_quantization':{}}
    traces={}
    for name,cls,tracker,cadence,corrected in [('historical_every1',old['COROSredSchedule'],old['DynamicMetricTracker'],1,False),('historical_every25',old['COROSredSchedule'],old['DynamicMetricTracker'],25,False),('current_every1',COROSredSchedule,DynamicMetricTracker,1,True),('current_every25',COROSredSchedule,DynamicMetricTracker,25,True)]:
        traces[name],report['metric_replays'][name]=schedule_replay(cls,tracker,cadence=cadence,corrected=corrected)
    reference=traces['current_every1']
    for name,trace in traces.items():
        report['metric_replays'][name]['max_difference_vs_current_every1']={key:max(abs(x[key]-y[key]) for x,y in zip(trace,reference)) for key in ['alpha','beta','gamma','mask_blend']}
    schedule=COROSredSchedule(max_steps=12716,alpha_min=.5,beta_max=.35)
    report['objective_step_quantization']={}
    for cadence in [10,25]:
        maxima={key:0. for key in ['alpha','beta']}
        for step in range(12716):
            fine=schedule.get_weights(step)
            coarse=schedule.get_weights((step//cadence)*cadence)
            for key in maxima: maxima[key]=max(maxima[key],abs(fine[key]-coarse[key]))
        report['objective_step_quantization'][str(cadence)]=maxima
    lrs={}
    for cadence in [1,10,25]:
        param=torch.nn.Parameter(torch.ones(1));opt=torch.optim.AdamW([param],lr=.0004242640687119285)
        sched=WarmupCosineLR(opt,254,12716,min_lr=4.2426406871192855e-05,update_cadence=cadence)
        values=[]
        for step in range(12716):
            values.append(opt.param_groups[0]['lr']);opt.step();sched.step()
        lrs[cadence]=values
        report['lr_quantization'][str(cadence)]={'integrated_lr':sum(values),'max_absolute_difference_vs_1':max(abs(a-b) for a,b in zip(values,lrs[1])),'integrated_relative_difference_vs_1':sum(values)/sum(lrs[1])-1}
    report['gradient_paths_current']=gradient_probe()
    report['gradient_paths_historical_loss']=gradient_probe(loss_old['corosred_unified_step_pytorch'],loss_old['RoutingMaskCache'])
    sched=COROSredSchedule(max_steps=12716,alpha_min=.5,beta_max=.35)
    report['routing_refresh_gate']={'without_metrics':sched.get_weights(4000)['mask_blend'],'with_auc_point8':sched.get_weights(4000,.8,.8)['mask_blend']}
    path=ROOT/'logs/cadence_path_audit_20261018.json'
    path.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));print(path)
if __name__=='__main__':main()
