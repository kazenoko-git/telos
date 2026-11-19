"""Physical XLA training, compilation, timing and replay checks. CPU mode is harness QA only."""
import argparse
import copy
import json
import time
from pathlib import Path
import torch
from telos.training.lr_schedule import WarmupCosineLR
from telos.training.xla_optimizer import XLAAdamW
from telos.models.transformer import TelosTransformer

p = argparse.ArgumentParser()
p.add_argument('--device', choices=['cpu', 'xla'], default='xla')
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
xm = met = None
if a.device == 'xla':
    from telos.training.xla_utils import get_xla_device
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.runtime as xr
    device = get_xla_device()
    assert xr.device_type() == 'TPU', xr.device_type()
else:
    device = torch.device('cpu')


def fence():
    if xm:
        xm.mark_step()
        xm.wait_device_ops()


def compiles():
    data = met.metric_data('CompileTime') if met else None
    return data[0] if data else 0


def host(tree):
    if isinstance(tree, torch.Tensor): return tree.detach().cpu().clone()
    if isinstance(tree, dict): return {k: host(v) for k, v in tree.items()}
    if isinstance(tree, list): return [host(v) for v in tree]
    if isinstance(tree, tuple): return tuple(host(v) for v in tree)
    return copy.deepcopy(tree)


def compare(left, right, exact=False):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0 if exact else 2e-5,
                                   atol=0 if exact else 2e-6)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for k in left: compare(left[k], right[k], exact)
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for x, y in zip(left, right): compare(x, y, exact)
    else: assert left == right, (left, right)


steps, warmup = 32, 8
# CPU-generated inputs keep the data and RNG independent of optimizer execution.
g = torch.Generator().manual_seed(73)
batches = [torch.randint(4, 128, (4, 64), generator=g) for _ in range(steps)]

def run(dynamic, resume=None, stop=None, collect_gradients=False):
    torch.manual_seed(19)
    model = TelosTransformer(vocab_size=128, d_model=128, n_layers=2, n_heads=2,
                             max_seq_len=64, dropout=0., is_causal=True).to(device)
    cls = XLAAdamW if dynamic else torch.optim.AdamW
    opt = cls(model.parameters(), lr=.003, betas=(.9, .95), foreach=False)
    sched = WarmupCosineLR(opt, 8, steps, min_lr=.0001, update_cadence=1)
    start = 0
    if resume:
        model.load_state_dict(resume['model'])
        opt.load_state_dict(resume['optimizer'])
        sched.load_state_dict(resume['scheduler'])
        start = resume['step']
    fence()
    if met: met.clear_all()
    rows, losses, rates, gradients = [], [], [], []
    started = None
    for index in range(start, stop or steps):
        if index == warmup:
            fence()
            started = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        tokens = batches[index].to(device)
        logits = model(tokens, next_token_only=True)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 128), tokens[:, 1:].reshape(-1))
        loss.backward()
        rates.append(sched.get_last_lr()[0])
        # Optional snapshots do not materialize until outside timed blocks.
        if collect_gradients:
            gradients.append([v.grad.detach().clone() for v in model.parameters() if v.grad is not None])
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step(); sched.step()
        losses.append(loss.detach())
        fence()
        rows.append(dict(step=index + 1, compile_count=compiles()))
    elapsed = time.perf_counter() - started if started is not None else None
    memory = xm.get_memory_info(device) if xm else None
    saved = host(dict(model=model.state_dict(), optimizer=opt.state_dict(), scheduler=sched.state_dict(),
                      step=stop or steps))
    result = dict(dynamic=dynamic, elapsed_seconds=elapsed,
                  tokens_per_second=(steps - warmup) * 4 * 64 / elapsed if elapsed else None,
                  compilation=rows, learning_rates=rates,
                  losses=host(losses), memory_info=memory,
                  metrics=met.metrics_report() if met else None)
    return result, saved, host(gradients)

report = dict(device=str(device), torch_version=torch.__version__,
              physical_tpu=a.device == 'xla', precision='float32', dropout=0.,
              configuration=dict(vocab=128, d_model=128, layers=2, heads=2,
                                 batch=4, sequence=64, steps=steps, warmup=warmup), trials=[])
a.output.parent.mkdir(parents=True, exist_ok=True)
try:
    states = []
    for dynamic in [False, True, True, False]:
        result, state, _ = run(dynamic)
        report['trials'].append(result); states.append(state)
        print(json.dumps({k: v for k, v in result.items() if k not in ('losses', 'metrics')}, default=str), flush=True)
    compare(states[0], states[1])
    compare(states[1], states[2], exact=True)
    compare(states[0], states[3], exact=True)
    assert report['trials'][0]['learning_rates'] == report['trials'][1]['learning_rates']
    report['native_candidate_state_parity'] = True
    # Compare raw gradients with diagnostic snapshots outside benchmark trials.
    native, _, native_grads = run(False, stop=4, collect_gradients=True)
    dynamic, _, dynamic_grads = run(True, stop=4, collect_gradients=True)
    compare(native['losses'], dynamic['losses']); compare(native_grads, dynamic_grads)
    report['gradient_parity'] = True
    # Interrupted continuation must match the uninterrupted candidate exactly.
    _, checkpoint, _ = run(True, stop=16)
    checkpoint_path = a.output.parent / 'candidate_step16.pt'
    torch.save(checkpoint, checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    _, resumed, _ = run(True, resume=checkpoint)
    compare(states[1], resumed, exact=True)
    report['candidate_resume_exact'] = True
    for trial in report['trials']:
        trial['losses'] = [float(x) for x in trial['losses']]
        rows = trial['compilation']
        trial['compiles_after_step4'] = rows[-1]['compile_count'] - rows[3]['compile_count'] if met else None
    report['dynamic_compilation_stable'] = (all(t['compiles_after_step4'] == 0 for t in report['trials'] if t['dynamic']) if met else None)
    report['status'] = 'PASS' if report['dynamic_compilation_stable'] is not False else 'FAIL_COMPILATION'
except Exception as error:
    report['status'] = 'FAIL'
    report['error'] = repr(error)
    raise
finally:
    # Save tensor-bearing evidence separately from the readable report.
    torch.save(report, a.output.with_suffix('.pt'))
    def json_value(value):
        if isinstance(value, torch.Tensor): return value.tolist()
        return str(value)
    a.output.write_text(json.dumps(report, indent=2, default=json_value) + '\n')
    print('REPORT:', a.output, 'STATUS:', report.get('status'), flush=True)
