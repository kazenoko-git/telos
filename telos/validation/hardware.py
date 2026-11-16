"""Fresh-process single-device release checks, runnable on CPU, CUDA or TPU.

python -m telos.validation.hardware --hardware cuda --output /tmp/telos-v172
TPU users must supply truthful TPU_ACCELERATOR_TYPE metadata (e.g. v5e-1).
All child stdout/stderr, commands, source hashes and results are preserved.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def provenance():
    import platform
    import torch
    import telos
    package = Path(telos.__file__).resolve().parent
    return {
        'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
        'argv': sys.argv,
        'torch': torch.__version__, 'torch_path': torch.__file__, 'telos_path': str(package),
        'source_sha256': {str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(package.rglob('*.py'))},
        'topology_environment': {k: os.environ[k] for k in (
            'PJRT_DEVICE', 'TPU_ACCELERATOR_TYPE', 'TPU_CHIPS_PER_HOST_BOUNDS',
            'TPU_CHIPS_PER_PROCESS_BOUNDS', 'TPU_PROCESS_BOUNDS', 'TPU_NUM_DEVICES',
            'PJRT_LOCAL_PROCESS_COUNT', 'WORLD_SIZE') if k in os.environ},
    }


def require_device(hardware):
    import torch
    if hardware == 'cuda':
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('This runner requires exactly one real CUDA device')
        from telos.training.hardware import detect_cuda_profile
        return vars(detect_cuda_profile())
    if hardware == 'xla':
        from telos.training.xla_utils import clean_tpu_environment
        clean_tpu_environment()  # Before importing/initializing PJRT.
        import torch_xla
        import torch_xla.runtime as xr
        if xr.device_type() != 'TPU' or xr.global_runtime_device_count() != 1:
            raise RuntimeError('This runner requires exactly one real TPU device')
        return {'torch_xla': torch_xla.__version__, 'device': xr.device_type(),
                'devices': xr.global_runtime_device_count()}
    return {'device': 'cpu', 'devices': 1}


def benchmark(args):
    """Use the production update loop and its completed-update timer."""
    import numpy as np
    import torch
    from telos.models import TelosTransformer
    from telos.training import set_global_seed
    from telos.training.trainer_pytorch import UnifiedPyTorchTrainer
    from telos.training.hardware import detect_cuda_profile
    set_global_seed(42)
    precision = detect_cuda_profile().precision if args.hardware == 'cuda' else ('bf16' if args.hardware == 'xla' else 'float32')
    cfg = {'paradigm': 'ar', 'model': {'vocab_size': 256, 'seq_len': args.seq_len},
           'training': {'max_steps': args.steps, 'batch_size': 2, 'gradient_accumulation': 1,
                        'warmup_steps': 1, 'max_lr': .001, 'min_lr': .0001, 'seed': 42,
                        'precision': precision, 'compile': args.compiled, 'compile_mode': 'default'},
           'data': {'synthetic': True},
           'checkpoint': {'checkpoint_dir': str(args.output/'train'), 'save_every_steps': 0}}
    model = TelosTransformer(vocab_size=256, d_model=128, n_layers=2,
                             n_heads=2, n_kv_heads=2, is_causal=True)
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    trainer = UnifiedPyTorchTrainer('ar', model, cfg, device_type=args.hardware)
    report = {}
    def capture(steps, elapsed, latencies, bs, grad_accum):
        if not steps or elapsed <= 0 or not all(np.isfinite(latencies)):
            raise AssertionError('No finite measured updates')
        report.update(measured_updates=steps, elapsed_seconds=elapsed,
                      input_tokens=steps*bs*grad_accum*args.seq_len,
                      tokens_per_second=steps*bs*grad_accum*args.seq_len/elapsed,
                      latency_ms=latencies, warmup_updates=5)
    trainer._print_benchmark_report = capture
    losses = []
    execute = trainer._execute_microbatch
    def track_loss(*values, **options):
        loss, metrics = execute(*values, **options)
        losses.append(loss.detach())
        return loss, metrics
    trainer._execute_microbatch = track_loss
    if args.hardware == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    trainer.train(benchmark=True, benchmark_duration=300)
    if not torch.isfinite(torch.stack(losses)).all().item():
        raise AssertionError('Nonfinite benchmark loss')
    raw = getattr(trainer.model, '_orig_mod', trainer.model)
    final = {k: v.detach().cpu() for k, v in raw.state_dict().items()}
    if not all(torch.isfinite(v).all() for v in final.values()):
        raise AssertionError('Nonfinite final parameters')
    delta = max((final[k]-v.cpu()).abs().max().item() for k, v in initial.items())
    if delta == 0 or trainer.global_step != args.steps or trainer.scheduler.last_epoch != args.steps:
        raise AssertionError('Benchmark did not advance the schedule and update weights')
    report.update(global_step=trainer.global_step, scheduler_step=trainer.scheduler.last_epoch,
                  max_parameter_change=delta, compiled=args.compiled, sequence_length=args.seq_len,
                  train_wall_seconds_including_warmup=time.perf_counter()-started)
    if args.hardware == 'cuda':
        report.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
    return report


def prepare(root):
    import torch
    from telos.models import TelosTransformer
    from telos.data.tokenizer import load_tokenizer
    torch.manual_seed(17)
    model = TelosTransformer(vocab_size=len(load_tokenizer().get_vocab()), d_model=64,
                             n_layers=1, n_heads=2, n_kv_heads=2, max_seq_len=32, is_causal=True)
    torch.save({'model_state_dict': model.state_dict(), 'config': {
        'paradigm': 'ar', 'model': vars(model.config)}}, root/'base.pt')
    rows = [{'prompt': p, 'response': a, 'chosen': a, 'rejected': b} for p, a, b in (
        ('What is two plus two?', 'Four.', 'Five.'), ('What color is grass?', 'Green.', 'Purple.'),
        ('Complete: the sun rises in the', 'east.', 'west.'))]
    write_json(root/'fixtures.json', rows)
    (root/'data.jsonl').write_text('\n'.join(json.dumps(row) for row in rows))


def training(args):
    root, out = args.output, args.output/'train'
    if args.case.startswith('pretrain-'):
        from telos.train.cli import train
        paradigm = args.case.removeprefix('pretrain-')
        train(paradigm=paradigm, hardware=args.hardware, devices=1,
              d_model=128, n_layers=1, vocab_size=256, seq_len=16,
              max_steps=args.steps, batch_size=2, grad_accum=1, seed=42,
              max_lr=.003, min_lr=.0001, warmup_steps=1, synthetic=True,
              checkpoint_dir=str(out), save_every=1,
              resume_checkpoint=out/'checkpoint_step_1.pt' if args.restart else None)
    else:
        from telos.posttrain.trainer import posttrain
        method, finetune = args.case.split('-')
        posttrain(root/'base.pt', root/'data.jsonl', out, method=method,
                  finetune=finetune, hardware=args.hardware, lora_rank=4, lora_dropout=.1,
                  max_steps=args.steps, batch_size=1, max_length=24, save_every=1,
                  learning_rate=.002, seed=42,
                  resume=out/'checkpoint_step_1.resume.pt' if args.restart else None)
    return {'completed': True}


def adapter_invariants(root, case):
    """Compare every frozen parameter and quantization buffer, plus changed factors."""
    import torch
    from telos.eval.runner import load_model_from_checkpoint
    from telos.posttrain.adapters import apply_lora
    from telos.validation.state import assert_state_equal
    state = torch.load(root/'train/checkpoint_final.resume.pt', weights_only=True, map_location='cpu')
    model, _, _ = load_model_from_checkpoint(root/'base.pt')
    mode = case.split('-')[1]
    if mode == 'full':
        initial = dict(model.state_dict())
        assert_state_equal(set_keys(initial), set_keys(state['model']))
        if state['reference'] is not None:
            assert_state_equal(initial, state['reference'], 'frozen DPO reference')
        if not any(not torch.equal(initial[k], v) for k, v in state['model'].items()):
            raise AssertionError('Full fine tuning did not change parameters')
        return {'status': 'PASS', 'updated': True}
    torch.manual_seed(42)
    apply_lora(model, rank=4, alpha=32., dropout=.1, quantized=mode == 'qlora')
    initial = dict(model.state_dict())
    assert_state_equal(set_keys(initial), set_keys(state['model']))
    if state['reference'] is not None:
        assert_state_equal(initial, state['reference'], 'frozen DPO reference')
    frozen = 0
    for key, value in initial.items():
        if not key.endswith(('lora_a', 'lora_b')):
            assert_state_equal(value, state['model'][key], key)
            frozen += 1
    factors = [v for k, v in state['model'].items() if k.endswith('lora_b')]
    if not factors or not any(torch.isfinite(v).all() and v.count_nonzero() for v in factors):
        raise AssertionError('No finite nonzero adapter update')
    return {'status': 'PASS', 'frozen_tensor_leaves': frozen, 'nonzero_adapter_update': True}


def set_keys(state):
    return sorted(state.keys())


def nf4_memory(args):
    """Report retained dense weights separately from actual CUDA allocator peaks."""
    import torch
    from torch.nn import functional as F
    from telos.posttrain.adapters import NF4Linear
    device = 'cuda' if args.hardware == 'cuda' else 'cpu'
    torch.manual_seed(7)
    layer = NF4Linear(torch.nn.Linear(512, 512)).to(device)
    source = torch.randn(2, 16, 512, device=device)
    results, grads, losses = [], [], []
    for recompute in (False, True):
        x = source.clone().requires_grad_()
        retained = []
        if device == 'cuda':
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        with torch.autograd.graph.saved_tensors_hooks(
                lambda t: retained.append((tuple(t.shape), t.numel()*t.element_size())) or t, lambda t: t):
            y = layer(x) if recompute else F.linear(x, layer.dequantize(), layer.bias)
            loss = y.square().mean()
        loss.backward()
        losses.append(loss.detach().cpu()); grads.append(x.grad.cpu())
        record = {'recompute': recompute, 'dense_weight_retained_bytes': sum(n for shape, n in retained if shape == (512, 512))}
        if device == 'cuda':
            torch.cuda.synchronize()
            record.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
        results.append(record)
        del y, loss, x
    from telos.validation.state import assert_state_equal
    assert_state_equal(losses[0], losses[1]); assert_state_equal(grads[0], grads[1])
    if not results[0]['dense_weight_retained_bytes'] or results[1]['dense_weight_retained_bytes']:
        raise AssertionError('Dense weight retention was not eliminated')
    return {'device': device, 'results': results, 'scope': 'single projection; not whole-model peak memory'}


def child(args):
    started = time.perf_counter()
    original = provenance()
    # Memory hooks are a separate CPU/CUDA diagnostic; TPU peak allocation is untested.
    # A launcher parent must never initialize PJRT before torch_xla.launch.
    if args.task == 'nf4':
        device = {'requested': args.hardware, 'scope': 'CPU probe for XLA; CUDA allocator probe for CUDA'}
    elif args.task == 'train' and args.hardware == 'xla':
        device = {'requested': 'xla', 'initialization': 'deferred to training worker'}
    else:
        device = require_device(args.hardware)
    if args.task == 'device':
        result = {'verified': True}
    elif args.task == 'benchmark':
        result = benchmark(args)
    elif args.task == 'nf4':
        result = nf4_memory(args)
    else:
        result = training(args)
    write_json(args.output/'worker.json', {'provenance': original, 'hardware': device, 'result': result,
                                               'worker_wall_seconds': time.perf_counter()-started})


def invoke(args, root, task, case='', compiled=False, restart=False, interrupt=False):
    root.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-m', 'telos.validation.hardware', '--worker', '--hardware', args.hardware,
               '--output', str(root), '--task', task, '--steps', str(args.steps),
               '--seq-len', str(args.seq_len), '--case', case]
    if compiled:
        command.append('--compiled')
    if restart:
        command.append('--restart')
    label = 'restart' if restart else ('interrupt' if interrupt else 'control')
    write_json(root/f'{label}-command.json', command)
    checkpoint = root/'train'/('checkpoint_step_1.pt' if case.startswith('pretrain-') else 'checkpoint_step_1.resume.pt')
    with (root/f'{label}.stdout').open('w') as stdout, (root/f'{label}.stderr').open('w') as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True)
        deadline = time.monotonic()+1800
        try:
            if interrupt:
                while not checkpoint.exists() and process.poll() is None:
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Checkpoint not committed within 30 minutes')
                    time.sleep(.005)
                if not checkpoint.exists() or process.poll() is not None:
                    raise AssertionError('Process finished before interruption; no interruption proof')
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=30)
                if list((root/'train').glob('checkpoint_final*')):
                    raise AssertionError('Process completed before interruption; increase steps')
                return {'interrupted_after_atomic_checkpoint': str(checkpoint), 'exit_code': process.returncode}
            code = process.wait(timeout=1800)
            if code:
                raise RuntimeError(f'{case or task} child exited {code}; see {root}/{label}.stderr')
            return json.loads((root/'worker.json').read_text())
        finally:
            # Also reap launcher workers on an interrupted or failed run.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hardware', choices=['cpu', 'cuda', 'xla'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=12)
    parser.add_argument('--seq-len', type=int, default=64)
    parser.add_argument('--compile-abba', action='store_true', help='Matched CUDA compile off/on/on/off in fresh processes')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--task', default='train', choices=['train', 'benchmark', 'nf4', 'device'])
    parser.add_argument('--case', default='pretrain-ar')
    parser.add_argument('--compiled', action='store_true')
    parser.add_argument('--restart', action='store_true')
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.hardware == 'xla':
        os.environ.setdefault('PJRT_DEVICE', 'TPU')
    if args.steps < 8:
        parser.error('--steps must be >=8 to measure after five warmup updates')
    if args.worker:
        child(args)
        return
    if args.output.exists():
        raise FileExistsError('Choose a new output directory to preserve prior evidence')
    args.output.mkdir(parents=True)
    results = {'status': 'RUNNING', 'cases': {}, 'untested': [
        'multi-device collectives and global token normalization', 'long-run convergence',
        'representative corpus throughput', 'full-model QLoRA peak memory']}
    try:
        write_json(args.output/'parent-provenance.json', provenance())
        from importlib.metadata import distributions
        write_json(args.output/'distributions.json', {d.metadata['Name']: d.version for d in distributions()})
        freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True)
        (args.output/'pip-freeze.txt').write_text(freeze.stdout+freeze.stderr)
        results['device'] = invoke(args, args.output/'device', 'device')
        from telos.validation.state import compare_checkpoints
        for case in ['pretrain-'+p for p in ('ar', 'mdlm', 'undlm', 'corosred')] + [
                f'{method}-{mode}' for method in ('sft', 'dpo') for mode in ('full', 'lora', 'qlora')]:
            root = args.output/case
            root.mkdir()
            prepare(root)
            write_json(root/'fixture-sha256.json', {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (root/'base.pt', root/'data.jsonl', root/'fixtures.json')})
            invoke(args, root, 'train', case)
            suffix = '.pt' if case.startswith('pretrain-') else '.resume.pt'
            final = root/'train'/('checkpoint_final'+suffix)
            control = root/'control.pt'
            shutil.copyfile(final, control)
            if not case.startswith('pretrain-'):
                results['cases'][case+'-invariants'] = adapter_invariants(root, case)
            shutil.rmtree(root/'train')  # Control state is preserved; all paths stay identical.
            interruption = invoke(args, root, 'train', case, interrupt=True)
            invoke(args, root, 'train', case, restart=True)
            comparison = compare_checkpoints(control, final)
            results['cases'][case] = {'status': 'PASS', **interruption, **comparison}
            write_json(args.output/'results.json', results)
        results['benchmark'] = invoke(args, args.output/'benchmark', 'benchmark')
        if args.compile_abba:
            if args.hardware != 'cuda':
                raise ValueError('--compile-abba requires CUDA')
            results['compile_abba'] = [invoke(args, args.output/f'compile-{i}', 'benchmark', compiled=value)
                                       for i, value in enumerate((False, True, True, False))]
            aggregate = {}
            for compiled, label in ((False, 'eager'), (True, 'compiled')):
                runs = [r for r in results['compile_abba'] if r['result']['compiled'] == compiled]
                tokens = sum(r['result']['input_tokens'] for r in runs)
                elapsed = sum(r['result']['elapsed_seconds'] for r in runs)
                aggregate[label] = {'input_tokens': tokens, 'elapsed_seconds': elapsed,
                                    'tokens_per_second': tokens/elapsed,
                                    'worker_wall_seconds': sum(r['worker_wall_seconds'] for r in runs)}
            aggregate['warmed_throughput_ratio'] = aggregate['compiled']['tokens_per_second']/aggregate['eager']['tokens_per_second']
            aggregate['end_to_end_time_ratio'] = aggregate['eager']['worker_wall_seconds']/aggregate['compiled']['worker_wall_seconds']
            results['compile_abba_aggregate'] = aggregate
        results['nf4_memory'] = invoke(args, args.output/'nf4', 'nf4')
        results['status'] = 'PASS_FOR_LISTED_CHECKS_ONLY'
    except BaseException as exc:
        results.update(status='FAIL', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write_json(args.output/'results.json', results)


if __name__ == '__main__':
    main()
