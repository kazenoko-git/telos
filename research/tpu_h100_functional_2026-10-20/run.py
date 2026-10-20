"""Matched full functional suites and contextual probes, with resumable raw traces."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'research/100m_checkpoint_comparison_2026-10-18'
sys.path.insert(0, str(ROOT))
from telos.eval import runner
from telos.eval.probes import load_contextual_probes

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

common = module('common_evaluation', BASE / 'evaluate.py')
strict = module('strict_scoring', BASE / 'humaneval/score_strict.py')
LABELS = ['100m_r50_tpu_v5e8_seed42_cad1', '100m_r50_modal_h100_seed42_cad1']

def save(path, value):
    temp = path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)

def update_report():
    lines = ['# TPU versus H100: functional evaluation', '',
             'Greedy FP32 generation, correct 8192-token tokenizer, maximum 256 new tokens in a 512-token context. Strict scores use function-boundary stops without source repair. Telos-cleaned scores are secondary.', '',
             '| Model | Suite | Strict passed | Strict pass@1 | Telos-cleaned passed |',
             '|---|---|---:|---:|---:|']
    for path in sorted(OUT.glob('*.functional.json')):
        r = json.loads(path.read_text())
        s = r['strict']
        lines.append(f"| {r['label']} | {r['suite']} | {s['passed']}/{s['tasks']} | {s['pass_at_1_pct']:.2f}% | {r['native']['execution_outcomes'].get('PASSED', 0)}/{s['tasks']} |")
    old = BASE / 'humaneval/strict_summary.json'
    if old.exists():
        s = json.loads(old.read_text())['models'][LABELS[1]]
        lines.append(f"| {LABELS[1]} | humaneval (completed prior run) | {s['passed']}/{s['tasks']} | {s['pass_at_1_pct']:.2f}% | 4/164 |")
    lines += ['', '| Model | Probes | Top-1 | Top-5 | Mean CE |', '|---|---|---:|---:|---:|']
    for path in sorted(OUT.glob('*.probes.json')):
        r = json.loads(path.read_text())
        for mode, report in r['report'].items():
            s = report['overall']
            lines.append(f"| {r['label']} | {mode}, {s['total_probes']} | {s['top1_acc_pct']:.2f}% | {s['top5_acc_pct']:.2f}% | {s['mean_ce']:.2f} |")
    lines += ['', 'Private_unseen is the local 512-task generated suite, evaluated separately with base specifications and algorithmic hints. Its name does not establish disjointness from training data. Probes measure first-target-token prediction, not full solution correctness. Infill probes include the supplied suffix; causal probes see only the prefix.', '']
    (OUT / 'comparison.md').write_text('\n'.join(lines))

def functional(model, tok, metadata, label, suite):
    result_path = OUT / f'{label}.{suite}.functional.json'
    if result_path.exists():
        print('[Already complete]', label, suite, flush=True)
        return
    tasks = json.loads((OUT / 'datasets' / (suite + '.json')).read_text())
    by_prompt = {t['prompt']: t for t in tasks}
    assert len(by_prompt) == len(tasks), 'Prompt collision prevents trace identification'
    trace_path = OUT / f'{label}.{suite}.generations.jsonl'
    cached = {}
    if trace_path.exists():
        for line in trace_path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            cached[row['id']] = row
        # Remove a possible partial final line before appending resumed results.
        trace_path.write_text(''.join(json.dumps(row) + '\n' for row in cached.values()))
    original = runner._generate_greedy_completion
    started = time.perf_counter()
    seen = []
    with trace_path.open('a') as trace:
        def traced(*args, **kwargs):
            task = by_prompt[kwargs['prompt']]
            if task['id'] in cached:
                row = cached[task['id']]
            else:
                t0 = time.perf_counter()
                raw = original(*args, **kwargs)
                completion = raw
                for stop in strict.STOPS:
                    completion = completion.split(stop, 1)[0]
                scored = strict.execute(task['prompt'] + completion + '\n' + task['test_harness'] + '\n')
                n = len(tok.encode(task['prompt']).ids)
                row = {'id': task['id'], 'raw_completion': raw,
                       'clean_completion': runner.clean_functional_completion(task['prompt'], raw),
                       'strict': scored, 'prompt_sha256': hashlib.sha256(task['prompt'].encode()).hexdigest(),
                       'prompt_tokens': n, 'max_new_tokens': kwargs['max_new_tokens'],
                       'generation_and_execution_seconds': time.perf_counter() - t0}
                trace.write(json.dumps(row) + '\n'); trace.flush()
                cached[task['id']] = row
            seen.append(row)
            save(OUT / 'progress.json', {'label': label, 'suite': suite, 'completed': len(seen), 'total': len(tasks),
                 'strict_passed': sum(r['strict']['outcome'] == 'PASSED' for r in seen)})
            return row['raw_completion']
        runner._generate_greedy_completion = traced
        try:
            with torch.inference_mode():
                native = runner.evaluate_functional(model=model, tokenizer=tok, backend='pytorch', suite=suite,
                    prompt_mode='hint' if suite.endswith('hint') else 'base', max_tasks=None,
                    timeout_seconds=5.0, max_new_tokens=256)
        finally:
            runner._generate_greedy_completion = original
    assert len(seen) == native['total_tasks'] == len(tasks)
    passed = sum(r['strict']['outcome'] == 'PASSED' for r in seen)
    strict_result = {'tasks': len(seen), 'passed': passed, 'pass_at_1_pct': 100 * passed / len(seen),
                     'passed_ids': [r['id'] for r in seen if r['strict']['outcome'] == 'PASSED']}
    save(result_path, {'label': label, 'suite': suite, 'metadata': metadata,
                      'strict': strict_result, 'native': native, 'seconds': time.perf_counter() - started})
    print('[Completed]', label, suite, strict_result, flush=True)
    update_report()

def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    tok = Tokenizer.from_file(str(ROOT / 'configs/tokenizer_mac.json'))
    assert tok.get_vocab_size() == 8192
    data_dir = OUT / 'datasets'
    data_dir.mkdir(exist_ok=True)
    sources = {
        'humaneval.json': ROOT / 'evals/benchmarks/humaneval_suite.json',
        'private_unseen_base.json': ROOT / 'evals/benchmarks/private_unseen_base.json',
        'private_unseen_hint.json': ROOT / 'evals/benchmarks/private_unseen_hint.json',
        'contextual_probes_1000.json': ROOT / 'evals/benchmarks/contextual_probes_1000.json',
    }
    for name, source in sources.items():
        dest = data_dir / name
        if dest.exists():
            assert common.digest_file(dest) == common.digest_file(source), 'Suite changed since launch'
        else:
            shutil.copy2(source, dest)
    manifest = {'device': device, 'dtype': 'float32', 'torch': str(torch.__version__), 'seed': 42,
                'max_new_tokens': 256, 'context_length': 512, 'timeout_seconds': 5,
                'tokenizer_sha256': common.digest_file(ROOT / 'configs/tokenizer_mac.json'),
                'datasets_sha256': {name: common.digest_file(data_dir / name) for name in sources},
                'sources_sha256': {str(p.relative_to(ROOT)): common.digest_file(p) for p in
                    [Path(__file__), ROOT / 'telos/eval/runner.py', ROOT / 'telos/models/transformer.py',
                     ROOT / 'telos/models/components.py', strict.HELPER]},
                'strict_processing': 'Native function-boundary stops only; no AST repair',
                'native_processing': 'Telos cleanup and AST back-trimming, secondary score',
                'h100_humaneval_reused': str(BASE / 'humaneval/strict_summary.json')}
    save(OUT / 'protocol.json', manifest)
    ref = strict.TASKS['HumanEval/0']
    harness = '\n' + ref['test'] + '\ncheck(' + ref['entry_point'] + ')'
    assert strict.execute(ref['prompt'] + ref['canonical_solution'] + harness)['outcome'] == 'PASSED'
    assert strict.execute(ref['prompt'] + '    return False\n' + harness)['outcome'] == 'FAILED_ASSERTION'
    print('[Protocol]', device, 'FP32; executor controls passed.', flush=True)
    # Fast probes first, then full generation suites; one GPU model at a time.
    for label in LABELS:
        path = OUT / f'{label}.probes.json'
        if path.exists():
            continue
        model, metadata = common.model_metadata(ROOT / 'checkpoints/corosred/unified' / label / 'checkpoint_final.pt')
        model = model.to(device).eval()
        with torch.inference_mode():
            probes = runner.evaluate_probes(model, tok, 'pytorch', paradigm='corosred', num_probes=1000, probe_type='both')
        save(path, {'label': label, 'metadata': metadata, 'report': probes})
        update_report()
        del model; gc.collect()
        if device == 'mps': torch.mps.empty_cache()
    for suite in ['humaneval', 'private_unseen_base', 'private_unseen_hint']:
        for label in LABELS:
            if suite == 'humaneval' and label == LABELS[1]:
                continue  # Already completed full164-task strict and native run.
            if (OUT / f'{label}.{suite}.functional.json').exists():
                continue
            torch.manual_seed(42)
            model, metadata = common.model_metadata(ROOT / 'checkpoints/corosred/unified' / label / 'checkpoint_final.pt')
            model = model.to(device).eval()
            functional(model, tok, metadata, label, suite)
            del model; gc.collect()
            if device == 'mps': torch.mps.empty_cache()
    save(OUT / 'status.json', {'status': 'complete'})
    update_report()

if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        save(OUT / 'status.json', {'status': 'error', 'type': type(exc).__name__, 'message': str(exc)})
        raise
