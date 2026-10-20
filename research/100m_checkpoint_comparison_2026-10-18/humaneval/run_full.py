"""Full164-task HumanEval comparison; native Telos generator/executor, common settings."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import gc
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import torch
from tokenizers import Tokenizer

ROOT = Path('/Users/ivansamuel/Documents/Projects/telos')
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from telos.eval import runner
from telos.eval.executor import execute_code_sandboxed, ExecutionResult

spec = importlib.util.spec_from_file_location('common_endpoint_evaluation', OUT.parent / 'evaluate.py')
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


def save(path, value):
    temp = path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    tokenizer_path = ROOT / 'configs/tokenizer_mac.json'
    tok = Tokenizer.from_file(str(tokenizer_path))
    assert tok.get_vocab_size() == 8192
    official_bytes = (OUT / 'HumanEval.jsonl.gz').read_bytes()
    official = [json.loads(line) for line in gzip.decompress(official_bytes).splitlines() if line.strip()]
    local_bytes = (ROOT / 'evals/benchmarks/humaneval_suite.json').read_bytes()
    local = json.loads(local_bytes)
    assert len(official) == len(local) == 164
    assert {t['task_id'] for t in official} == {f'HumanEval/{i}' for i in range(164)}
    by_id = {t['task_id']: t for t in official}
    import ast
    for task in local:
        ref = by_id[task['id']]
        assert task['prompt'] == ref['prompt'] and task['entry_point'] == ref['entry_point']
        assert ast.dump(ast.parse(task['test_harness'])) == ast.dump(ast.parse(ref['test'] + '\ncheck(' + ref['entry_point'] + ')'))
    first = official[0]
    harness = first['test'] + '\ncheck(' + first['entry_point'] + ')'
    good, detail = execute_code_sandboxed(first['prompt'] + first['canonical_solution'], harness, timeout_seconds=5.0)
    bad, _ = execute_code_sandboxed(first['prompt'] + '    return False\n', harness, timeout_seconds=5.0)
    if good != ExecutionResult.PASSED or bad != ExecutionResult.FAILED_ASSERTION:
        raise RuntimeError(f'Executor sanity check failed: canonical={good}, incorrect={bad}; {detail}')
    protocol = {
        'suite': 'HumanEval', 'tasks': 164, 'samples_per_task': 1, 'decoding': 'greedy',
        'device': device, 'dtype': 'float32', 'torch': str(torch.__version__), 'seed': 42,
        'max_new_tokens': 256, 'context_length': 512, 'timeout_seconds': 5.0,
        'context_policy': 'Native generator: total context<=512; completion ceiling reduced to512-prompt_tokens; prompts>=496 left-truncated. None of this164-task fixture exceeds496.',
        'stop_policy': 'Native generator: token0/PAD or3/EOS; new top-level def/class/__main__ boundary',
        'postprocessing': 'Native clean_functional_completion, including AST back-trimming; raw outputs preserved in generation traces',
        'tokenizer': str(tokenizer_path), 'tokenizer_sha256': common.digest_file(tokenizer_path),
        'official_dataset_sha256': hashlib.sha256(official_bytes).hexdigest(),
        'local_dataset_sha256': hashlib.sha256(local_bytes).hexdigest(),
        'official_local_match': 'All164 prompts/entrypoints identical; test-harness ASTs identical',
        'executor_sanity': {'canonical': good.value, 'deliberately_incorrect': bad.value},
        'source_sha256': {str(p.relative_to(ROOT)): common.digest_file(p) for p in [Path(__file__), ROOT/'telos/eval/runner.py', ROOT/'telos/eval/executor.py', ROOT/'telos/models/transformer.py', ROOT/'telos/models/components.py']},
    }
    prompt_to_id = {task['prompt']: task['id'] for task in local}
    summary = {'protocol': protocol, 'models': {}, 'errors': {}}
    save(OUT / 'summary.json', summary)
    print(f'[Protocol] {device} FP32; all164 tasks; greedy; max256 new tokens; correct8192 tokenizer; executor sanity passed.', flush=True)
    original_generate = runner._generate_greedy_completion
    for label in common.LABELS:
        checkpoint = ROOT/'checkpoints/corosred/unified'/label/'checkpoint_final.pt'
        report_path = OUT / (label + '.json')
        trace_path = OUT / (label + '.generations.jsonl')
        if report_path.exists():
            prior = json.loads(report_path.read_text())
            if prior.get('protocol') == protocol and prior.get('report', {}).get('total_tasks') == 164:
                summary['models'][label] = prior
                print(f'[Already complete] {label}', flush=True)
                continue
        started = time.perf_counter()
        print(f'[Starting] {label}', flush=True)
        model = None
        try:
            model, metadata = common.model_metadata(checkpoint)
            model = model.to(device).eval()
            with trace_path.open('w', encoding='utf-8') as trace:
                def traced(*args, **kwargs):
                    prompt = kwargs['prompt']
                    generation_started = time.perf_counter()
                    text = original_generate(*args, **kwargs)
                    n_prompt = len(tok.encode(prompt).ids)
                    record = {
                        'id': prompt_to_id[prompt], 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                        'prompt_tokens': n_prompt, 'raw_completion': text,
                        'clean_completion': runner.clean_functional_completion(prompt, text),
                        'requested_max_new_tokens': kwargs.get('max_new_tokens'),
                        'context_capped_max_new_tokens': min(kwargs.get('max_new_tokens',256),512-n_prompt),
                        'generation_seconds': time.perf_counter() - generation_started,
                    }
                    trace.write(json.dumps(record) + '\n')
                    trace.flush()
                    return text
                runner._generate_greedy_completion = traced
                with torch.inference_mode():
                    report = runner.evaluate_functional(
                        model=model, tokenizer=tok, backend='pytorch', suite='humaneval',
                        prompt_mode='base', max_tasks=None, timeout_seconds=5.0, max_new_tokens=256,
                    )
            runner._generate_greedy_completion = original_generate
            assert report['total_tasks'] == 164
            result = {'protocol': protocol, 'metadata': metadata, 'report': report, 'seconds': time.perf_counter() - started, 'trace': str(trace_path)}
            save(report_path, result)
            summary['models'][label] = result
            print(f'[Done] {label}: {report["execution_outcomes"].get("PASSED",0)}/164 passed; {result["seconds"]:.1f}s', flush=True)
        except Exception as exc:
            runner._generate_greedy_completion = original_generate
            summary['errors'][label] = {'type': type(exc).__name__, 'message': str(exc)}
            print(f'[Error] {label}: {type(exc).__name__}: {exc}', flush=True)
        finally:
            if model is not None:
                del model
            gc.collect()
            if device == 'mps':
                torch.mps.empty_cache()
            save(OUT / 'summary.json', summary)
    print(f'Results: {OUT / "summary.json"}', flush=True)


if __name__ == '__main__':
    main()
