"""Fresh deterministic HumanEval reruns, followed by resuming private suites."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import traceback

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('functional_comparison', HERE / 'run.py')
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)

def main():
    import gc
    import shutil
    import torch
    from tokenizers import Tokenizer
    out = HERE / 'humaneval_repeat1'
    out.mkdir(exist_ok=True)
    (out / 'datasets').mkdir(exist_ok=True)
    shutil.copy2(HERE / 'datasets/humaneval.json', out / 'datasets/humaneval.json')
    evaluation.OUT = out
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    tok_path = evaluation.ROOT / 'configs/tokenizer_mac.json'
    tok = Tokenizer.from_file(str(tok_path))
    assert tok.get_vocab_size() == 8192
    protocol = json.loads((HERE / 'protocol.json').read_text())
    assert protocol['device'] == device
    for source, sha in protocol['sources_sha256'].items():
        assert evaluation.common.digest_file(evaluation.ROOT / source) == sha, source
    assert evaluation.common.digest_file(tok_path) == protocol['tokenizer_sha256']
    evaluation.save(out / 'protocol.json', {**protocol, 'purpose': 'Independent fresh greedy repeat for both checkpoints; raw completion comparison against first runs'})
    comparisons = {}
    for label in evaluation.LABELS:
        torch.manual_seed(42)
        model, metadata = evaluation.common.model_metadata(evaluation.ROOT / 'checkpoints/corosred/unified' / label / 'checkpoint_final.pt')
        model = model.to(device).eval()
        evaluation.functional(model, tok, metadata, label, 'humaneval')
        del model
        gc.collect()
        if device == 'mps': torch.mps.empty_cache()
        first = (HERE / f'{label}.humaneval.generations.jsonl' if 'tpu' in label else
                 evaluation.BASE / 'humaneval' / f'{label}.generations.jsonl')
        prior = {r['id']: r for r in map(json.loads, first.read_text().splitlines())}
        repeated = {r['id']: r for r in map(json.loads, (out / f'{label}.humaneval.generations.jsonl').read_text().splitlines())}
        assert set(prior) == set(repeated) and len(prior) == 164
        changed = [key for key in prior if prior[key]['raw_completion'] != repeated[key]['raw_completion']]
        result = json.loads((out / f'{label}.humaneval.functional.json').read_text())
        comparisons[label] = {'tasks': 164, 'identical_raw_completions': 164 - len(changed),
                             'changed_task_ids': changed, 'repeat_strict': result['strict']}
        evaluation.save(out / 'repeatability.json', comparisons)
        print('[Repeatability]', label, comparisons[label], flush=True)
    evaluation.save(out / 'status.json', {'status': 'complete'})

if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        # The previous private-suite process ended; its saved raw traces allow resume.
        print('[Continuing] Original private-suite pipeline, preserving cached generations.', flush=True)
        subprocess.run([sys.executable, '-u', str(HERE / 'run.py')], cwd=evaluation.ROOT, check=True)
