"""Full contextual-probe anti-cheat audit plus saved generation diagnostics."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import collections
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import torch
from tokenizers import Tokenizer
from telos.eval.anticheat import evaluate_span_infill_probe, summarize_anticheat_suite

spec = importlib.util.spec_from_file_location('common', OUT.parent / '100m_checkpoint_comparison_2026-10-18/evaluate.py')
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)
LABELS = ['100m_r50_tpu_v5e8_seed42_cad1', '100m_r50_modal_h100_seed42_cad1']

def save(path, value):
    temp = path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)

def diagnostics():
    reports = {}
    for label in LABELS:
        reports[label] = {}
        for suite in ['humaneval', 'private_unseen_base', 'private_unseen_hint']:
            if suite == 'humaneval':
                source = OUT / 'humaneval_repeat1' / f'{label}.humaneval.functional.json'
            else:
                source = OUT / f'{label}.{suite}.functional.json'
            r = json.loads(source.read_text())
            native = r['native']
            reports[label][suite] = {
                'source': str(source), 'source_sha256': common.digest_file(source),
                'tasks': native['total_tasks'], 'strict': r['strict'],
                'native_cleaned_ast_validity_pct': native['ast_validity_pct'],
                'native_execution_outcomes': native['execution_outcomes'],
                'repetition': native['repetition'], 'numerical_accuracy': native['numerical_accuracy'],
                'extended_metrics': native['extended_metrics'],
            }
    save(OUT / 'generation_diagnostics.json', {
        'processing': 'Existing Telos functional-report diagnostics; native completion cleanup applies. No additional generation.',
        'models': reports})

def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    diagnostics()
    tok = Tokenizer.from_file(str(ROOT / 'configs/tokenizer_mac.json'))
    assert tok.get_vocab_size() == 8192
    dataset = OUT / 'datasets/contextual_probes_1000.json'
    probes = json.loads(dataset.read_text())
    assert len(probes) == 1000
    manifest = {'device': 'cpu', 'dtype': 'float32', 'torch': str(torch.__version__),
                'probes': len(probes), 'requested_spans': [1, 2, 4, 8],
                'dataset_sha256': common.digest_file(dataset),
                'tokenizer_sha256': common.digest_file(ROOT / 'configs/tokenizer_mac.json'),
                'evaluator_sha256': common.digest_file(ROOT / 'telos/eval/anticheat.py'),
                'limitations': ['Native anti-cheat heuristic; a boundary match does not establish copying causally.',
                    'Requested K is capped by target token length; actual spans are separately reported.',
                    'Native context-aware target tokenization is retained for comparability; BPE boundary merges may affect targets.',
                    'Shortened target spans retain the supplied suffix, as implemented in the native evaluator.']}
    save(OUT / 'anticheat_protocol.json', manifest)
    for label in LABELS:
        destination = OUT / f'{label}.anticheat.json'
        if destination.exists():
            continue
        model, metadata = common.model_metadata(ROOT / 'checkpoints/corosred/unified' / label / 'checkpoint_final.pt')
        started = time.perf_counter()
        rows = []
        with torch.inference_mode():
            for index, probe in enumerate(probes):
                result = evaluate_span_infill_probe(model, tok, 'pytorch', probe['prompt'],
                    probe.get('multi_token_target') or probe['target'], probe.get('suffix', ''),
                    span_lengths=[1, 2, 4, 8])
                rows.append({'id': probe.get('id', str(index)), 'category': probe.get('category'), 'spans': result})
                if (index + 1) % 50 == 0:
                    save(OUT / 'anticheat_progress.json', {'label': label, 'completed': index + 1, 'total': len(probes)})
                    print(label, index + 1, '/', len(probes), flush=True)
        summaries = summarize_anticheat_suite([r['spans'] for r in rows])
        actual = {}
        for key in ['span_1', 'span_2', 'span_4', 'span_8']:
            groups = collections.defaultdict(list)
            for row in rows:
                if key in row['spans']:
                    item = row['spans'][key]
                    groups[item['actual_k']].append(item)
            actual[key] = {str(k): {'count': len(items),
                'exact_match_pct': 100 * sum(i['exact_match'] for i in items) / len(items),
                'suffix_copy_rate_pct': 100 * sum(i['copied_suffix_first'] for i in items) / len(items)}
                for k, items in sorted(groups.items())}
        save(destination, {'label': label, 'metadata': metadata, 'summary': summaries,
            'actual_span_breakdown': actual, 'seconds': time.perf_counter() - started, 'probes': rows})
        print('[Completed]', label, json.dumps(summaries), flush=True)
        del model
    save(OUT / 'anticheat_status.json', {'status': 'complete'})

if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        save(OUT / 'anticheat_status.json', {'status': 'error', 'message': str(exc)})
        raise
