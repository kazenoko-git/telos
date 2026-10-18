"""Resumable seven-suite assessment against local LM Studio Gemma 4 E4B.

Run with the repository's Python environment, with Gemma already loaded and
LM Studio's server listening. Stores complete API responses per request so an
interrupted suite can resume without repeating completed generations.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from telos.eval.adapters.openai_api import OpenAIAPIAdapter
from telos.eval.runner import evaluate

SUITES = {
    'arc': ('science', 'arc_challenge_suite.json', 1024),
    'mmlu_science': ('science', 'mmlu_science_suite.json', 2048),
    'gpqa_diamond': ('science', 'gpqa_diamond_suite.json', 8192),
    'competition_math': ('math', 'competition_math_suite.json', 8192),
    'humaneval': ('code', 'humaneval_suite.json', 2048),
    'tooluse': ('tooluse', 'tooluse_suite.json', 2048),
    'cyber': ('cyber', 'cybersecurity_suite.json', 4096),
}


class RecordedAdapter(OpenAIAPIAdapter):
    def __init__(self, model, journal, token_budget):
        super().__init__(model, api_base='http://127.0.0.1:1234/v1',
                         concurrency=1, timeout=600)
        self.journal = Path(journal)
        self.token_budget = token_budget
        self.cache = {}
        self.latest_response = None
        if self.journal.exists():
            for line in self.journal.read_text(encoding='utf-8').splitlines():
                record = json.loads(line)
                self.cache[record['key']] = record['response']

    def _make_http_request(self, endpoint, payload, max_retries=5):
        payload = dict(payload, max_tokens=self.token_budget)
        serialized = json.dumps({'endpoint': endpoint, 'payload': payload}, sort_keys=True)
        key = hashlib.sha256(serialized.encode()).hexdigest()
        if key in self.cache:
            self.latest_response = self.cache[key]
            print(f'  [Cached response] {key[:12]}', flush=True)
            return self.latest_response
        started = time.perf_counter()
        response = super()._make_http_request(endpoint, payload, max_retries=max_retries)
        seconds = time.perf_counter() - started
        record = {'key': key, 'request': payload, 'response': response, 'seconds': seconds}
        with self.journal.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            stream.flush()
        self.cache[key] = response
        self.latest_response = response
        usage = response.get('usage', {})
        finish = response.get('choices', [{}])[0].get('finish_reason')
        print(f'  [Generated] {seconds:.1f}s | tokens={usage.get("completion_tokens")} '
              f'| finish={finish}', flush=True)
        return response

    def generate(self, prompt, max_new_tokens=256, temperature=0.0, stop=None, **kwargs):
        # Retain the base adapter's request construction, but never score its
        # reasoning-only fallback as a final answer when content is absent.
        super().generate_with_finish_reason(prompt, max_new_tokens, temperature, stop, **kwargs)
        message = self.latest_response['choices'][0].get('message', {})
        answer = message.get('content') or ''
        if '</think>' in answer:
            answer = answer.rsplit('</think>', 1)[-1]
        if '<|channel>thought' in answer:
            answer = answer.split('<channel|>', 1)[1] if '<channel|>' in answer else ''
        return answer.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model', default='google/gemma-4-e4b')
    parser.add_argument('--suites', nargs='+', choices=list(SUITES), default=list(SUITES))
    parser.add_argument('--max-tasks', type=int)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    suite_info = {}
    for name in args.suites:
        kind, filename, budget = SUITES[name]
        path = ROOT / 'evals/benchmarks' / filename
        data = path.read_bytes()
        suite_info[name] = {'benchmark_type': kind, 'source': filename,
                            'source_tasks': len(json.loads(data)),
                            'sha256': hashlib.sha256(data).hexdigest(),
                            'max_tokens': budget}
    config = {'model': args.model, 'backend': 'LM Studio local API',
              'model_format': 'Installed 4-bit MLX; verify loaded model metadata',
              'temperature': 0.0, 'thinking': 'LM Studio loaded-model defaults',
              'max_tasks': args.max_tasks, 'suites': suite_info,
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return
    with urllib.request.urlopen('http://127.0.0.1:1234/v1/models', timeout=5) as response:
        available = json.load(response)
    if args.model not in {item['id'] for item in available.get('data', [])}:
        raise RuntimeError(f'Load {args.model} in LM Studio before starting')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / 'configuration.json'
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise RuntimeError('Run configuration changed; select a new output directory')
    manifest.write_text(json.dumps(config, indent=2), encoding='utf-8')
    try:
        metadata = subprocess.check_output(
            [str(Path.home() / '.lmstudio/bin/lms'), 'ps', '--json'], text=True)
        (args.output_dir / 'loaded_model.json').write_text(metadata, encoding='utf-8')
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    summary = {}
    for name in args.suites:
        info = suite_info[name]
        report_path = args.output_dir / f'{name}_report.json'
        if report_path.exists():
            report = json.loads(report_path.read_text())
            print(f'[Already complete] {name}', flush=True)
        else:
            adapter = RecordedAdapter(args.model, args.output_dir / f'{name}_responses.jsonl',
                                      info['max_tokens'])
            print(f'\n[Starting] {name}: budget={info["max_tokens"]}', flush=True)
            report = evaluate(adapter, mode='functional', suite=name,
                              benchmark_type=info['benchmark_type'],
                              max_tasks=args.max_tasks, output_path=str(report_path))
        summary[name] = {'report_file': str(report_path), 'complete': True}
        (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'Completed assessment: {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
