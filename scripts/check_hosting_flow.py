"""Explicit live hosting smoke test, using a disposable account and tiny CPU model."""
import argparse
import json
import os
from pathlib import Path
import runpy
import secrets
import signal
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    report = {'tiny_cpu_only': True}
    with tempfile.TemporaryDirectory(prefix='telos-flow-') as directory:
        root = Path(directory)
        environment = {**os.environ, 'TELOS_HOSTING_CREDENTIALS': str(root / 'session.json')}
        slug = 'flow-' + secrets.token_hex(5)
        password = secrets.token_urlsafe(24)
        email = slug + '@example.invalid'
        process = None
        keys = []

        def cli(*words, stdin=None):
            result = subprocess.run([sys.executable, '-m', 'telos.cli', *words], cwd=repo,
                                    env=environment, input=stdin, text=True, capture_output=True, timeout=60)
            if result.returncode:
                raise RuntimeError('CLI failed: ' + ' '.join(words[:2]) + ' (credentials omitted)')
            return json.loads(result.stdout)

        def request(key, path='/models', data=None):
            req = Request('https://tls.wingit.tech/' + slug + '/v1' + path,
                          headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json',
                                   'User-Agent': 'telos-ml/hosting-flow'},
                          data=None if data is None else json.dumps(data).encode())
            try:
                with urlopen(req, timeout=15) as response:
                    return response.status, response.read().decode()
            except HTTPError as error:
                return error.code, error.read().decode()

        def stop():
            nonlocal process
            if process and process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=40)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            process = None

        try:
            account = cli('account', 'signup', '--email', email, stdin=password + '\n')
            report['signup'] = account['email'] == email
            key = cli('keys', 'create', 'disposable flow test')
            keys.append(key['id'])
            report['key_created'] = bool(key['key'])
            reserved = cli('tunnel', 'reserve', slug, '--model', 'fixture')
            report['reserved_path'] = reserved['public_url'] == f'https://tls.wingit.tech/{slug}/v1'
            checkpoint = root / 'checkpoint'
            checkpoint.mkdir()
            runpy.run_path(str(repo / 'tests/test_serve.py'))['serve_checkpoint'].__wrapped__(checkpoint)
            from telos.inference import TextGenerator
            expected = TextGenerator.from_checkpoint(checkpoint, device='cpu').generate('hello', max_new_tokens=4, temperature=0)
            body = {'model': 'fixture', 'prompt': 'hello', 'max_tokens': 4, 'temperature': 0}
            for attempt in range(2):
                with (root / f'serve-{attempt}.log').open('w') as log:
                    process = subprocess.Popen([sys.executable, '-m', 'telos.cli', 'serve', '--checkpoint', str(checkpoint),
                        '--device', 'cpu', '--model-name', 'fixture', '--tunnel', slug, '--port', '18766',
                        '--context-length', '32'], cwd=repo, env=environment, stdout=log, stderr=log)
                    deadline = time.monotonic() + 180
                    while time.monotonic() < deadline:
                        if process.poll() is not None:
                            raise RuntimeError('Tiny server exited; private logs omitted.')
                        try:
                            status, result = request(key['key'])
                            if status == 200 and any(row['id'] == 'fixture' for row in json.loads(result)['data']):
                                break
                        except (URLError, TimeoutError):
                            pass
                        time.sleep(2)
                    else:
                        raise TimeoutError('Public endpoint did not become ready.')
                    status, result = request(key['key'], '/completions', body)
                    report['initial_exact' if attempt == 0 else 'restart_same_key_exact'] = status == 200 and json.loads(result)['choices'][0]['text'] == expected
                    status, result = request(key['key'], '/completions', {**body, 'stream': True})
                    chunks = [json.loads(line[6:]) for line in result.splitlines() if line.startswith('data: ') and line != 'data: [DONE]']
                    generated = ''.join(chunk['choices'][0].get('text', '') for chunk in chunks if chunk.get('choices'))
                    report[f'stream_{attempt}_exact'] = status == 200 and generated == expected and 'data: [DONE]' in result
                    if attempt == 0:
                        stop()
                        report['session_persisted'] = cli('account', 'whoami')['email'] == email
                        report['reservation_persisted'] = any(host['slug'] == slug for host in cli('tunnel', 'list'))
                    else:
                        rotated = cli('keys', 'rotate', key['id'])
                        keys.append(rotated['id'])
                        report['old_key_rejected'] = request(key['key'])[0] == 401
                        status, result = request(rotated['key'], '/completions', body)
                        report['rotated_key_exact'] = status == 200 and json.loads(result)['choices'][0]['text'] == expected
                        cli('keys', 'revoke', rotated['id'])
                        report['revoked_key_rejected'] = request(rotated['key'])[0] == 401
            assert all(report.values()), report
        finally:
            stop()
            for key_id in keys:
                try:
                    cli('keys', 'revoke', key_id)
                except RuntimeError:
                    pass
            try:
                cli('account', 'logout')
            except RuntimeError:
                pass
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
