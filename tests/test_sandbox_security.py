"""Real sandbox regressions: launching must not broaden the isolation policy."""
import subprocess
import socket
import sys
from pathlib import Path

import pytest

from telos.eval.executor import ExecutionResult, execute_code_sandboxed
from telos.eval.sandbox import sandbox_command


def assert_passed(code, **kwargs):
    status, details = execute_code_sandboxed(code, **kwargs)
    assert status == ExecutionResult.PASSED, details


def test_scratch_alias_and_python_runtime_start(tmp_path):
    real = tmp_path / 'real'
    real.mkdir()
    alias = tmp_path / 'alias'
    alias.symlink_to(real, target_is_directory=True)
    assert_passed("from pathlib import Path\np=Path('result.txt')\np.write_text('ok')\nassert p.read_text() == 'ok'",
                  scratch_dir=str(alias))
    assert (real / 'result.txt').read_text() == 'ok'


def test_unrelated_file_read_and_write_remain_denied(tmp_path):
    outside = tmp_path / 'outside.txt'
    outside.write_text('must remain private')
    code = f'''from pathlib import Path
p = Path({str(outside)!r})
for operation in (p.read_text, lambda: p.write_text('overwritten')):
    try:
        operation()
    except (PermissionError, FileNotFoundError):
        pass
    else:
        raise AssertionError('outside scratch access was permitted')
'''
    assert_passed(code)
    assert outside.read_text() == 'must remain private'


def test_network_is_still_denied():
    # A known live parent listener distinguishes isolation from a closed port.
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        listener.listen()
        port = listener.getsockname()[1]
        assert_passed(f"""import socket
try:
    sock = socket.socket()
    sock.settimeout(.2)
    sock.connect(('127.0.0.1', {port}))
except OSError:
    pass
else:
    raise AssertionError('parent network listener was reachable')
""")
        listener.settimeout(.1)
        with pytest.raises(socket.timeout):
            listener.accept()


def test_environment_is_scrubbed(monkeypatch):
    monkeypatch.setenv('TELOS_SANDBOX_SENTINEL', 'do-not-inherit')
    assert_passed("import os\nassert 'TELOS_SANDBOX_SENTINEL' not in os.environ")


@pytest.mark.skipif(sys.platform != 'darwin', reason='macOS process-fork policy')
def test_python_worker_cannot_fork(tmp_path):
    # Run directly under the OS policy, without the executor's RLIMIT_NPROC.
    # This ensures the sandbox itself denies fork, independently of user load.
    code = """import os
try:
    child = os.fork()
except PermissionError:
    pass
else:
    if child == 0:
        os._exit(0)
    os.waitpid(child, 0)
    raise AssertionError('process forking was permitted')
"""
    command, env = sandbox_command([sys.executable, '-I', '-S', '-c', code], tmp_path/'scratch')
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=3)
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(sys.platform != 'darwin', reason='macOS literal sandbox rules')
def test_macos_launch_rules_are_narrow(tmp_path):
    entry = tmp_path / 'python-alias'
    entry.symlink_to(Path(sys.executable).resolve())
    command, env = sandbox_command([str(entry), '-I', '-S', '-c', 'pass'], tmp_path/'scratch')
    profile = command[2]
    assert command[3] == str(Path(sys.executable).resolve())
    assert '(allow file-read-data (literal "/"))' in profile
    assert '(subpath "/")' not in profile
    assert '(allow process-fork)' not in profile
    assert '(deny network*)' in profile
    assert env['HOME'] == str((tmp_path/'scratch').resolve())
