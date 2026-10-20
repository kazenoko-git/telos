"""Fresh interpreter worker: execute supplied candidate/tests, without source repair."""
import contextlib
import io
import json
import socket
import sys

payload = json.load(sys.stdin)
try:
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
except (ImportError, ValueError, OSError):
    pass

def no_network(*args, **kwargs):
    raise PermissionError('Network disabled during benchmark execution')
socket.socket = no_network

try:
    compiled = compile(payload['code'], '<humaneval_candidate>', 'exec')
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        exec(compiled, {'__name__': '__main__'})
    result = {'outcome': 'PASSED', 'details': 'All official assertions passed'}
except SyntaxError as exc:
    result = {'outcome': 'SYNTAX_ERROR', 'details': str(exc)}
except AssertionError as exc:
    result = {'outcome': 'FAILED_ASSERTION', 'details': str(exc)}
except MemoryError:
    result = {'outcome': 'MEMORY_EXCEEDED', 'details': 'Memory limit exceeded'}
except BaseException as exc:
    result = {'outcome': 'RUNTIME_EXCEPTION', 'details': type(exc).__name__ + ': ' + str(exc)}
sys.stdout.write(json.dumps(result) + '\n')
