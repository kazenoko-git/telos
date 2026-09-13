"""
Télos Evaluation Suite Module.

Exports:
- evaluate: Master programmatic entrypoint for model evaluation.
- main: CLI runner entrypoint.
- check_ast_validity, categorize_syntax_error: AST syntax parsing and diagnostics.
- execute_code_sandboxed, ExecutionResult: Sandboxed subprocess code execution.
- evaluate_probes: Contextual probes benchmark (1,000 probes across 8 categories).
- evaluate_functional: Pass@1 functional unit testing with sandboxing.
- evaluate_anticheat: Suffix-copy cheat and multi-token chunk masking detector.
- load_contextual_probes, PROBE_SUITE_100: Benchmark probe suites.
"""

import sys
import types
from .runner import evaluate, main, evaluate_probes, evaluate_functional, evaluate_anticheat
from .probes import PROBE_SUITE_100, load_contextual_probes
from .syntax import check_ast_validity, categorize_syntax_error
from .executor import execute_code_sandboxed, ExecutionResult


class _EvalModule(types.ModuleType):
    def __call__(self, *args, **kwargs):
        """Evaluates model directly when calling telos.eval(...)"""
        return evaluate(*args, **kwargs)


sys.modules[__name__].__class__ = _EvalModule

__all__ = [
    "evaluate",
    "main",
    "evaluate_probes",
    "evaluate_functional",
    "evaluate_anticheat",
    "check_ast_validity",
    "categorize_syntax_error",
    "execute_code_sandboxed",
    "ExecutionResult",
    "PROBE_SUITE_100",
    "load_contextual_probes",
]
