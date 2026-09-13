"""
Comprehensive Unit Test Suite for Télos Next-Generation Evaluation System.

Verifies:
1. Inline AST syntax validity checking and error categorization.
2. Subprocess sandboxed execution with timeout, memory cap, and status enums.
3. Anti-cheat boundary token copy detection (suffix-copying and prefix-copying).
4. 13-gram contamination screening.
5. Contextual probe loading and benchmark structure.
6. Statistical bootstrap confidence interval computation.
"""

import time
import pytest
from telos.eval.syntax import check_ast_validity, categorize_syntax_error, analyze_syntax_batch
from telos.eval.executor import execute_code_sandboxed, ExecutionResult
from telos.eval.anticheat import compute_boundary_copy_metrics, summarize_anticheat_suite
from telos.eval.contamination import extract_token_ngrams, ContaminationDetector
from telos.eval.probes import load_contextual_probes
from telos.eval.runner import bootstrap_confidence_interval


class TestSyntaxAnalyzer:
    """Tests for inline AST syntax checking."""

    def test_valid_code(self):
        code = "def add(a: int, b: int) -> int:\n    return a + b\n"
        is_valid, err = check_ast_validity(code)
        assert is_valid is True
        assert err is None
        assert categorize_syntax_error(err) == "valid"

    def test_syntax_error(self):
        code = "def bad_func(:\n    pass"
        is_valid, err = check_ast_validity(code)
        assert is_valid is False
        assert err is not None
        assert "SyntaxError" in err

    def test_indentation_error(self):
        code = "def foo():\npass\n"
        is_valid, err = check_ast_validity(code)
        assert is_valid is False
        assert err is not None
        assert categorize_syntax_error(err) == "indentation"

    def test_unclosed_delimiter(self):
        code = "x = (1 + 2"
        is_valid, err = check_ast_validity(code)
        assert is_valid is False
        assert err is not None
        assert categorize_syntax_error(err) == "unclosed_delimiter"

    def test_unterminated_string(self):
        code = 'msg = "hello world'
        is_valid, err = check_ast_validity(code)
        assert is_valid is False
        assert err is not None
        assert categorize_syntax_error(err) == "unclosed_delimiter"

    def test_syntax_batch_analysis(self):
        batch = [
            "x = 10\nprint(x)",
            "def foo(:\npass",
            "for i in range(5):\nprint(i)",
        ]
        res = analyze_syntax_batch(batch)
        assert res["total_count"] == 3
        assert res["valid_count"] == 1
        assert res["valid_pct"] == 33.33
        assert res["error_breakdown"]["valid"] == 1
        assert res["error_breakdown"]["indentation"] == 1


class TestSandboxExecutor:
    """Tests for subprocess sandboxed execution engine."""

    def test_passed_execution(self):
        code = "def multiply(a, b):\n    return a * b\n"
        test = "assert multiply(3, 4) == 12\nassert multiply(-1, 5) == -5\n"
        status, details = execute_code_sandboxed(code, test_harness=test, timeout_seconds=2.0)
        assert status == ExecutionResult.PASSED

    def test_failed_assertion(self):
        code = "def multiply(a, b):\n    return a + b\n"
        test = "assert multiply(3, 4) == 12\n"
        status, details = execute_code_sandboxed(code, test_harness=test, timeout_seconds=2.0)
        assert status == ExecutionResult.FAILED_ASSERTION

    def test_syntax_error_in_sandbox(self):
        code = "def broken(\n    return 42"
        status, details = execute_code_sandboxed(code, timeout_seconds=2.0)
        assert status == ExecutionResult.SYNTAX_ERROR

    def test_runtime_exception(self):
        code = "def crash():\n    return 1 / 0\ncrash()\n"
        status, details = execute_code_sandboxed(code, timeout_seconds=2.0)
        assert status == ExecutionResult.RUNTIME_EXCEPTION
        assert "ZeroDivisionError" in details

    def test_hard_timeout_forced_kill(self):
        # Infinite loop that never cooperative yields
        code = "while True:\n    pass\n"
        start_t = time.time()
        status, details = execute_code_sandboxed(code, timeout_seconds=1.0)
        elapsed = time.time() - start_t
        assert status == ExecutionResult.TIMEOUT
        assert elapsed < 2.5, f"Timeout took too long ({elapsed}s), forced kill failed!"


class TestAntiCheatEngine:
    """Tests for boundary token copy detection and span infill metrics."""

    def test_suffix_copy_detection(self):
        prefix_tokens = [10, 20, 30]
        suffix_tokens = [99, 100]  # Suffix begins with 99
        target_tokens = [45]
        # Model predicted 99 (literally copied suffix[0])
        predicted_tokens = [99]

        metrics = compute_boundary_copy_metrics(
            predicted_tokens=predicted_tokens,
            prefix_tokens=prefix_tokens,
            suffix_tokens=suffix_tokens,
            target_tokens=target_tokens
        )
        assert metrics["copied_suffix_first"] is True
        assert metrics["copied_prefix_last"] is False
        assert metrics["is_exact_match"] is False

    def test_prefix_copy_detection(self):
        prefix_tokens = [10, 20, 30]  # Prefix ends with 30
        suffix_tokens = [99, 100]
        target_tokens = [45]
        # Model predicted 30 (literally copied prefix[-1])
        predicted_tokens = [30]

        metrics = compute_boundary_copy_metrics(
            predicted_tokens=predicted_tokens,
            prefix_tokens=prefix_tokens,
            suffix_tokens=suffix_tokens,
            target_tokens=target_tokens
        )
        assert metrics["copied_suffix_first"] is False
        assert metrics["copied_prefix_last"] is True
        assert metrics["is_exact_match"] is False

    def test_exact_match(self):
        prefix_tokens = [1, 2]
        suffix_tokens = [4, 5]
        target_tokens = [42]
        predicted_tokens = [42]

        metrics = compute_boundary_copy_metrics(
            predicted_tokens=predicted_tokens,
            prefix_tokens=prefix_tokens,
            suffix_tokens=suffix_tokens,
            target_tokens=target_tokens
        )
        assert metrics["is_exact_match"] is True
        assert metrics["token_accuracy"] == 1.0
        assert metrics["copied_suffix_first"] is False
        assert metrics["copied_prefix_last"] is False


class TestContaminationDetector:
    """Tests for 13-gram token extraction and overlap detection."""

    def test_ngram_extraction(self):
        tokens = list(range(20))
        ngrams = extract_token_ngrams(tokens, n=13)
        assert len(ngrams) == 8  # 20 - 13 + 1 = 8
        assert tuple(range(13)) in ngrams

    def test_short_tokens(self):
        tokens = [1, 2, 3]
        ngrams = extract_token_ngrams(tokens, n=13)
        assert len(ngrams) == 0


class TestBenchmarkLoader:
    """Tests for probe loading and confidence interval calculations."""

    def test_load_contextual_probes(self):
        probes = load_contextual_probes(num_probes=50)
        assert len(probes) == 50
        sample = probes[0]
        assert "prompt" in sample
        assert "target" in sample
        assert "category" in sample
        assert "suffix" in sample

    def test_bootstrap_ci(self):
        outcomes = [1.0] * 80 + [0.0] * 20  # 80% accuracy
        lower, upper = bootstrap_confidence_interval(outcomes, n_resamples=500)
        assert 65.0 <= lower <= 80.0
        assert 80.0 <= upper <= 95.0
