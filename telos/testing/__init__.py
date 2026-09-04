"""
Unified testing and verification package for Télos.
Exposes test runner and CLI entrypoints.
"""

from .suite import run_unified_testing_suite, main

__all__ = [
    "run_unified_testing_suite",
    "main",
]
