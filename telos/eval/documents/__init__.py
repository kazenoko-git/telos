"""General-purpose structured document evaluation; no provider-specific adapters."""
from .suite import load_suite, reference_responses

__all__ = ['load_suite', 'reference_responses', 'evaluate_documents', 'replay_responses']


def __getattr__(name):
    if name in ('evaluate_documents', 'replay_responses'):
        from . import runner
        return getattr(runner, name)
    raise AttributeError(name)
