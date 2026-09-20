"""
Unit tests verifying global seed setting and deterministic model initialization.
"""

import torch
import numpy as np
from telos.training.core import set_global_seed
from telos.models import TelosTransformer
from telos.configs import build_config


def test_set_global_seed_pytorch_determinism():
    """Verify that set_global_seed produces identical model initialization weights."""
    set_global_seed(42)
    m1 = TelosTransformer(vocab_size=100, d_model=64, n_layers=2, n_heads=2)
    w1 = m1.tok_embeddings.weight.clone()

    set_global_seed(42)
    m2 = TelosTransformer(vocab_size=100, d_model=64, n_layers=2, n_heads=2)
    w2 = m2.tok_embeddings.weight.clone()

    assert torch.equal(w1, w2), "Models initialized with same seed must have identical weights"


def test_set_global_seed_different_seeds():
    """Verify that different seeds produce different initialization weights."""
    set_global_seed(42)
    m1 = TelosTransformer(vocab_size=100, d_model=64, n_layers=2, n_heads=2)

    set_global_seed(999)
    m2 = TelosTransformer(vocab_size=100, d_model=64, n_layers=2, n_heads=2)

    assert not torch.equal(m1.tok_embeddings.weight, m2.tok_embeddings.weight)


def test_builder_records_seed():
    """Verify that build_config correctly sets and records the seed."""
    cfg = build_config(paradigm="ar", params="12M", seed=123)
    assert cfg["training"]["seed"] == 123
    assert cfg["seed"] == 123

    # Default seed should be 42
    cfg_default = build_config(paradigm="ar", params="12M")
    assert cfg_default["training"]["seed"] == 42
    assert cfg_default["seed"] == 42
