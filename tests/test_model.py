"""Unit tests for model parameter counting and architecture contracts."""

import torch
from telos.model.transformer import TelosTransformer, TelosConfig
from telos.model.param_counter import count_parameters, verify_with_model


def test_parameter_count_matching():
    """Verifies analytical parameter count matches numel() instantiation."""
    config = TelosConfig(
        vocab_size=4096,
        d_model=128,
        n_layers=6,
        n_heads=4,
        max_seq_len=256,
        tied_embeddings=True
    )
    analytical_count = count_parameters(config)["total"]
    actual_count = verify_with_model(config)

    assert analytical_count == actual_count, \
        f"Param mismatch: analytical={analytical_count}, actual={actual_count}"


def test_embedding_weight_tying():
    """Verifies token embedding and output projection share exact memory address."""
    config = TelosConfig(vocab_size=1000, d_model=64, tied_embeddings=True)
    model = TelosTransformer(config)

    assert model.tok_embeddings.weight is model.output_projection.weight
