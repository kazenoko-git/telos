"""Unit tests for model parameter counting and architecture contracts."""

import torch
from telos.models.transformer import TelosTransformer, TelosConfig
from telos.models.param_counter import count_parameters, verify_with_model


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


def test_grouped_query_attention():
    """Verifies GQA forward pass and parameter calculation matching."""
    config = TelosConfig(
        vocab_size=1000,
        d_model=128,
        n_layers=2,
        n_heads=8,
        n_kv_heads=2,  # 4 query heads per KV head (GQA)
        max_seq_len=64
    )
    model = TelosTransformer(config)
    analytical_count = count_parameters(config)["total"]
    actual_count = verify_with_model(config)

    assert analytical_count == actual_count
    
    # Test forward pass with GQA
    x = torch.randint(0, 1000, (2, 16))
    out = model(x)
    assert out.shape == (2, 16, 1000)
