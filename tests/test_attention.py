"""Unit test for verifying bidirectional self-attention.

Directly tests that attention is NOT causal:
- Modifying a token at position i (right) MUST change the hidden representation
  at position j < i (left).
"""

import torch
from mdiff.model.transformer import TelosTransformer, TelosConfig


def test_attention_is_bidirectional():
    """Verifies that changing a right-hand token affects left-hand representations."""
    config = TelosConfig(
        vocab_size=100,
        d_model=64,
        n_layers=2,
        n_heads=2,
        max_seq_len=64,
        dropout=0.0
    )
    model = TelosTransformer(config)
    model.eval()

    # Initial input sequence: [10, 20, 30, 40, 50]
    seq_a = torch.tensor([[10, 20, 30, 40, 50]], dtype=torch.long)
    # Modified sequence: change right-hand token at position 4 from 50 to 99
    seq_b = torch.tensor([[10, 20, 30, 40, 99]], dtype=torch.long)

    with torch.no_grad():
        logits_a = model(seq_a)  # [1, 5, 100]
        logits_b = model(seq_b)  # [1, 5, 100]

    diff_pos_0 = (logits_a[0, 0] - logits_b[0, 0]).abs().sum().item()

    # In a causal model, diff_pos_0 would be exactly 0.0.
    # In a bidirectional model, diff_pos_0 MUST be strictly > 0.0.
    assert diff_pos_0 > 1e-4, f"Attention behaves causally! Left token logits unchanged (diff={diff_pos_0})"
