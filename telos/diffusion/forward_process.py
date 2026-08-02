"""forward (noising) process for MDLM (Masked Diffusion Language Modeling)

in absorbing-state discrete diffusion:
- sample a time step t ~ Uniform(eps, 1.0) independently for each sequence in a batch.
- each content token position is independently replaced with a special [MASK] token
  with probability t.
- special tokens (PAD, BOS, EOS) are explicitly preserved and NEVER masked.
"""

import torch

def apply_masking(
    input_ids: torch.Tensor,
    mask_token_id: int,
    special_token_ids: set[int] | None = None,
    eps: float = 1e-5
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ppplies the MDLM forward process by masking content tokens dynamically.

    args:
        input_ids: tensor of shape [batch_size, seq_len] with original clean token IDs.
        mask_token_id: token ID reserved for [MASK].
        special_token_ids: set of token IDs (PAD, BOS, EOS) that should NOT be masked.
        eps: minimum value for t to prevent division by zero in 1/t loss reweighting.

    Returns:
        masked_input_ids: tensor of shape [batch_size, seq_len] with masked tokens inserted.
        mask_positions: boolean tensor of shape [batch_size, seq_len], True where [MASK] was applied.
        t_values: tensor of shape [batch_size, 1] containing the sampled time t for each example.
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    if special_token_ids is None:
        special_token_ids = set()

    # sample t ~ Uniform(eps, 1.0) per example in the batch: shape [batch_size, 1]
    t_values = torch.zeros(batch_size, 1, device=device).uniform_(eps, 1.0)

    # draw random probability matrix [batch_size, seq_len] ~ Uniform(0, 1)
    rand_matrix = torch.rand(batch_size, seq_len, device=device)

    # determine raw mask positions: True where rand < t for that example
    raw_mask = rand_matrix < t_values

    # identify special tokens (PAD, BOS, EOS) that must stay unmasked
    is_special = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    for token_id in special_token_ids:
        is_special = is_special | (input_ids == token_id)

    # final mask positions: mask content tokens, exclude special tokens
    mask_positions = raw_mask & (~is_special)

    # construct masked input sequence: replace masked positions with mask_token_id
    # uses torch.where instead of in-place indexing for cleaner XLA/Metal graph tracing
    masked_input_ids = torch.where(
        mask_positions,
        torch.full_like(input_ids, mask_token_id),
        input_ids
    )

    return masked_input_ids, mask_positions, t_values
