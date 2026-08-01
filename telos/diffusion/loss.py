"""ELBO-consistent Loss Function for MDLM (Masked Diffusion Language Modeling)
i have no idea how the fuck this works
"""

import torch
import torch.nn.functional as F


def mdlm_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask_positions: torch.Tensor,
    t_values: torch.Tensor,
    label_smoothing: float = 0.0
) -> tuple[torch.Tensor, dict[str, float]]:
    """compute 1/t reweighted MDLM Cross-Entropy loss.

    Args:
        TO BE FILLED LATER

    Returns:
        TO BE FILLED LATER
    """
    batch_size, seq_len, vocab_size = logits.shape

    # compute per-token Cross Entropy loss across all positions (reduction='none'): shape [batch_size, seq_len]
    ce_loss_per_token = F.cross_entropy(
        logits.view(-1, vocab_size),
        targets.view(-1),
        reduction="none",
        label_smoothing=label_smoothing
    ).view(batch_size, seq_len)

    # mask out unmasked positions (only calculate loss on tokens replaced with [MASK])
    masked_ce_loss = ce_loss_per_token * mask_positions.float()

    # calculate per-example mean cross-entropy over its masked positions
    masked_count_per_example = mask_positions.sum(dim=1).float().clamp(min=1.0)
    per_example_ce = masked_ce_loss.sum(dim=1) / masked_count_per_example

    # apply 1/t ELBO loss reweighting (t_values shape: [batch_size, 1] -> squeeze to [batch_size])
    t_weights = 1.0 / t_values.squeeze(-1)
    reweighted_per_example_loss = per_example_ce * t_weights

    # final loss is average reweighted loss across batch
    total_loss = reweighted_per_example_loss.mean()

    # calculate unweighted metric for monitoring
    unweighted_ce = per_example_ce.mean().item()

    metrics = {
        "loss": total_loss.item(),
        "unweighted_ce": unweighted_ce,
        "masked_tokens_avg": mask_positions.sum().float().item() / batch_size
    }

    return total_loss, metrics
