"""
Evaluation Metrics for télos Masked Diffusion Language Model.
"""

import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from mdiff.diffusion.loss import mdlm_loss


@torch.no_grad()
def evaluate_perplexity(
    model: nn.Module,
    dataloader: DataLoader,
    device: str | torch.device = "cpu"
) -> dict[str, float]:
    model.eval()
    device = torch.device(device)
    model.to(device)

    total_loss = 0.0
    total_ce = 0.0
    num_batches = 0

    for masked_input_ids, targets, mask_positions, t_values in dataloader:
        masked_input_ids = masked_input_ids.to(device)
        targets = targets.to(device)
        mask_positions = mask_positions.to(device)
        t_values = t_values.to(device)

        logits = model(masked_input_ids)
        loss, batch_metrics = mdlm_loss(logits, targets, mask_positions, t_values)

        # .item() is fine here — eval runs rarely, need scalar accumulation
        total_loss += batch_metrics["loss"].item()
        total_ce += batch_metrics["unweighted_ce"].item()
        num_batches += 1

    avg_loss = total_loss / max(1, num_batches)
    avg_ce = total_ce / max(1, num_batches)
    # Perplexity is exp(unweighted_ce)
    perplexity = math.exp(min(avg_ce, 20.0))  # Clamp exponent to prevent overflow

    return {
        "val_loss": avg_loss,
        "unweighted_ce": avg_ce,
        "perplexity": perplexity
    }
