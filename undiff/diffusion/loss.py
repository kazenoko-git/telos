"""
ELBO-consistent Loss Function for UNDLM (Uniform Noise Diffusion Language Modeling)

Loss is computed at all content positoins.
"""

import mlx.core as mx
import mlx.nn as nn

def undlm_loss(model, noisy_ids, clean_targets, t_values, vocab_size, special_token_lut=None):
    # compute 1/t reweighted UNDLM cross entropy loss over all content positions
    logits = model(noisy_ids) # (B, T, V)
    B, T, V = logits.shape

    # Upcast logits to float32 for numerically stable log-softmax in cross entropy
    logits_flat = logits.reshape(-1, V) # (B*T, V)
    targets_flat = clean_targets.reshape(-1) # (B*T,)

    # CE on every position in float32
    ce_per_token = nn.losses.cross_entropy(logits_flat, targets_flat, reduction="none").reshape(B, T)

    # Exclude special tokens (PAD, MASK, BOS, EOS) from CE loss if LUT provided
    if special_token_lut is not None:
        content_mask = ~special_token_lut[clean_targets] # [B, T] True for content
        ce_per_token = ce_per_token * content_mask.astype(mx.float32)
        content_count = mx.clip(mx.sum(content_mask.astype(mx.float32), axis=1), 1.0, float(T))
        per_example_ce = mx.sum(ce_per_token, axis=1) / content_count # [B]
    else:
        per_example_ce = mx.mean(ce_per_token, axis=1) # [B]

    unweighted_ce = mx.mean(per_example_ce)

    # 1/t ELBO reweighting
    t_weights = 1.0 / mx.clip(mx.squeeze(t_values, -1), 1e-3, 1.0)
    reweighted_loss = mx.mean(per_example_ce * t_weights)

    return reweighted_loss, unweighted_ce