"""
Exact analytical parameter & TPU v5e-8 / M5 Pro memory auditor for TelosConfig.

Computes exact parameter counts (total vs. active), per-chip TPU v5e-8 HBM
breakdown (persistent sharded state, transient AllGather, checkpointed vs.
uncheckpointed activations, XLA workspace), and M5 Pro active-byte decode budget.
"""

from dataclasses import dataclass
import math
from .config import TelosConfig


@dataclass
class HBMReport:
    total_params: int
    active_params_per_token: int
    active_body_params: int
    lm_head_params: int
    mtp_params: int
    persistent_state_gib_per_chip: float
    allgather_transient_gib: float
    activations_gib_no_ckpt: float
    activations_gib_with_ckpt: float
    xla_workspace_gib: float
    peak_gib_no_ckpt: float
    peak_gib_with_ckpt: float
    usable_hbm_gib: float
    fits_with_ckpt: bool
    fits_without_ckpt: bool
    m5_pro_single_tok_active_mb: float
    m5_pro_fixed_state_mb: float


def audit_config_memory(
    cfg: TelosConfig,
    *,
    num_chips: int = 8,
    fsdp_sharded: bool = True,
    local_batch_size: int = 4,
    seq_len: int | None = None,
    usable_hbm_gib: float = 15.0,
    xla_workspace_gib: float = 2.75,
) -> HBMReport:
    """Compute exact parameter counts and per-chip TPU v5e HBM estimates."""
    t = seq_len if seq_len is not None else cfg.max_seq_len
    d = cfg.d_model
    layer_types = cfg.resolved_layer_types()
    n_gdn = sum(1 for lt in layer_types if lt == "gdn")
    n_attn = len(layer_types) - n_gdn

    # Embeddings / tied LM head
    embed_params = cfg.vocab_size * d
    lm_head_params = embed_params
    total_embed = embed_params if cfg.tied_embeddings else 2 * embed_params

    # Sequence mixers
    gdn_heads = cfg.gdn_n_heads if cfg.gdn_n_heads is not None else cfg.n_heads
    gdn_hd = cfg.gdn_head_dim if cfg.gdn_head_dim is not None else (d // gdn_heads)
    gdn_inner = gdn_heads * gdn_hd
    # Q, K, V, G, O + alpha/beta + depthwise conv + norms
    gdn_layer_params = (
        4 * d * gdn_inner
        + gdn_inner * d
        + 2 * (d * gdn_heads + gdn_heads)
        + 3 * gdn_inner * (cfg.gdn_conv_kernel + 1)
        + gdn_hd
        + d
        + (gdn_inner if cfg.use_subln else 0)
    )

    n_kv = cfg.n_kv_heads if cfg.n_kv_heads is not None else cfg.n_heads
    attn_hd = d // cfg.n_heads
    attn_layer_params = (
        d * (cfg.n_heads * attn_hd)
        + 2 * d * (n_kv * attn_hd)
        + d * d
        + d
        + (d if cfg.use_subln else 0)
    )

    total_mixers = n_gdn * gdn_layer_params + n_attn * attn_layer_params

    # FFNs / MoE
    if cfg.ffn_type == "moe_relu2" or cfg.moe_num_experts > 0:
        d_ff = cfg.moe_d_ff if cfg.moe_d_ff is not None else (cfg.d_ff or 2 * d)
        shared_d_ff = cfg.moe_shared_d_ff or 0
        router_p = d * cfg.moe_num_experts
        one_expert_p = 2 * d * d_ff
        routed_total_p = cfg.moe_num_experts * one_expert_p + (d_ff if cfg.use_subln else 0)
        routed_active_p = cfg.moe_top_k * one_expert_p + (d_ff if cfg.use_subln else 0)
        shared_p = (2 * d * shared_d_ff + (shared_d_ff if cfg.use_subln else 0)) if shared_d_ff > 0 else 0
        ffn_layer_total = router_p + routed_total_p + shared_p + d
        ffn_layer_active = router_p + routed_active_p + shared_p + d
    elif cfg.ffn_type == "relu2":
        d_ff = cfg.d_ff if cfg.d_ff is not None else 2 * d
        ffn_layer_total = 2 * d * d_ff + d + (d_ff if cfg.use_subln else 0)
        ffn_layer_active = ffn_layer_total
    else:
        d_ff = cfg.d_ff if cfg.d_ff is not None else (64 * ((int(8 * d / 3) + 63) // 64))
        ffn_layer_total = 3 * d * d_ff + d
        ffn_layer_active = ffn_layer_total

    total_ffn = cfg.n_layers * ffn_layer_total
    active_ffn = cfg.n_layers * ffn_layer_active

    # MTP heads
    mtp_params = 0
    if cfg.mtp_num_heads > 0:
        mtp_ff = cfg.mtp_d_ff if cfg.mtp_d_ff is not None else 2 * d
        one_mtp = (2 * d * d) + (2 * d * mtp_ff) + 4 * d + (mtp_ff if cfg.use_subln else 0)
        mtp_params = cfg.mtp_num_heads * one_mtp

    total_params = total_embed + total_mixers + total_ffn + mtp_params + d
    active_body_params = total_mixers + active_ffn + d
    active_params_per_token = active_body_params + lm_head_params

    # TPU v5e persistent state: FP32 master (4B) + FP32 grad (4B) + AdamW m,v (8B) = 16 B/param
    shard_div = num_chips if fsdp_sharded else 1
    gib = 1024 ** 3
    persistent_state_gib = (total_params * 16.0 / shard_div) / gib

    # Transient AllGather for 2 prefetched layers (BF16 = 2B/param) under FSDP
    largest_layer = max(gdn_layer_params, attn_layer_params) + ffn_layer_total
    allgather_transient_gib = (2 * largest_layer * 2.0 / gib) if (fsdp_sharded and num_chips > 1) else 0.0

    # Activation memory per chip (BF16 = 2B)
    n_tok = local_batch_size * t
    if cfg.moe_num_experts > 0:
        d_ff = cfg.moe_d_ff if cfg.moe_d_ff is not None else (cfg.d_ff or 2 * d)
        cap = max(1, math.ceil((n_tok * cfg.moe_top_k / cfg.moe_num_experts) * cfg.moe_capacity_factor))
        # Dispatched (E*C*D) + W_up pre/post ReLU2 (2 * E*C*d_ff) + shared expert + QAT STE buffers
        moe_act_bytes = (cfg.moe_num_experts * cap * (d + 2 * d_ff) * 2) + (n_tok * (cfg.moe_shared_d_ff or 0) * 4)
    else:
        d_ff = cfg.d_ff or 2 * d
        moe_act_bytes = n_tok * (d + 2 * d_ff) * 2

    # GDN chunk states: (T / C) * B * H * hd * hd * 2B + QKV/gate activations
    n_chunks = max(1, math.ceil(t / cfg.gdn_chunk_size))
    gdn_act_bytes = (n_chunks * local_batch_size * gdn_heads * gdn_hd * gdn_hd * 2) + (n_tok * 5 * gdn_inner * 2)
    attn_act_bytes = n_tok * 4 * d * 2

    per_gdn_layer_act = gdn_act_bytes + moe_act_bytes
    per_attn_layer_act = attn_act_bytes + moe_act_bytes
    # Tiled CE avoids full [B, T, V] logits; peak ~0.35 GiB across main + MTP
    ce_peak_bytes = int(0.35 * gib)

    act_no_ckpt_gib = (n_gdn * per_gdn_layer_act + n_attn * per_attn_layer_act + ce_peak_bytes) / gib
    # With gradient checkpointing: save L layer inputs (B*T*D*2B) + 1 active recomputed layer + tiled CE
    saved_inputs_bytes = cfg.n_layers * n_tok * d * 2
    act_with_ckpt_gib = (saved_inputs_bytes + max(per_gdn_layer_act, per_attn_layer_act) + ce_peak_bytes) / gib

    peak_no_ckpt = persistent_state_gib + allgather_transient_gib + act_no_ckpt_gib + xla_workspace_gib
    peak_with_ckpt = persistent_state_gib + allgather_transient_gib + act_with_ckpt_gib + xla_workspace_gib

    # M5 Pro inference fixed state (BF16) & active read (2-bit ternary body + 4-bit LM head)
    gdn_state_bytes = n_gdn * gdn_heads * gdn_hd * gdn_hd * 2
    swa_window = cfg.sliding_window if cfg.sliding_window else t
    swa_kv_bytes = n_attn * 2 * swa_window * n_kv * attn_hd * 2
    fixed_state_mb = (gdn_state_bytes + swa_kv_bytes) / (1024 ** 2)
    body_bytes_per_param = 0.265 if cfg.use_bitlinear else 2.0
    m5_active_mb = (active_body_params * body_bytes_per_param + lm_head_params * 0.55) / (1024 ** 2) + fixed_state_mb

    return HBMReport(
        total_params=total_params,
        active_params_per_token=active_params_per_token,
        active_body_params=active_body_params,
        lm_head_params=lm_head_params,
        mtp_params=mtp_params,
        persistent_state_gib_per_chip=persistent_state_gib,
        allgather_transient_gib=allgather_transient_gib,
        activations_gib_no_ckpt=act_no_ckpt_gib,
        activations_gib_with_ckpt=act_with_ckpt_gib,
        xla_workspace_gib=xla_workspace_gib,
        peak_gib_no_ckpt=peak_no_ckpt,
        peak_gib_with_ckpt=peak_with_ckpt,
        usable_hbm_gib=usable_hbm_gib,
        fits_with_ckpt=peak_with_ckpt <= usable_hbm_gib,
        fits_without_ckpt=peak_no_ckpt <= usable_hbm_gib,
        m5_pro_single_tok_active_mb=m5_active_mb,
        m5_pro_fixed_state_mb=fixed_state_mb,
    )
