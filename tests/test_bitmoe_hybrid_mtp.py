"""Unit and numerical parity tests for Telos v1.9.2 BitMoE + Hybrid GDN/SWA + MTP."""

import pytest
import torch

from telos.models.bitlinear import (
    BitLinear,
    bitlinear_activation_quant,
    bitlinear_weight_quant,
    pack_ternary_2bit,
    unpack_ternary_2bit,
)
from telos.models.config import TelosConfig
from telos.models.memory_audit import audit_config_memory
from telos.models.moe import FixedCapacityMoE
from telos.models.recurrent import GatedDeltaNet, _gdn_chunkwise, _gdn_step_reference
from telos.models.transformer import TelosTransformer


def test_bitlinear_ste_and_2bit_packing():
    torch.manual_seed(0)
    layer = BitLinear(64, 32, use_subln=True)
    x = torch.randn(2, 8, 64, requires_grad=True)
    y = layer(x)
    assert y.shape == (2, 8, 32)
    y.sum().backward()
    assert layer.weight.grad is not None and torch.isfinite(layer.weight.grad).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()

    # Verify quantized weights live strictly on {-gamma, 0, +gamma}
    wq = bitlinear_weight_quant(layer.weight.detach())
    gamma = layer.weight.detach().abs().mean().clamp(min=1e-5)
    trits = torch.round(wq / gamma)
    assert set(trits.unique().tolist()).issubset({-1.0, 0.0, 1.0})

    # Verify lossless 2-bit packing round-trip
    packed, scale = pack_ternary_2bit(layer.weight.detach())
    unpacked = unpack_ternary_2bit(packed, scale, layer.weight.shape[-1])
    assert torch.allclose(wq.float(), unpacked, atol=1e-5)


def test_gated_deltanet_chunkwise_matches_recurrent_and_state_rollback():
    torch.manual_seed(42)
    b, h, t, d = 2, 4, 40, 16
    q = torch.nn.functional.normalize(torch.randn(b, h, t, d), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(b, h, t, d), dim=-1)
    v = torch.randn(b, h, t, d)
    alpha = torch.sigmoid(torch.randn(b, h, t, 1) + 2.0)
    beta = torch.sigmoid(torch.randn(b, h, t, 1))

    out_ref, s_ref = _gdn_step_reference(q, k, v, alpha, beta)
    out_chk, s_chk = _gdn_chunkwise(q, k, v, alpha, beta, chunk_size=16)
    assert torch.allclose(out_ref, out_chk, atol=1e-4, rtol=1e-4)
    assert torch.allclose(s_ref, s_chk, atol=1e-4, rtol=1e-4)

    # Verify full module prefill + incremental step matches full-sequence output
    gdn = GatedDeltaNet(d_model=64, n_heads=4, head_dim=16, conv_kernel=4, chunk_size=8, use_bitlinear=True, use_subln=True)
    gdn.eval()
    x = torch.randn(1, 12, 64)
    with torch.no_grad():
        y_full, _ = gdn(x, return_state=True, use_chunkwise=True)
        y_pre, st = gdn(x[:, :9], return_state=True, use_chunkwise=True)
        # Snapshot state before speculative verify
        st_snapshot = (st[0].clone(), st[1].clone())
        y_suf, _ = gdn(x[:, 9:], state=st, return_state=True, use_chunkwise=False)
        # Caller's snapshot must remain unmodified for rollback
        assert torch.equal(st[0], st_snapshot[0])
        assert torch.equal(st[1], st_snapshot[1])
        assert torch.allclose(y_full[:, 9:], y_suf, atol=1e-4, rtol=1e-4)


def test_fixed_capacity_moe_static_shapes_and_grads():
    torch.manual_seed(7)
    moe = FixedCapacityMoE(
        d_model=64, num_experts=8, top_k=2, d_ff=128, shared_d_ff=128,
        capacity_factor=1.25, z_loss_coef=1e-3, bias_update_rate=1e-2,
        use_bitlinear=True, use_subln=True,
    )
    x = torch.randn(2, 16, 64, requires_grad=True)
    bias_before = moe.expert_bias.clone()
    y = moe(x)
    assert y.shape == x.shape
    assert moe.last_aux_loss is not None and moe.last_aux_loss.item() >= 0.0
    assert not torch.equal(moe.expert_bias, bias_before)

    loss = y.pow(2).mean() + moe.last_aux_loss
    loss.backward()
    for name, p in moe.named_parameters():
        assert p.grad is not None, f"Missing grad for {name}"
        assert torch.isfinite(p.grad).all(), f"Non-finite grad for {name}"


def test_hybrid_bitmoe_mtp_forward_backward_and_cache():
    torch.manual_seed(123)
    cfg = TelosConfig(
        vocab_size=256,
        d_model=64,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=64,
        is_causal=True,
        use_bitlinear=True,
        use_subln=True,
        hybrid_swa_every=2,
        sliding_window=8,
        gdn_n_heads=4,
        gdn_head_dim=16,
        gdn_chunk_size=8,
        ffn_type="moe_relu2",
        moe_num_experts=4,
        moe_top_k=2,
        moe_d_ff=128,
        moe_shared_d_ff=128,
        mtp_num_heads=2,
        mtp_d_ff=128,
        mtp_loss_weight=0.3,
    )
    model = TelosTransformer(cfg)
    assert [layer.layer_type for layer in model.layers] == ["gdn", "swa", "gdn", "swa"]

    tokens = torch.randint(0, cfg.vocab_size, (2, 20))
    main_losses, mtp_loss = model(
        tokens, mask_override=True, next_token_only=True,
        loss_targets=tokens[:, 1:], vocab_chunk_size=64, vocab_chunk_axis="token",
        return_mtp_loss=True,
    )
    aux_loss = model.collect_aux_loss()
    total = main_losses.mean() + mtp_loss + aux_loss
    total.backward()
    assert torch.isfinite(total)

    # Verify bounded SWA KV cache + GDN recurrent cache + MTP drafting
    model.eval()
    with torch.no_grad():
        h_pre, cache = model.hidden_states_with_cache(tokens[:1, :14], mask_override=True)
        # SWA layers (indices 1, 3) must cap KV length at sliding_window=8
        assert cache[1][0].shape[2] == 8
        assert cache[3][0].shape[2] == 8
        h_step, cache2 = model.hidden_states_with_cache(
            tokens[:1, 14:15], kv_cache=cache, mask_override=True, position_offset=14
        )
        assert cache2[1][0].shape[2] == 8
        draft = model.mtp.draft_tokens(h_step, tokens[:1, 14:15], model.tok_embeddings, model.output_projection.weight)
        assert draft.shape == (1, 2)


def test_bitmoe_2_25b_tpu_v5e8_memory_audit():
    cfg = TelosConfig(
        vocab_size=32768,
        d_model=1024,
        n_layers=20,
        n_heads=16,
        n_kv_heads=4,
        max_seq_len=2048,
        tied_embeddings=True,
        is_causal=True,
        use_grad_checkpoint=True,
        use_bitlinear=True,
        use_subln=True,
        hybrid_swa_every=4,
        sliding_window=512,
        gdn_n_heads=8,
        gdn_head_dim=128,
        gdn_chunk_size=64,
        ffn_type="moe_relu2",
        moe_num_experts=24,
        moe_top_k=2,
        moe_d_ff=2048,
        moe_shared_d_ff=2048,
        mtp_num_heads=3,
        mtp_d_ff=2048,
    )
    # Replicated (no FSDP) must NOT fit on 16 GiB v5e
    rep = audit_config_memory(cfg, num_chips=8, fsdp_sharded=False, local_batch_size=4)
    assert not rep.fits_with_ckpt

    # FSDP sharded + grad checkpointing MUST fit with comfortable margin
    fsdp = audit_config_memory(cfg, num_chips=8, fsdp_sharded=True, local_batch_size=4)
    assert 2.2e9 <= fsdp.total_params <= 2.35e9
    assert 350e6 <= fsdp.active_params_per_token <= 410e6
    assert fsdp.fits_with_ckpt
    assert fsdp.peak_gib_with_ckpt < 12.0
    assert fsdp.m5_pro_single_tok_active_mb < 130.0


def test_trainer_ar_bitmoe_mtp_smoke_step():
    from telos.training.trainer_pytorch import UnifiedPyTorchTrainer

    cfg_dict = {
        "paradigm": "ar",
        "model": {
            "vocab_size": 128,
            "d_model": 64,
            "n_layers": 2,
            "n_heads": 4,
            "n_kv_heads": 2,
            "max_seq_len": 32,
            "is_causal": True,
            "use_bitlinear": True,
            "use_subln": True,
            "hybrid_swa_every": 2,
            "sliding_window": 8,
            "gdn_n_heads": 4,
            "gdn_head_dim": 16,
            "gdn_chunk_size": 8,
            "ffn_type": "moe_relu2",
            "moe_num_experts": 4,
            "moe_top_k": 2,
            "moe_d_ff": 128,
            "moe_shared_d_ff": 128,
            "mtp_num_heads": 2,
            "mtp_d_ff": 128,
            "mtp_loss_weight": 0.3,
        },
        "training": {
            "precision": "fp32",
            "fused_vocab_loss": True,
            "fused_vocab_chunk_axis": "token",
            "fused_vocab_chunk_size": 16,
            "batch_size": 2,
            "gradient_accumulation": 1,
            "max_steps": 2,
            "warmup_steps": 1,
            "max_lr": 1e-3,
            "min_lr": 1e-4,
            "seed": 42,
        },
        "data": {"synthetic": True},
    }
    model = TelosTransformer(**cfg_dict["model"])
    trainer = UnifiedPyTorchTrainer("ar", model, cfg_dict, device_type="cpu")
    batch = torch.randint(4, 128, (2, 32))
    loss, metrics = trainer._execute_microbatch(batch, compute_metrics=True)
    assert "mtp_loss" in metrics and "aux_loss" in metrics
    loss.backward()
    assert torch.isfinite(loss)


def _install_mock_spmd_xla(monkeypatch, *, is_spmd: bool = True, world_size: int = 8):
    import sys
    import types
    from telos.training import xla_utils

    sharding_calls: list[tuple[int, tuple[int, ...], tuple]] = []

    class FakeMesh:
        def __init__(self, device_ids, mesh_shape, axis_names):
            self._size = len(device_ids)
            self.mesh_shape = mesh_shape
            self.axis_names = axis_names

        def size(self):
            return self._size

    def mark_sharding(tensor, mesh, spec):
        sharding_calls.append((id(tensor), tuple(tensor.shape), tuple(spec)))
        return tensor

    fake_xm = types.SimpleNamespace(
        xla_device=lambda: torch.device("cpu"),
        set_rng_state=lambda *args: None,
        get_rng_state=lambda *args: 0,
        reduce_gradients=lambda opt: None,
        mark_step=lambda: None,
        rendezvous=lambda tag: None,
    )
    fake_xr = types.SimpleNamespace(
        is_spmd=lambda: is_spmd,
        global_runtime_device_count=lambda: world_size,
        global_ordinal=lambda: 0,
        world_size=lambda: world_size,
        process_index=lambda: 0,
    )
    fake_xs = types.SimpleNamespace(Mesh=FakeMesh, mark_sharding=mark_sharding)

    fake_xla = types.ModuleType("torch_xla")
    fake_core = types.ModuleType("torch_xla.core")
    fake_dist = types.ModuleType("torch_xla.distributed")
    fake_core.xla_model = fake_xm
    fake_dist.spmd = fake_xs
    fake_xla.core = fake_core
    fake_xla.runtime = fake_xr
    fake_xla.distributed = fake_dist

    for name, mod in {
        "torch_xla": fake_xla,
        "torch_xla.core": fake_core,
        "torch_xla.core.xla_model": fake_xm,
        "torch_xla.runtime": fake_xr,
        "torch_xla.distributed": fake_dist,
        "torch_xla.distributed.spmd": fake_xs,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)

    monkeypatch.setattr(xla_utils, "_CACHED_XLA_DEVICE", None)
    monkeypatch.setattr(xla_utils, "_CACHED_XLA_WORLD_SIZE", None)
    monkeypatch.setattr(xla_utils, "_CACHED_IS_MASTER", None)
    monkeypatch.setattr(xla_utils, "_CACHED_SPMD_MESH", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    return sharding_calls


def _make_spmd_bitmoe_cfg(tmp_path, *, batch_size: int = 8, max_steps: int = 2):
    return {
        "paradigm": "ar",
        "model": {
            "vocab_size": 128,
            "d_model": 128,
            "n_layers": 2,
            "n_heads": 4,
            "n_kv_heads": 2,
            "max_seq_len": 32,
            "is_causal": True,
            "use_bitlinear": True,
            "use_subln": True,
            "hybrid_swa_every": 2,
            "sliding_window": 8,
            "gdn_n_heads": 4,
            "gdn_head_dim": 32,
            "gdn_chunk_size": 8,
            "ffn_type": "moe_relu2",
            "moe_num_experts": 8,
            "moe_top_k": 2,
            "moe_d_ff": 128,
            "moe_shared_d_ff": 128,
            "mtp_num_heads": 2,
            "mtp_d_ff": 128,
            "mtp_loss_weight": 0.3,
        },
        "training": {
            "precision": "fp32",
            "xla_fsdp": True,
            "xla_dynamic_optimizer": True,
            "gradient_checkpointing": False,
            "fused_vocab_loss": True,
            "fused_vocab_chunk_axis": "token",
            "fused_vocab_chunk_size": 16,
            "batch_size": batch_size,
            "gradient_accumulation": 1,
            "max_steps": max_steps,
            "warmup_steps": 1,
            "max_lr": 1e-3,
            "min_lr": 1e-4,
            "seed": 42,
        },
        "checkpoint": {
            "dir": str(tmp_path / "ckpts"),
            "save_every_steps": 0,
            "optimizer_state_dtype": "bf16",
        },
        "data": {"synthetic": True},
    }


def test_xla_fsdp_fails_fast_without_spmd_or_bad_batch_size(monkeypatch, tmp_path):
    from telos.training.trainer_pytorch import UnifiedPyTorchTrainer

    _install_mock_spmd_xla(monkeypatch, is_spmd=False, world_size=8)
    cfg = _make_spmd_bitmoe_cfg(tmp_path)
    with pytest.raises(RuntimeError, match="requires PyTorch/XLA SPMD mode"):
        UnifiedPyTorchTrainer("ar", TelosTransformer(**cfg["model"]), cfg, device_type="xla")

    _install_mock_spmd_xla(monkeypatch, is_spmd=True, world_size=8)
    bad_cfg = _make_spmd_bitmoe_cfg(tmp_path, batch_size=6)
    with pytest.raises(RuntimeError, match="must be divisible by TPU world_size"):
        UnifiedPyTorchTrainer("ar", TelosTransformer(**bad_cfg["model"]), bad_cfg, device_type="xla")


def test_xla_spmd_fsdp_checkpoint_save_reload_reshards_and_resumes(monkeypatch, tmp_path):
    from telos.training.trainer_pytorch import UnifiedPyTorchTrainer

    sharding_calls = _install_mock_spmd_xla(monkeypatch, is_spmd=True, world_size=8)
    cfg1 = _make_spmd_bitmoe_cfg(tmp_path, batch_size=8, max_steps=3)
    cfg1["checkpoint"]["save_every_steps"] = 2
    model1 = TelosTransformer(**cfg1["model"])
    trainer1 = UnifiedPyTorchTrainer("ar", model1, cfg1, device_type="xla")

    # Verify 3D MoE expert tensors [8, 128, 128] were sharded on expert dim 0
    moe_sharded = [c for c in sharding_calls if c[1] == (8, 128, 128) and c[2] == ("data", None, None)]
    assert len(moe_sharded) == 4  # 2 layers * (w_up, w_down)

    sharding_calls.clear()
    trainer1.train()

    # Verify XLAAdamW explicitly sharded optimizer state tensors during step 1
    opt_moe_sharded = [c for c in sharding_calls if c[1] == (8, 128, 128) and c[2] == ("data", None, None)]
    assert len(opt_moe_sharded) == 8  # 4 MoE tensors * (exp_avg, exp_avg_sq)

    ckpt_path = tmp_path / "ckpts" / "checkpoint_step_2.pt"
    assert ckpt_path.exists()
    raw_saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert raw_saved["global_step"] == 2
    assert raw_saved["next_row"] == 16  # 2 steps * global batch 8
    # Verify optimizer moments were compressed to bf16 on disk
    saved_opt_states = raw_saved["optimizer_state_dict"]["state"]
    assert any(
        isinstance(v, torch.Tensor) and v.dtype == torch.bfloat16
        for st in saved_opt_states.values()
        for k, v in st.items()
        if k != "step"
    )

    # Fresh trainer with identical run config resumes from step 2 -> step 3
    # and must re-shard both model params AND optimizer states after load_state_dict
    cfg2 = _make_spmd_bitmoe_cfg(tmp_path, batch_size=8, max_steps=3)
    cfg2["checkpoint"]["save_every_steps"] = 2
    model2 = TelosTransformer(**cfg2["model"])
    trainer2 = UnifiedPyTorchTrainer("ar", model2, cfg2, device_type="xla")
    sharding_calls.clear()
    trainer2.load_checkpoint(ckpt_path)

    # Post-load re-sharding must cover both 3D MoE params (4) and their exp_avg/exp_avg_sq (8) = 12
    re_sharded_moe = [c for c in sharding_calls if c[1] == (8, 128, 128) and c[2] == ("data", None, None)]
    assert len(re_sharded_moe) == 12
    # Optimizer moments must be upcast back to float32 to match FP32 master params
    for param, st in trainer2.optimizer.state.items():
        if "exp_avg" in st:
            assert st["exp_avg"].dtype == param.dtype == torch.float32
            assert st["exp_avg_sq"].dtype == param.dtype == torch.float32

    trainer2.train(resume_step=2)
    assert trainer2.global_step == 3
    assert trainer2._next_row == 24  # 3 steps * global batch 8


def test_bitmoe_bf16_autocast_multi_step_stability():
    """Regression test for Canary 1 step-3 NaN under BF16 autocast in GatedDeltaNet."""
    torch.manual_seed(42)
    cfg = TelosConfig(
        vocab_size=512,
        d_model=128,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=128,
        is_causal=True,
        use_bitlinear=True,
        use_subln=True,
        hybrid_swa_every=4,
        sliding_window=32,
        gdn_n_heads=4,
        gdn_head_dim=32,
        gdn_chunk_size=64,
        ffn_type="moe_relu2",
        moe_num_experts=8,
        moe_top_k=2,
        moe_d_ff=256,
        moe_shared_d_ff=256,
        mtp_num_heads=2,
        mtp_d_ff=256,
        mtp_loss_weight=0.3,
    )
    model = TelosTransformer(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, betas=(0.9, 0.95), weight_decay=0.01)
    for step in range(1, 5):
        x = torch.randint(4, 512, (2, 128))
        opt.zero_grad()
        with torch.amp.autocast("cpu", dtype=torch.bfloat16):
            tl, mtp = model(
                x, mask_override=True, next_token_only=True,
                loss_targets=x[:, 1:], vocab_chunk_size=64, vocab_chunk_axis="token",
                return_mtp_loss=True,
            )
            aux = model.collect_aux_loss()
            loss = tl.mean() + mtp + aux
        assert torch.isfinite(loss), f"Non-finite loss at BF16 step {step}"
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"Non-finite grad in {name} at BF16 step {step}"
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
