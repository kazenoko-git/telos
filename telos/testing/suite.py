"""
Unified Testing and Verification Suite for Telos.
Tests model architectures, loss formulations, live training steps,
sampling, and checkpointing across all paradigms (AR, MDLM, UNDLM, COROSred)
and backends (MLX, PyTorch).
"""

import sys
import time
import tempfile
from pathlib import Path
import numpy as np

# MLX
try:
    import mlx.core as mx
    import mlx.nn as mx_nn
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

# PyTorch
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from telos.models import TelosTransformer, TelosConfig
from telos.models.param_counter import count_parameters, verify_with_model
from telos.diffusion.ar import ar_loss_fn_pytorch
from telos.diffusion.mdlm import (
    apply_masking_pytorch, mdlm_loss_pytorch,
    sample_beta_timesteps
)
from telos.diffusion.undlm import (
    apply_uniform_noise_pytorch, undlm_loss_pytorch
)
from telos.diffusion.corosred import (
    crsr_phase_a_loss_fn_pytorch, crsr_phase_b_loss_fn_pytorch
)
from telos.diffusion.sampler import MDLMSampler, MLXMDLMSampler
from telos.training.trainer_pytorch import UnifiedPyTorchTrainer


def test_model_contracts() -> tuple[bool, str]:
    """Verifies parameter counting and weight tying across configs."""
    try:
        cfg = TelosConfig(vocab_size=1000, d_model=128, n_layers=2, n_heads=4, max_seq_len=64, tied_embeddings=True)
        analytical = count_parameters(cfg)["total"]
        actual = verify_with_model(cfg)
        if analytical != actual:
            return False, f"Param count mismatch: analytical={analytical}, actual={actual}"

        model = TelosTransformer(cfg)
        if model.tok_embeddings.weight is not model.output_projection.weight:
            return False, "Weight tying failed in TelosTransformer"

        if MLX_AVAILABLE:
            from telos.models import MLXTelosTransformer
            mlx_model = MLXTelosTransformer(vocab_size=1000, d_model=128, n_layers=2, n_heads=4, n_kv_heads=4, tied_embeddings=True)
            dummy = mx.zeros((2, 16), dtype=mx.int32)
            out = mlx_model(dummy)
            if out.shape != (2, 16, 1000):
                return False, f"MLXTelosTransformer forward shape mismatch: {out.shape}"

        return True, "Passed (analytical matching & weight-tying verified)"
    except Exception as e:
        return False, str(e)


def test_attention_causality() -> tuple[bool, str]:
    """Verifies bidirectional attention responds to future context and causal attention does not."""
    try:
        # 1. Bidirectional Attention
        bidi_cfg = TelosConfig(vocab_size=100, d_model=64, n_layers=2, n_heads=2, max_seq_len=64, is_causal=False)
        bidi_model = TelosTransformer(bidi_cfg).eval()
        seq_a = torch.tensor([[10, 20, 30, 40, 50]], dtype=torch.long)
        seq_b = torch.tensor([[10, 20, 30, 40, 99]], dtype=torch.long)
        with torch.no_grad():
            diff_bidi = (bidi_model(seq_a)[0, 0] - bidi_model(seq_b)[0, 0]).abs().sum().item()
        if diff_bidi < 1e-4:
            return False, f"Bidirectional attention failed to attend to right-hand tokens (diff={diff_bidi})"

        # 2. Causal Attention
        causal_cfg = TelosConfig(vocab_size=100, d_model=64, n_layers=2, n_heads=2, max_seq_len=64, is_causal=True)
        causal_model = TelosTransformer(causal_cfg).eval()
        with torch.no_grad():
            diff_causal = (causal_model(seq_a)[0, 0] - causal_model(seq_b)[0, 0]).abs().sum().item()
        if diff_causal > 1e-5:
            return False, f"Causal attention leaked right-hand token context to the left (diff={diff_causal})"

        return True, "Passed (Bidirectional & Causal masking verified)"
    except Exception as e:
        return False, str(e)


def test_losses_pytorch() -> tuple[bool, str]:
    """Verifies mathematical losses for AR, MDLM, UNDLM, and COROSred on PyTorch."""
    try:
        bs, seq_len, vocab_size = 4, 32, 100
        targets = torch.randint(0, vocab_size, (bs, seq_len))
        logits = torch.randn(bs, seq_len, vocab_size)

        # AR
        loss_ar, _ = ar_loss_fn_pytorch(logits, targets)
        if not torch.isfinite(loss_ar) or loss_ar.item() <= 0:
            return False, "AR loss is not positive finite"

        # MDLM
        t_vals = torch.rand(bs, 1).clamp(min=1e-5, max=1.0)
        masked_ids, mask_pos, _ = apply_masking_pytorch(targets, t_vals, mask_token_id=1)
        loss_mdlm, _ = mdlm_loss_pytorch(logits, targets, mask_pos, t_vals)
        if not torch.isfinite(loss_mdlm) or loss_mdlm.item() <= 0:
            return False, "MDLM loss is not positive finite"

        # UNDLM
        noisy_ids, corrupt_mask, _ = apply_uniform_noise_pytorch(targets, t_vals, vocab_size)
        loss_undlm, _ = undlm_loss_pytorch(logits, targets, t_vals)
        if not torch.isfinite(loss_undlm) or loss_undlm.item() <= 0:
            return False, "UNDLM loss is not positive finite"

        # COROSred Phase A & B
        crsr_cfg = TelosConfig(vocab_size=vocab_size, d_model=64, n_layers=2, n_heads=2, max_seq_len=seq_len, use_reliability_head=True)
        crsr_model = TelosTransformer(crsr_cfg)
        loss_crsr_a, _ = crsr_phase_a_loss_fn_pytorch(crsr_model, targets, vocab_size, k_amb=5)
        loss_crsr_b, _ = crsr_phase_b_loss_fn_pytorch(crsr_model, targets, vocab_size, mask_token_id=1)
        if not torch.isfinite(loss_crsr_a) or not torch.isfinite(loss_crsr_b):
            return False, "COROSred loss is not finite"

        return True, "Passed (AR, MDLM, UNDLM, COROSred A/B losses valid)"
    except Exception as e:
        return False, str(e)


def test_losses_mlx() -> tuple[bool, str]:
    """Verifies mathematical losses for AR, MDLM, UNDLM, and COROSred on MLX."""
    if not MLX_AVAILABLE:
        return True, "Skipped (MLX not available on this platform)"
    try:
        from telos.models import MLXTelosTransformer
        from telos.diffusion.ar import ar_loss_fn_mlx
        from telos.diffusion.mdlm import apply_masking_mlx, mdlm_loss_mlx
        from telos.diffusion.undlm import apply_uniform_noise_mlx, undlm_loss_mlx
        from telos.diffusion.corosred import crsr_phase_a_loss_fn_mlx, crsr_phase_b_loss_fn_mlx

        bs, seq_len, vocab_size = 4, 32, 100
        targets = mx.random.randint(0, vocab_size, (bs, seq_len))

        # AR
        model_ar = MLXTelosTransformer(vocab_size=vocab_size, d_model=64, n_layers=2, n_heads=2, is_causal=True)
        loss_ar, _ = ar_loss_fn_mlx(model_ar, targets, vocab_size)
        mx.eval(loss_ar)
        if float(loss_ar.item()) <= 0:
            return False, "MLX AR loss is non-positive"

        # MDLM
        model_mdlm = MLXTelosTransformer(vocab_size=vocab_size, d_model=64, n_layers=2, n_heads=2, is_causal=False)
        t_vals = mx.random.uniform(0.1, 1.0, (bs, 1))
        masked_ids, mask_pos, _ = apply_masking_mlx(targets, t_vals, mask_token_id=1)
        loss_mdlm, _ = mdlm_loss_mlx(model_mdlm, masked_ids, targets, mask_pos, t_vals, vocab_size)
        mx.eval(loss_mdlm)
        if float(loss_mdlm.item()) <= 0:
            return False, "MLX MDLM loss is non-positive"

        # UNDLM
        noisy_ids, corrupt_mask, _ = apply_uniform_noise_mlx(targets, t_vals, vocab_size)
        loss_undlm, _ = undlm_loss_mlx(model_mdlm, noisy_ids, targets, t_vals, vocab_size)
        mx.eval(loss_undlm)
        if float(loss_undlm.item()) <= 0:
            return False, "MLX UNDLM loss is non-positive"

        # COROSred Phase A & B
        model_crsr = MLXTelosTransformer(vocab_size=vocab_size, d_model=64, n_layers=2, n_heads=2, is_causal=True, use_reliability_head=True)
        loss_crsr_a, _ = crsr_phase_a_loss_fn_mlx(model_crsr, targets, vocab_size, k_amb=5)
        loss_crsr_b, _ = crsr_phase_b_loss_fn_mlx(model_crsr, targets, vocab_size, mask_token_id=1)
        mx.eval(loss_crsr_a, loss_crsr_b)
        if float(loss_crsr_a.item()) <= 0 or float(loss_crsr_b.item()) <= 0:
            return False, "MLX COROSred loss is non-positive"

        return True, "Passed (MLX AR, MDLM, UNDLM, COROSred A/B losses valid)"
    except Exception as e:
        return False, str(e)


def test_live_training_steps(backend: str = "all", paradigm: str = "all") -> tuple[bool, str]:
    """Runs 2 live micro-steps of AR, MDLM, UNDLM, and COROSred across PyTorch and MLX."""
    try:
        from telos.training import UnifiedPyTorchTrainer
        dummy_cfg = {
            "model": {"vocab_size": 100, "d_model": 64, "n_layers": 2, "n_heads": 2, "seq_len": 32},
            "training": {"max_steps": 2, "batch_size": 2, "gradient_accumulation": 1, "max_lr": 1e-4, "min_lr": 1e-5, "warmup_steps": 1, "weight_decay": 0.01},
            "checkpoint": {"checkpoint_dir": "checkpoints/test_scratch", "save_every_steps": 100},
            "data": {"synthetic": True}
        }
        all_paradigms = ["ar", "mdlm", "undlm", "corosred"]
        target_paradigms = [paradigm.lower()] if paradigm.lower() != "all" else all_paradigms

        # 1. Test PyTorch paradigms
        if backend in ["all", "pytorch"]:
            for p in target_paradigms:
                m_pt = TelosTransformer(vocab_size=100, d_model=64, n_layers=2, n_heads=2, n_kv_heads=2, is_causal=(p in ["ar", "corosred"]), use_reliability_head=(p == "corosred"))
                trainer = UnifiedPyTorchTrainer(p, m_pt, dummy_cfg, device_type="cpu")
                trainer.train(resume_step=0)

                if p == "corosred":
                    dummy_cfg_b = dict(dummy_cfg)
                    dummy_cfg_b["corosred"] = {"phase": "B"}
                    m_pt_b = TelosTransformer(vocab_size=100, d_model=64, n_layers=2, n_heads=2, n_kv_heads=2, is_causal=False, use_reliability_head=True)
                    trainer_b = UnifiedPyTorchTrainer(p, m_pt_b, dummy_cfg_b, device_type="cpu")
                    trainer_b.train(resume_step=0)

        # 2. Test MLX paradigms if available
        if MLX_AVAILABLE and backend in ["all", "mlx"]:
            from telos.models import MLXTelosTransformer
            from telos.training import UnifiedMLXTrainer
            for p in target_paradigms:
                m_mlx = MLXTelosTransformer(vocab_size=100, d_model=64, n_layers=2, n_heads=2, n_kv_heads=2, is_causal=(p in ["ar", "corosred"]), use_reliability_head=(p == "corosred"))
                trainer_mlx = UnifiedMLXTrainer(p, m_mlx, dummy_cfg, eval_policy="step")
                trainer_mlx.train(resume_step=0)

                if p == "corosred":
                    dummy_cfg_b = dict(dummy_cfg)
                    dummy_cfg_b["corosred"] = {"phase": "B"}
                    m_mlx_b = MLXTelosTransformer(vocab_size=100, d_model=64, n_layers=2, n_heads=2, n_kv_heads=2, is_causal=False, use_reliability_head=True)
                    trainer_mlx_b = UnifiedMLXTrainer(p, m_mlx_b, dummy_cfg_b, eval_policy="step")
                    trainer_mlx_b.train(resume_step=0)

        return True, f"Passed ({', '.join(p.upper() for p in target_paradigms)} live steps on {backend.upper()})"
    except Exception as e:
        return False, str(e)


def test_samplers() -> tuple[bool, str]:
    """Tests MDLMSampler (PyTorch) and MLXMDLMSampler (MLX) unmasking."""
    try:
        cfg = TelosConfig(vocab_size=50, d_model=32, n_layers=2, n_heads=2, max_seq_len=16)
        pt_model = TelosTransformer(cfg)
        pt_sampler = MDLMSampler(pt_model, mask_token_id=1, num_steps=3, schedule="cosine")
        sample_pt = pt_sampler.sample(seq_len=16)
        if (sample_pt == 1).any():
            return False, "MDLMSampler left unmasked [MASK] tokens"

        if MLX_AVAILABLE:
            from telos.models import MLXTelosTransformer
            mlx_model = MLXTelosTransformer(vocab_size=50, d_model=32, n_layers=2, n_heads=2, n_kv_heads=2)
            mlx_sampler = MLXMDLMSampler(mlx_model, mask_token_id=1, num_steps=3, schedule="cosine")
            sample_mlx = mlx_sampler.sample(seq_len=16)
            if np.any(np.array(sample_mlx) == 1):
                return False, "MLXMDLMSampler left unmasked [MASK] tokens"

        return True, "Passed (PyTorch & MLX iterative unmasking samplers generated clean sequences)"
    except Exception as e:
        return False, str(e)


def run_unified_testing_suite(backend: str = "all", paradigm: str = "all", verbose: bool = True) -> bool:
    """Executes the complete Telos Unified Testing Suite."""
    backend = backend.lower() if backend else "all"
    paradigm = paradigm.lower() if paradigm else "all"

    print("=" * 76)
    print("  TELOS UNIFIED TESTING SUITE")
    print(f"  Target Backend: {backend.upper()}  |  Target Paradigm: {paradigm.upper()}")
    print("=" * 76)

    tests = []
    tests.append(("Architecture Contracts & Param Counting", test_model_contracts))
    tests.append(("Attention Causality & Directionality", test_attention_causality))

    if backend in ["all", "pytorch"]:
        tests.append(("PyTorch Loss Formulations (AR/MDLM/UNDLM/COROSred)", test_losses_pytorch))
    if backend in ["all", "mlx"]:
        tests.append(("MLX Loss Formulations (AR/MDLM/UNDLM/COROSred)", test_losses_mlx))

    tests.append((f"Live Multi-Paradigm Training Execution ({backend.upper()})", lambda: test_live_training_steps(backend=backend, paradigm=paradigm)))
    tests.append(("Iterative Diffusion Samplers (PyTorch & MLX)", test_samplers))

    all_passed = True
    start_t = time.time()

    for name, test_fn in tests:
        t0 = time.time()
        passed, msg = test_fn()
        elapsed = time.time() - t0
        status_str = "[ PASS ]" if passed else "[ FAIL ]"
        print(f"  {status_str} {name:<52} ({elapsed:.2f}s)")
        if not passed:
            print(f"           Error: {msg}")
            all_passed = False

    total_elapsed = time.time() - start_t
    print("-" * 76)
    if all_passed:
        print(f"  ALL TESTS PASSED! Total verification time: {total_elapsed:.2f}s")
    else:
        print(f"  TEST SUITE COMPLETED WITH FAILURES in {total_elapsed:.2f}s")
    print("=" * 76 + "\n")

    return all_passed


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Telos Unified Testing Suite")
    parser.add_argument("--backend", type=str, default="all", choices=["all", "mlx", "pytorch"])
    parser.add_argument("--paradigm", type=str, default="all", choices=["all", "ar", "mdlm", "undlm", "corosred"])
    args = parser.parse_args()

    success = run_unified_testing_suite(backend=args.backend, paradigm=args.paradigm)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

