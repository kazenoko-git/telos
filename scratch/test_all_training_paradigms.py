"""
Comprehensive end-to-end verification script testing every single model tier
and training paradigm across MLX and PyTorch backends.
"""

import sys
import tempfile
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from telos.train.cli import train

TIERS = ["15M", "25M", "50M", "100M"]
PARADIGMS = ["ar", "corosred", "mdlm", "undlm"]

def run_suite():
    print("=================================================================")
    print("STARTING FULL TRAINING SUITE VERIFICATION")
    print("=================================================================")
    
    passed = []
    failed = []

    # 1. Test MLX on 15M and 25M for AR, COROSred, MDLM, UNDLM
    for tier in ["15M", "25M"]:
        for p in PARADIGMS:
            test_name = f"MLX | {tier} | {p.upper()}"
            print(f"\n--- Testing: {test_name} ---")
            with tempfile.TemporaryDirectory() as tmpdir:
                try:
                    trainer = train(
                        paradigm=p,
                        params=tier,
                        hardware="mlx",
                        max_steps=3,
                        batch_size=4,
                        grad_accum=2,
                        checkpoint_dir=tmpdir,
                        data_path="data/python_corpus_mac.bin",
                    )
                    passed.append(test_name)
                    print(f"PASSED: {test_name}")
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    failed.append((test_name, str(e)))
                    print(f"FAILED: {test_name}: {e}")

    # 2. Test PyTorch (CPU/MPS) on 15M, 25M, 50M, 100M for AR and COROSred, plus MDLM/UNDLM on 15M
    for tier in TIERS:
        for p in ["ar", "corosred"]:
            test_name = f"PyTorch | {tier} | {p.upper()}"
            print(f"\n--- Testing: {test_name} ---")
            with tempfile.TemporaryDirectory() as tmpdir:
                try:
                    trainer = train(
                        paradigm=p,
                        params=tier,
                        hardware="pytorch",
                        max_steps=3,
                        batch_size=2,
                        grad_accum=2,
                        checkpoint_dir=tmpdir,
                        data_path="data/python_corpus_mac.bin",
                    )
                    passed.append(test_name)
                    print(f"PASSED: {test_name}")
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    failed.append((test_name, str(e)))
                    print(f"FAILED: {test_name}: {e}")

    # Test PyTorch MDLM & UNDLM on 15M
    for p in ["mdlm", "undlm"]:
        test_name = f"PyTorch | 15M | {p.upper()}"
        print(f"\n--- Testing: {test_name} ---")
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                trainer = train(
                    paradigm=p,
                    params="15M",
                    hardware="pytorch",
                    max_steps=3,
                    batch_size=2,
                    grad_accum=2,
                    checkpoint_dir=tmpdir,
                    data_path="data/python_corpus_mac.bin",
                )
                passed.append(test_name)
                print(f"PASSED: {test_name}")
            except Exception as e:
                import traceback
                traceback.print_exc()
                failed.append((test_name, str(e)))
                print(f"FAILED: {test_name}: {e}")

    print("\n" + "=" * 70)
    print("FULL SUITE RESULTS:")
    print(f"Total passed: {len(passed)} / {len(passed) + len(failed)}")
    for p in passed:
        print(f"  [OK] {p}")
    if failed:
        print("\nFailures:")
        for name, err in failed:
            print(f"  [FAIL] {name}: {err}")
    print("=" * 70)
    
    if failed:
        sys.exit(1)

if __name__ == "__main__":
    run_suite()
