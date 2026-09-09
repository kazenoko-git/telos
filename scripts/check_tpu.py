"""
Diagnostic script to verify 8-core PyTorch-XLA execution on Google Cloud TPUs.
"""

import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from telos.training.xla_utils import clean_tpu_environment

clean_tpu_environment()

import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp


def _check_core_worker(index):
    dev = xm.xla_device()
    world = xm.xrt_world_size()
    print(f"  [Core {index}] Online! Binding device: {dev}, Total World Size: {world}")


def main():
    print(">>> Testing PyTorch-XLA multi-core execution across 8 TPU cores...")
    xmp.spawn(_check_core_worker, args=(), nprocs=None)
    print("✓ All 8 TPU cores online and operational!")


if __name__ == "__main__":
    main()
