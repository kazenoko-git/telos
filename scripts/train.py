"""
Unified Training Script for Telos paradigms.

Usage:
  python -m scripts.train --paradigm mdlm --backend mlx --config configs/shared/kappa_25m.yaml
  python -m scripts.train --paradigm corosred --phase A --backend pytorch --device cuda --config configs/shared/kappa_25m.yaml
"""

import argparse
import yaml
from pathlib import Path
import os
import sys

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from telos.models import TelosTransformer, MLXTelosTransformer
from telos.training import UnifiedMLXTrainer, UnifiedPyTorchTrainer

def main():
    parser = argparse.ArgumentParser(description="Telos Unified Trainer")
    parser.add_argument("--paradigm", type=str, required=True, choices=["ar", "mdlm", "undlm", "corosred"], help="Training paradigm")
    parser.add_argument("--phase", type=str, default="A", choices=["A", "B", "a", "b"], help="Phase for COROSred paradigm")
    parser.add_argument("--backend", type=str, required=True, choices=["mlx", "pytorch"], help="Hardware backend")
    parser.add_argument("--device", type=str, default="cpu", help="PyTorch device (cuda, mps, cpu, xla)")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--resume", type=int, default=0, help="Step to resume training from")
    parser.add_argument("--eval-policy", type=str, default="auto", choices=["auto", "eager", "step", "lazy"], help="Memory evaluation policy for Apple Silicon MLX")
    parser.add_argument("--benchmark", action="store_true", help="Run throughput benchmark (at most 5 minutes)")
    parser.add_argument("--benchmark-duration", type=int, default=300, help="Benchmark duration in seconds (capped at 300)")
    
    args = parser.parse_args()
    
    # Load YAML config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
        
    # Apply paradigm specific overrides from CLI
    if args.paradigm == "corosred":
        if "corosred" not in cfg:
            cfg["corosred"] = {}
        cfg["corosred"]["phase"] = args.phase.upper()
        
    # Model Setup
    print(f"Initializing {args.paradigm.upper()} model using {args.backend.upper()} backend...")
    
    # Check if we need causality
    is_causal = args.paradigm in ["ar", "corosred"]
    
    m_cfg = cfg.get("model", {})
    t_cfg = cfg.get("training", {})
    
    if args.backend == "mlx":
        model = MLXTelosTransformer(
            vocab_size=m_cfg.get("vocab_size", 8192),
            d_model=m_cfg.get("d_model", 512),
            n_layers=m_cfg.get("n_layers", 12),
            n_heads=m_cfg.get("n_heads", 16),
            n_kv_heads=m_cfg.get("n_kv_heads", None),
            is_causal=is_causal,
            use_grad_checkpoint=t_cfg.get("gradient_checkpointing", False) or m_cfg.get("use_grad_checkpoint", False)
        )
        trainer = UnifiedMLXTrainer(paradigm=args.paradigm, model=model, cfg=cfg, eval_policy=args.eval_policy)
        
    elif args.backend == "pytorch":
        model = TelosTransformer(
            vocab_size=m_cfg.get("vocab_size", 8192),
            d_model=m_cfg.get("d_model", 512),
            n_layers=m_cfg.get("n_layers", 12),
            n_heads=m_cfg.get("n_heads", 16),
            n_kv_heads=m_cfg.get("n_kv_heads", None),
            is_causal=is_causal
        )
        trainer = UnifiedPyTorchTrainer(paradigm=args.paradigm, model=model, cfg=cfg, device_type=args.device)
        
    else:
        raise ValueError(f"Unknown backend: {args.backend}")
        
    # Start training or benchmark
    try:
        # Cap benchmark duration at strictly 300 seconds (5 minutes)
        bench_dur = min(args.benchmark_duration, 300)
        trainer.train(
            resume_step=args.resume,
            benchmark=args.benchmark,
            benchmark_duration=bench_dur
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(0)

if __name__ == "__main__":
    main()
