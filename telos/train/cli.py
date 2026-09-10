import os
import sys
import argparse
from pathlib import Path

# Allocator hygiene: Set expandable_segments BEFORE torch is imported
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Thread hygiene: Restrict thread pools BEFORE torch / numpy load to stop multi-core thread storms
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

from telos.configs import build_config


def _mp_train_worker(index, kwargs):
    """Worker entrypoint executed on each spawned TPU core."""
    import torch
    # Crucial for multi-core TPU VM: Restrict intra-op thread count to 1 per worker
    # to stop 8 spawned processes from generating 64+ competing OpenMP spin-lock threads
    # that pin CPU utilization at 650-800%.
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except Exception:
            pass

    try:
        import torch_xla
        if not hasattr(torch, "xla"):
            torch.xla = torch_xla
    except ImportError:
        pass
    from telos.training import xla_utils
    xla_utils.clean_tpu_environment()
    # Reset singleton references in forked child process so xm.xla_device() binds to assigned core
    xla_utils._CACHED_XLA_DEVICE = None
    xla_utils._CACHED_XLA_WORLD_SIZE = None
    xla_utils._CACHED_IS_MASTER = None
    train(**kwargs)



def _load_checkpoint_into_mlx(model, ckpt_path: str | Path):
    """Loads weights from either a safetensors file or a PyTorch (.pt/.bin) checkpoint into an MLX model."""
    import mlx.core as mx
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        print(f"  [Init Warning] Checkpoint not found at {ckpt_path}, starting from scratch.")
        return

    if ckpt_path.suffix in (".safetensors", ".npz"):
        print(f"  [Init] Loading native MLX weights from {ckpt_path}...")
        model.load_weights(str(ckpt_path), strict=False)
        return

    # PyTorch (.pt / .bin) checkpoint translation
    import torch
    print(f"  [Init] Loading PyTorch weights into MLX model from {ckpt_path}...")
    ckpt_state = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt_state.get("model_state_dict", ckpt_state)

    mlx_weights = {}
    qkv = {}
    tied_embeddings = getattr(model, "tied_embeddings", True)

    for k, v in sd.items():
        k = k.removeprefix("module.")
        if not isinstance(v, torch.Tensor):
            continue
        arr = mx.array(v.float().numpy()).astype(mx.bfloat16)

        if k == "tok_embeddings.weight":
            mlx_weights["emb.weight"] = arr
        elif k == "final_norm.weight":
            mlx_weights["norm.weight"] = arr
        elif k == "output_projection.weight":
            if not tied_embeddings:
                mlx_weights["head.weight"] = arr
        elif ".attn_norm.weight" in k:
            layer_idx = k.split(".")[1]
            mlx_weights[f"layers.{layer_idx}.norm1.weight"] = arr
        elif ".mlp_norm.weight" in k:
            layer_idx = k.split(".")[1]
            mlx_weights[f"layers.{layer_idx}.norm2.weight"] = arr
        elif ".attn.out_proj.weight" in k:
            layer_idx = k.split(".")[1]
            mlx_weights[f"layers.{layer_idx}.out.weight"] = arr
        elif ".attn.q_proj.weight" in k:
            layer_idx = k.split(".")[1]
            qkv.setdefault(layer_idx, {})["q"] = arr
        elif ".attn.k_proj.weight" in k:
            layer_idx = k.split(".")[1]
            qkv.setdefault(layer_idx, {})["k"] = arr
        elif ".attn.v_proj.weight" in k:
            layer_idx = k.split(".")[1]
            qkv.setdefault(layer_idx, {})["v"] = arr
        elif ".mlp.w1.weight" in k:
            layer_idx = k.split(".")[1]
            mlx_weights[f"layers.{layer_idx}.mlp.w1.weight"] = arr
        elif ".mlp.v.weight" in k:
            layer_idx = k.split(".")[1]
            mlx_weights[f"layers.{layer_idx}.mlp.w2.weight"] = arr
        elif ".mlp.w2.weight" in k:
            layer_idx = k.split(".")[1]
            mlx_weights[f"layers.{layer_idx}.mlp.w3.weight"] = arr
        elif "reliability_head." in k:
            parts = k.split(".")
            mlx_weights[f"reliability_head.layers.{parts[1]}.{parts[2]}"] = arr

    for layer_idx, p in qkv.items():
        if "q" in p and "k" in p and "v" in p:
            mlx_weights[f"layers.{layer_idx}.qkv_proj.weight"] = mx.concatenate([p["q"], p["k"], p["v"]], axis=0)

    model.load_weights(list(mlx_weights.items()), strict=False)
    print(f"  [Init] Successfully loaded {len(mlx_weights)} parameter tensors into MLX model.")


def train(
    paradigm: str = "mdlm",
    phase: str | None = None,
    params: str | int | None = "12M",
    tokens: str | int | None = None,
    effective_batch: int | str | None = None,
    batch_size: int | None = None,
    grad_accum: int | None = None,
    seq_len: int = 512,
    tokenizer: str | None = None,
    vocab_size: int | None = None,
    hardware: str | None = "auto",
    devices: int | str | None = "auto",
    max_steps: int | None = None,
    max_lr: float | None = None,
    min_lr: float | None = None,
    warmup_steps: int | None = None,
    weight_decay: float | None = None,
    checkpoint_dir: str | None = None,
    save_every: int | None = None,
    config_path: str | Path | None = None,
    data_path: str | Path | None = None,
    synthetic: bool = False,
    resume_step: int = 0,
    eval_policy: str = "auto",
    benchmark: bool = False,
    benchmark_duration: float = 300.0,
    self_condition: bool = True,
    self_cond_prob: float = 0.5,
    init_checkpoint: str | Path | None = None,
    compile: bool | None = None,
    **kwargs
):
    """
    Programmatic entrypoint to train any Télos model.
    """
    cfg = build_config(
        paradigm=paradigm,
        phase=phase,
        params=params,
        tokens=tokens,
        effective_batch=effective_batch,
        batch_size=batch_size,
        grad_accum=grad_accum,
        seq_len=seq_len,
        tokenizer=tokenizer,
        vocab_size=vocab_size,
        hardware=hardware,
        devices=devices,
        max_steps=max_steps,
        max_lr=max_lr,
        min_lr=min_lr,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        checkpoint_dir=checkpoint_dir,
        save_every=save_every,
        config_path=config_path,
        data_path=data_path,
        synthetic=synthetic,
        self_condition=self_condition,
        self_cond_prob=self_cond_prob,
        compile=compile,
        **kwargs
    )

    backend = cfg["_backend"]
    device = cfg["_device"]
    dev_count = cfg.get("_device_count", 1)
    is_causal = paradigm.lower() in ["ar", "corosred"]

    m_cfg = cfg["model"]
    t_cfg = cfg["training"]

    if not kwargs.get("_is_spawned", False):

        print("=" * 76)
        phase_str = f" (Phase {phase.upper()})" if phase else (" (Unified Continuous)" if paradigm.lower() == "corosred" else "")
        print(f"  TÉLOS UNIFIED TRAINER  |  Paradigm: {paradigm.upper()}{phase_str}")
        print(f"  Hardware Backend:     {backend.upper()} ({device})")
        if "_resolved_params" in cfg:
            print(f"  Target Parameters:    {cfg['_resolved_params']:,} (~{params})")
        dev_multiplier = dev_count if dev_count > 1 else 1
        eff_batch = t_cfg["batch_size"] * t_cfg["gradient_accumulation"] * dev_multiplier
        dev_str = f", devices={dev_count}" if dev_count > 1 else ""
        print(f"  Batch Config:         batch_size={t_cfg['batch_size']}, grad_accum={t_cfg['gradient_accumulation']}{dev_str} (effective={eff_batch} seqs / {eff_batch * m_cfg['seq_len']:,} tok)")
        print(f"  Training Steps:       {t_cfg['max_steps']:,} steps | LR: {t_cfg['max_lr']:.2e} -> {t_cfg['min_lr']:.2e} (warmup={t_cfg['warmup_steps']})")
        print(f"  Checkpoint Dir:       {cfg['checkpoint']['checkpoint_dir']} (every {cfg['checkpoint']['save_every_steps']} steps)")
    if backend == "pytorch" and device == "xla" and dev_count > 1 and not kwargs.get("_is_spawned", False):
        try:
            from telos.training.xla_utils import clean_tpu_environment
            clean_tpu_environment()
            import torch_xla.distributed.xla_multiprocessing as xmp
            print(f"  [Hardware] Launching multi-core PyTorch-XLA execution across {dev_count} TPU cores via xmp.spawn...")
            spawn_args = dict(
                    paradigm=paradigm,
                    phase=phase,
                    params=params,
                    tokens=None,  # Pre-resolved cluster max_steps takes precedence in workers
                    effective_batch=None,
                    batch_size=t_cfg["batch_size"],
                    grad_accum=t_cfg["gradient_accumulation"],
                    seq_len=seq_len,
                    tokenizer=tokenizer,
                    vocab_size=vocab_size,
                    hardware=hardware,
                    devices=dev_count,
                    max_steps=t_cfg["max_steps"],
                    max_lr=t_cfg["max_lr"],
                    min_lr=t_cfg["min_lr"],
                    warmup_steps=t_cfg["warmup_steps"],
                    weight_decay=weight_decay,
                    checkpoint_dir=checkpoint_dir,
                    save_every=save_every,
                    config_path=config_path,
                    data_path=data_path,
                    synthetic=synthetic,
                    resume_step=resume_step,
                    eval_policy=eval_policy,
                    benchmark=benchmark,
                    benchmark_duration=benchmark_duration,
                    self_condition=self_condition,
                    self_cond_prob=self_cond_prob,
                    init_checkpoint=init_checkpoint,
                    compile=compile,
                    _is_spawned=True,
                )
            xmp.spawn(_mp_train_worker, args=(spawn_args,), nprocs=None)
            return None

        except Exception as e:
            print(f"  [Notice] Multi-core xmp.spawn skipped ({e}). Proceeding on single core.")

    if backend == "mlx":
        # Defer MLX imports so non-Apple-Silicon environments don't resolve MLX
        from telos.models import MLXTelosTransformer
        from telos.training import UnifiedMLXTrainer

        precision = t_cfg.get("precision", "bfloat16")
        model = MLXTelosTransformer(
            vocab_size=m_cfg.get("vocab_size", 8192),
            d_model=m_cfg.get("d_model", 512),
            n_layers=m_cfg.get("n_layers", 12),
            n_heads=m_cfg.get("n_heads", 16),
            n_kv_heads=m_cfg.get("n_kv_heads", None),
            is_causal=is_causal,
            use_reliability_head=bool(m_cfg.get("use_reliability_head", paradigm.lower() == "corosred")),
            use_grad_checkpoint=t_cfg.get("gradient_checkpointing", False) or m_cfg.get("use_grad_checkpoint", False),
            precision=precision
        )
        # Auto-detect or load initial checkpoint (e.g. chaining Phase A weights into Phase B or Phase C)
        init_ckpt_path = init_checkpoint
        is_unified = cfg.get("corosred", {}).get("unified", False)
        if init_ckpt_path is None and paradigm.lower() == "corosred" and not is_unified and str(phase).upper() in ["B", "C"]:
            cand = Path(cfg["checkpoint"]["checkpoint_dir"]).parent / "phase_a" / "checkpoint_final.pt"
            if cand.exists():
                init_ckpt_path = str(cand)

        if init_ckpt_path:
            ckpt_file = Path(init_ckpt_path)
            if not ckpt_file.exists():
                raise FileNotFoundError(
                    f"Initial checkpoint file '{init_ckpt_path}' does not exist! "
                    f"Aborting to avoid training from uninitialized random weights."
                )
            _load_checkpoint_into_mlx(model, init_ckpt_path)

        trainer = UnifiedMLXTrainer(paradigm=paradigm, model=model, cfg=cfg, eval_policy=eval_policy)
    else:
        from telos.models import TelosTransformer
        from telos.training import UnifiedPyTorchTrainer

        model = TelosTransformer(
            vocab_size=m_cfg.get("vocab_size", 8192),
            d_model=m_cfg.get("d_model", 512),
            n_layers=m_cfg.get("n_layers", 12),
            n_heads=m_cfg.get("n_heads", 16),
            n_kv_heads=m_cfg.get("n_kv_heads", None),
            is_causal=is_causal,
            use_reliability_head=(paradigm.lower() == "corosred"),
            use_grad_checkpoint=t_cfg.get("gradient_checkpointing", False) or m_cfg.get("use_grad_checkpoint", False) or (backend == "pytorch" and device in ["xla", "cuda"])
        )


        # Auto-detect or load initial checkpoint (e.g. chaining Phase A weights into Phase B or Phase C)
        init_ckpt_path = init_checkpoint
        is_unified = cfg.get("corosred", {}).get("unified", False)
        if init_ckpt_path is None and paradigm.lower() == "corosred" and not is_unified and str(phase).upper() in ["B", "C"]:
            cand = Path(cfg["checkpoint"]["checkpoint_dir"]).parent / "phase_a" / "checkpoint_final.pt"
            if cand.exists():
                init_ckpt_path = str(cand)

        if init_ckpt_path:
            ckpt_file = Path(init_ckpt_path)
            if not ckpt_file.exists():
                raise FileNotFoundError(
                    f"Initial checkpoint file '{init_ckpt_path}' does not exist! "
                    f"Aborting to avoid training from uninitialized random weights."
                )
            import torch
            print(f"  [Init] Loading model weights from {init_ckpt_path}...")
            ckpt_state = torch.load(init_ckpt_path, map_location="cpu")
            sd = ckpt_state.get("model_state_dict", ckpt_state)
            sd = {k.removeprefix("module."): v for k, v in sd.items()}
            model.load_state_dict(sd, strict=False)
            print(f"  [Init] Successfully loaded weights from {init_ckpt_path}.")

        trainer = UnifiedPyTorchTrainer(paradigm=paradigm, model=model, cfg=cfg, device_type=device)

    # Execute training or benchmark
    bench_dur = min(float(benchmark_duration), 300.0)
    trainer.train(
        resume_step=resume_step,
        benchmark=benchmark,
        benchmark_duration=bench_dur
    )
    return trainer


def main():
    parser = argparse.ArgumentParser(description="Télos Unified Dimensional Trainer (Zero Config)")
    
    # 6 Fundamental Dimensions
    parser.add_argument("--paradigm", type=str, default="mdlm", choices=["ar", "mdlm", "undlm", "corosred", "custom"], help="Training paradigm")
    parser.add_argument("--phase", type=str, default=None, help="Phase for legacy COROSred paradigm (A, B, C). If omitted, COROSred runs in unified continuous mode.")
    parser.add_argument("--legacy-phases", action="store_true", help="Force legacy sequential phase execution (A -> B -> C) instead of unified continuous loop")
    parser.add_argument("--alpha-max", type=float, default=0.85, help="Initial causal weight alpha_max (default: 0.85)")
    parser.add_argument("--alpha-min", type=float, default=0.20, help="Permanent floor for causal weight alpha_min (default: 0.20)")
    parser.add_argument("--beta-min", type=float, default=0.15, help="Initial infilling weight beta_min (default: 0.15)")
    parser.add_argument("--beta-max", type=float, default=0.70, help="Peak infilling weight beta_max (default: 0.70)")
    parser.add_argument("--gamma-max", type=float, default=0.10, help="Peak reliability head weight gamma_max (default: 0.10)")
    parser.add_argument("--hold-frac", type=float, default=0.20, help="Fraction of steps to hold alpha at alpha_max (default: 0.20)")
    parser.add_argument("--decay-power", type=float, default=2.5, help="Polynomial exponent p for hold-then-decay schedule (default: 2.5)")
    parser.add_argument("--acc-gate", type=float, default=0.65, help="LRH classification accuracy threshold to activate gamma (default: 0.65)")
    parser.add_argument("--causal-ratio", type=float, default=0.75, help="Fraction of microbatch sequences dedicated to causal pass (default: 0.75)")
    parser.add_argument("--routing-cache-steps", type=int, default=50, help="Steps between pre-computed routing mask refreshes (default: 50)")
    parser.add_argument("--adaptive-rebalance", action="store_true", help="Enable trust-region dynamic loss rebalancing")
    parser.add_argument("--params", type=str, default="12M", help="Target parameter budget (e.g. 12M, 25M, 50M, 100M, 500M)")
    parser.add_argument("--tokens", type=str, default=None, help="Target total training tokens (e.g. 2.5B, 300M, 50M)")
    parser.add_argument("--effective-batch", type=str, default=None, help="Target effective batch size in sequences or tokens (e.g. 32, 64, 32k)")
    parser.add_argument("--batch-size", type=int, default=None, help="Microbatch size override")
    parser.add_argument("--grad-accum", type=int, default=None, help="Gradient accumulation steps override")
    parser.add_argument("--tokenizer", type=str, default=None, help="Path to custom BPE tokenizer JSON or HF model")
    parser.add_argument("--vocab-size", type=int, default=None, help="Vocabulary size override")
    parser.add_argument("--hardware", type=str, default="auto", choices=["auto", "mlx", "cuda", "mps", "xla", "cpu"], help="Hardware backend")
    parser.add_argument("--devices", type=str, default="auto", help="Hardware device count (e.g. 1, 4, 8, auto)")

    # Training Dynamics & Checkpoints
    parser.add_argument("--seq-len", type=int, default=512, help="Context sequence length")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum training steps override")
    parser.add_argument("--max-lr", type=float, default=None, help="Peak learning rate override")
    parser.add_argument("--min-lr", type=float, default=None, help="Floor learning rate override")
    parser.add_argument("--warmup-steps", type=int, default=None, help="Warmup steps override")
    parser.add_argument("--weight-decay", type=float, default=None, help="Weight decay regularization override")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Directory to save model checkpoints")
    parser.add_argument("--save-every", type=int, default=None, help="Save checkpoint cadence (steps)")
    parser.add_argument("--resume", type=int, default=0, help="Step to resume training from")
    parser.add_argument("--eval-policy", type=str, default="auto", choices=["auto", "eager", "step", "lazy"], help="Memory evaluation policy for Apple Silicon MLX")

    # Data & Config Bypass
    parser.add_argument("--data", type=str, default=None, help="Path to pre-tokenized binary dataset (.bin)")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic token stream without loading disk dataset")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config file bypass")

    # Benchmark options
    parser.add_argument("--benchmark", action="store_true", help="Run throughput benchmark (at most 5 minutes)")
    parser.add_argument("--benchmark-duration", type=int, default=300, help="Benchmark duration in seconds (capped at 300)")

    # COROSred Self-Conditioning options
    parser.add_argument("--self-condition", action=argparse.BooleanOptionalAction, default=True, help="Enable self-conditioned draft training in COROSred Phase B")
    parser.add_argument("--self-cond-prob", type=float, default=0.5, help="Probability of training on model drafts vs clean masks in Phase B")
    parser.add_argument("--init-checkpoint", type=str, default=None, help="Path to initial checkpoint to load weights from before training")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None, help="Enable torch.compile for PyTorch CUDA execution")

    args = parser.parse_args()

    try:
        train(
            paradigm=args.paradigm,
            phase=args.phase,
            params=args.params,
            tokens=args.tokens,
            effective_batch=args.effective_batch,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            seq_len=args.seq_len,
            tokenizer=args.tokenizer,
            vocab_size=args.vocab_size,
            hardware=args.hardware,
            devices=args.devices,
            max_steps=args.max_steps,
            max_lr=args.max_lr,
            min_lr=args.min_lr,
            warmup_steps=args.warmup_steps,
            weight_decay=args.weight_decay,
            checkpoint_dir=args.checkpoint_dir,
            save_every=args.save_every,
            config_path=args.config,
            data_path=args.data,
            synthetic=args.synthetic,
            resume_step=args.resume,
            eval_policy=args.eval_policy,
            benchmark=args.benchmark,
            benchmark_duration=args.benchmark_duration,
            self_condition=args.self_condition,
            self_cond_prob=args.self_cond_prob,
            init_checkpoint=args.init_checkpoint,
            compile=args.compile,
            legacy_phases=args.legacy_phases,
            alpha_max=args.alpha_max,
            alpha_min=args.alpha_min,
            beta_min=args.beta_min,
            beta_max=args.beta_max,
            gamma_max=args.gamma_max,
            hold_fraction=args.hold_frac,
            decay_power=args.decay_power,
            acc_gate_threshold=args.acc_gate,
            causal_ratio=args.causal_ratio,
            routing_cache_steps=args.routing_cache_steps,
            adaptive_rebalance=args.adaptive_rebalance,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
