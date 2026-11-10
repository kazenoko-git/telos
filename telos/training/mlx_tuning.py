"""Measured microbatch tuning with isolated trials and a fixed effective batch.

The coordinator is model-agnostic: a trial runner accepts a BatchPlan and returns
measurements. The built-in subprocess adapter currently supports Telos models.
"""

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


@dataclass(frozen=True)
class BatchPlan:
    batch_size: int
    grad_accum: int
    checkpointing: bool = False

    @property
    def effective_batch(self):
        return self.batch_size * self.grad_accum


def batch_plans(effective_batch, *, candidates=None, checkpointing=False):
    """Only exact divisors: never silently change the optimization batch."""
    if effective_batch < 1:
        raise ValueError("effective_batch must be positive")
    candidates = candidates if candidates is not None else [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    sizes = sorted({int(n) for n in candidates if int(n) > 0 and effective_batch % int(n) == 0})
    return [BatchPlan(n, effective_batch // n, checkpointing) for n in sizes]


def select_trial(trials, *, memory_budget_bytes):
    """Choose measured throughput; prefer lower memory within 2% of the best."""
    valid = [t for t in trials if t.get("status") == "ok"
             and math.isfinite(t.get("tokens_per_second", float("nan")))
             and t["tokens_per_second"] > 0
             and t.get("peak_memory_bytes", math.inf) <= memory_budget_bytes]
    if not valid:
        raise RuntimeError("No tuning trial completed within the memory budget; lower the effective batch or context length")
    best = max(t["tokens_per_second"] for t in valid)
    return min((t for t in valid if t["tokens_per_second"] >= 0.98 * best),
               key=lambda t: (t["peak_memory_bytes"], -t["tokens_per_second"]))


def measure_plans(plans, runner, *, memory_budget_bytes):
    """Generic measurement boundary for future backbones and task adapters."""
    trials = []
    for plan in plans:
        result = dict(runner(plan))
        result["plan"] = asdict(plan)
        trials.append(result)
    return {"selected": select_trial(trials, memory_budget_bytes=memory_budget_bytes), "trials": trials}


def tune_mlx_batch(trainer, dataset_matrix):
    """Benchmark a representative data sample without mutating training state.

    Every trial starts in a fresh process from the same weights and seed. MLX's
    memory limit is advisory, so estimates screen allocations before execution
    and observed peak allocations must also pass the budget to be selected.
    """
    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_flatten
    from .dataloader import get_global_targets_contiguous

    cfg = trainer.cfg
    training = cfg["training"]
    effective = int(training["batch_size"]) * int(training["gradient_accumulation"])
    current = int(training["batch_size"])
    checkpointing = bool(getattr(trainer.model, "use_grad_checkpoint", False))
    plans = batch_plans(effective, candidates=training.get("autotune_candidates"), checkpointing=checkpointing)
    # Always measure the configured baseline, including non-power-of-two sizes.
    baseline = BatchPlan(current, effective // current, checkpointing)
    plans = [baseline] + [p for p in plans if p != baseline]
    if training.get("autotune_checkpointing", True) and not checkpointing:
        plans += [BatchPlan(p.batch_size, p.grad_accum, True) for p in plans if p.batch_size >= current]
    info = mx.device_info()
    budget = int(training.get("autotune_memory_gb", 0) * 1024**3)
    if not budget:
        budget = int(min(info["max_recommended_working_set_size"] * 0.75, info["memory_size"] * 0.60))
    params = sum(p.size for _, p in tree_flatten(trainer.model.parameters()))
    print(f"  [Autotune] Fixed effective batch: {effective}; allocation budget: {budget / 1024**3:.1f} GiB", flush=True)
    with tempfile.TemporaryDirectory(prefix="telos-mlx-tune-") as directory:
        root = Path(directory)
        trainer.model.save_weights(str(root / "weights.safetensors"))
        sample, _ = get_global_targets_contiguous(dataset_matrix, 0, effective, trainer.seq_len)
        np.save(root / "sample.npy", sample)
        spec = {
            "cfg": cfg, "paradigm": trainer.paradigm,
            "weights": str(root / "weights.safetensors"), "sample": str(root / "sample.npy"),
            "budget": budget, "warmup_steps": int(training.get("autotune_warmup_steps", 3)),
            "measure_steps": int(training.get("autotune_measure_steps", 3)),
        }
        def runner(plan):
            # Conservative screening, not a throughput prediction. Checkpointed
            # blocks retain their inputs plus a layer's working intermediates.
            m = cfg["model"]
            layers = m["n_layers"]
            activation_factor = layers + 8 if plan.checkpointing else layers * 16
            estimate = params * 20 + plan.batch_size * trainer.seq_len * (
                m["d_model"] * activation_factor + trainer.vocab_size * 6)
            # This intentionally errs low on candidates. Past 100M/context128
            # runs, the observed peak exceeded the unadjusted estimate by up
            # to 1.6x, before the host had to start swapping.
            estimate = int(estimate * 1.8)
            if estimate > budget:
                result = {"status": "skipped_estimate", "estimated_memory_bytes": estimate}
            else:
                path = root / "trial.json"
                path.write_text(json.dumps({**spec, "plan": asdict(plan)}))
                env = os.environ.copy()
                # Make a source checkout importable when it is not installed.
                env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2]) + os.pathsep + env.get("PYTHONPATH", "")
                try:
                    process = subprocess.run(
                        [sys.executable, "-m", "telos.training.mlx_tuning", str(path)],
                        capture_output=True, text=True, env=env,
                        timeout=float(training.get("autotune_trial_timeout", 90)),
                    )
                    lines = [line.removeprefix("TELOS_TRIAL=") for line in process.stdout.splitlines() if line.startswith("TELOS_TRIAL=")]
                    if process.returncode == 0 and lines:
                        result = json.loads(lines[-1])
                    else:
                        result = {"status": "failed", "returncode": process.returncode,
                                  "error": (process.stderr or process.stdout)[-2000:]}
                except subprocess.TimeoutExpired:
                    result = {"status": "timeout"}
            speed = result.get("tokens_per_second", 0)
            print(f"  [Autotune] BS={plan.batch_size}, accum={plan.grad_accum}, checkpoint={plan.checkpointing}: {result['status']} {speed:,.0f} tok/s", flush=True)
            return result

        report = measure_plans(plans, runner, memory_budget_bytes=budget)
    report.update(effective_batch=effective, memory_budget_bytes=budget, device=info["device_name"],
                  mlx_version=mx.__version__, sequence_length=trainer.seq_len,
                  data_source="representative training-data sample", parameter_count=params)
    return report


def _run_trial(spec):
    import copy
    import numpy as np
    import mlx.core as mx
    import mlx.optimizers as optim
    from telos.models.mlx_transformer import MLXTelosTransformer
    from .trainer_mlx import UnifiedMLXTrainer
    from .mlx_step import MLXTrainingStep

    cfg = copy.deepcopy(spec["cfg"])
    plan = BatchPlan(**spec["plan"])
    cfg["training"]["gradient_checkpointing"] = plan.checkpointing
    cfg["model"]["use_grad_checkpoint"] = plan.checkpointing
    mx.set_memory_limit(spec["budget"])
    mx.set_cache_limit(min(spec["budget"] // 8, 512 * 1024**2))
    mx.random.seed(int(cfg["training"].get("seed", 42)))
    model_cfg = dict(cfg["model"])
    precision = model_cfg.pop("precision", cfg["training"].get("precision", "bfloat16"))
    model = MLXTelosTransformer(**model_cfg, precision=precision)
    model.load_weights(spec["weights"])
    trainer = UnifiedMLXTrainer(spec["paradigm"], model, cfg)
    # The coordinator has already chosen checkpointing for this measurement.
    model.use_grad_checkpoint = plan.checkpointing
    fn, state = trainer._setup_training_functions()
    state.append(mx.random.state)
    fn = mx.compile(fn, inputs=state, outputs=state)
    opt = optim.AdamW(cfg["training"]["max_lr"], weight_decay=cfg["training"]["weight_decay"],
                      betas=[0.9, 0.95], bias_correction=True)
    step = MLXTrainingStep(model, opt, fn, grad_clip=trainer.grad_clip,
                           fused=cfg["training"].get("fused_step", True))
    sample = np.load(spec["sample"])
    def batches():
        for i in range(plan.grad_accum):
            x = mx.array(sample[i * plan.batch_size:(i + 1) * plan.batch_size], dtype=mx.int32)
            if trainer.paradigm in {"mdlm", "undlm"}:
                yield x, trainer._sample_beta_timesteps(plan.batch_size)
            elif trainer.paradigm == "corosred" and trainer.is_unified:
                c = cfg["corosred"]
                # Include the reliability-head backward even when its schedule
                # has not yet activated, avoiding an optimistic phase estimate.
                yield x, mx.array(c["alpha_max"]), mx.array(c["beta_max"]), mx.array(c["gamma_max"])
            else:
                yield x
    loss = None
    for _ in range(max(3, spec["warmup_steps"])):
        loss, _ = step(batches(), grad_accum=plan.grad_accum, eval_every_microbatch=True)
        if mx.get_peak_memory() > spec["budget"]:
            return {"status": "over_budget", "peak_memory_bytes": mx.get_peak_memory()}
    mx.reset_peak_memory()
    times = []
    for _ in range(max(3, spec["measure_steps"])):
        start = time.perf_counter()
        loss, _ = step(batches(), grad_accum=plan.grad_accum, eval_every_microbatch=True)
        times.append(time.perf_counter() - start)
    peak = mx.get_peak_memory()
    if not math.isfinite(loss.item()):
        return {"status": "nonfinite_loss"}
    elapsed = sum(times)
    return {
        "status": "ok" if peak <= spec["budget"] else "over_budget",
        "tokens_per_second": len(times) * plan.effective_batch * sample.shape[1] / elapsed,
        "elapsed_seconds": elapsed, "steps": len(times), "step_seconds": times,
        "peak_memory_bytes": peak, "loss": loss.item() / plan.grad_accum,
    }


if __name__ == "__main__":
    print("TELOS_TRIAL=" + json.dumps(_run_trial(json.loads(Path(sys.argv[1]).read_text()))), flush=True)
