import math

import pytest

from telos.training.mlx_tuning import batch_plans, measure_plans, select_trial


def test_batch_plans_preserve_effective_batch_and_include_large_microbatches():
    plans = batch_plans(512)
    assert {128, 256, 512}.issubset({p.batch_size for p in plans})
    assert all(p.effective_batch == 512 for p in plans)
    assert [p.batch_size for p in batch_plans(30, candidates=[2, 4, 5, 8, 10])] == [2, 5, 10]


def test_selection_rejects_failed_over_budget_and_nonfinite_trials():
    trials = [
        {"status": "failed", "tokens_per_second": 10000, "peak_memory_bytes": 10},
        {"status": "ok", "tokens_per_second": 10000, "peak_memory_bytes": 1001},
        {"status": "ok", "tokens_per_second": math.nan, "peak_memory_bytes": 10},
        {"status": "ok", "tokens_per_second": 100, "peak_memory_bytes": 900},
        {"status": "ok", "tokens_per_second": 99, "peak_memory_bytes": 500},
        {"status": "ok", "tokens_per_second": 80, "peak_memory_bytes": 100},
    ]
    assert select_trial(trials, memory_budget_bytes=1000) is trials[4]
    with pytest.raises(RuntimeError, match="No tuning trial"):
        select_trial(trials[:3], memory_budget_bytes=1000)


def test_generic_runner_measures_every_configuration():
    seen = []
    def runner(plan):
        seen.append(plan)
        return {"status": "ok", "tokens_per_second": plan.batch_size * 100, "peak_memory_bytes": 1}
    plans = batch_plans(256)
    report = measure_plans(plans, runner, memory_budget_bytes=1000)
    assert seen == plans
    assert report["selected"]["plan"] == {"batch_size": 256, "grad_accum": 1, "checkpointing": False}


def test_isolated_trials_do_not_mutate_model_or_rng():
    import numpy as np
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("Metal required")
    from mlx.utils import tree_flatten
    from telos.models.mlx_transformer import MLXTelosTransformer
    from telos.training.trainer_mlx import UnifiedMLXTrainer
    from telos.training.mlx_tuning import tune_mlx_batch

    cfg = {
        "model": dict(vocab_size=128, d_model=64, n_layers=1, n_heads=1, seq_len=16, is_causal=True),
        "training": dict(batch_size=2, gradient_accumulation=2, max_lr=1e-3, weight_decay=0.1,
                         autotune_candidates=[2, 4], autotune_checkpointing=False),
    }
    model = MLXTelosTransformer(**cfg["model"])
    trainer = UnifiedMLXTrainer("ar", model, cfg)
    before = [np.array(p.astype(mx.float32)) for _, p in tree_flatten(model.parameters())]
    rng_before = [np.array(p) for p in mx.random.state]
    report = tune_mlx_batch(trainer, (np.arange(32).reshape(2, 16) % 124 + 4).astype(np.int32))
    assert len(report["trials"]) == 2
    assert report["effective_batch"] == 4
    assert report["selected"]["plan"]["batch_size"] in {2, 4}
    for a, (_, p) in zip(before, tree_flatten(model.parameters())):
        np.testing.assert_array_equal(a, np.array(p.astype(mx.float32)))
    for a, p in zip(rng_before, mx.random.state):
        np.testing.assert_array_equal(a, np.array(p))


@pytest.mark.parametrize("auto_enabled,tune_checkpoints,expected", [
    (True, True, [2]),
    (False, True, []),
    (True, False, []),
])
def test_smaller_batches_can_drop_automatic_checkpointing(monkeypatch, auto_enabled, tune_checkpoints, expected):
    from types import SimpleNamespace
    import numpy as np
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("Metal required")
    from telos.training import mlx_tuning

    model = SimpleNamespace(use_grad_checkpoint=True, parameters=lambda: {}, save_weights=lambda path: None)
    cfg = {
        "model": {},
        "training": {
            "batch_size": 4, "gradient_accumulation": 2,
            "autotune_candidates": [2, 4, 8], "autotune_checkpointing": tune_checkpoints,
        },
    }
    trainer = SimpleNamespace(cfg=cfg, model=model, auto_enabled_checkpointing=auto_enabled,
                              seq_len=4, paradigm="ar")
    seen = []

    def capture(plans, runner, **kwargs):
        seen.extend(plans)
        return {}

    monkeypatch.setattr(mlx_tuning, "measure_plans", capture)
    mlx_tuning.tune_mlx_batch(trainer, np.zeros((8, 4), dtype=np.int32))
    assert [p.batch_size for p in seen if not p.checkpointing] == expected
    assert all(p.effective_batch == 8 for p in seen)


def test_memory_policy_chooses_largest_fitting_batch_even_if_slower():
    trials = [
        dict(status='ok', tokens_per_second=200, peak_memory_bytes=100,
             plan=dict(batch_size=2, grad_accum=8, checkpointing=False)),
        dict(status='ok', tokens_per_second=150, peak_memory_bytes=900,
             plan=dict(batch_size=8, grad_accum=2, checkpointing=False)),
        dict(status='ok', tokens_per_second=300, peak_memory_bytes=1001,
             plan=dict(batch_size=16, grad_accum=1, checkpointing=False)),
    ]
    assert select_trial(trials, memory_budget_bytes=1000, prefer_largest_batch=True) is trials[1]


def test_candidates_include_every_divisor_and_full_effective_batch():
    assert [p.batch_size for p in batch_plans(30)] == [1, 2, 3, 5, 6, 10, 15, 30]
    assert batch_plans(1024)[-1].batch_size == 1024


@pytest.mark.parametrize('policy,async_eval,compiled', [('eager', True, False), ('step', True, True)])
def test_trial_uses_training_compilation_and_dispatch(tmp_path, monkeypatch, policy, async_eval, compiled):
    import numpy as np
    mx = pytest.importorskip('mlx.core')
    if not mx.metal.is_available():
        pytest.skip('Metal required')
    from telos.models.mlx_transformer import MLXTelosTransformer
    from telos.training import mlx_step, mlx_tuning
    weights, sample = tmp_path / 'weights.safetensors', tmp_path / 'sample.npy'
    cfg = {
        'model': dict(vocab_size=32, d_model=64, n_layers=1, n_heads=2, seq_len=8, is_causal=True),
        'training': dict(precision='bfloat16', batch_size=2, gradient_accumulation=2,
                         max_lr=.001, weight_decay=.1, compiled_accumulation=compiled,
                         async_accumulation=False, async_eval=async_eval),
    }
    model = MLXTelosTransformer(**cfg['model'])
    model.save_weights(str(weights))
    np.save(sample, np.full((4, 8), 7, dtype=np.int32))
    build = mlx_step.build_training_step
    seen = []
    def wrapped(*args, **kwargs):
        actual = build(*args, **kwargs)
        assert (actual.accumulation_step is not None) == compiled
        assert actual.async_accumulation is False
        class Observed:
            def __call__(self, *batch, **options):
                seen.append(options)
                return actual(*batch, **options)
            def synchronize(self):
                return actual.synchronize()
        return Observed()
    monkeypatch.setattr(mlx_step, 'build_training_step', wrapped)
    budget = 2 * 1024**3
    old_memory = mx.set_memory_limit(budget)
    old_cache = mx.set_cache_limit(128 * 1024**2)
    try:
        report = mlx_tuning._run_trial(dict(cfg=cfg, paradigm='ar', weights=str(weights), sample=str(sample),
            plan=dict(batch_size=2, grad_accum=2, checkpointing=False), eval_policy=policy,
            budget=budget, warmup_steps=3, measure_steps=3))
    finally:
        mx.set_memory_limit(old_memory)
        mx.set_cache_limit(old_cache)
    assert report['status'] == 'ok'
    assert seen and all(x['eval_every_microbatch'] == (policy == 'eager') for x in seen)
    assert all(x['asynchronous'] == async_eval for x in seen)
