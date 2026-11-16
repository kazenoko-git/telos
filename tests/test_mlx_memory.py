import pytest

from telos.training.mlx_memory import (
    GIB, estimate_training_bytes, plan_training_batch, training_memory_budget,
)

MODEL = dict(d_model=512, n_layers=14, n_heads=8, seq_len=256, vocab_size=8192)


def test_budget_includes_headroom_and_honors_smaller_caps():
    budget = training_memory_budget(24 * GIB)
    assert budget.ceiling_bytes == int(24 * GIB * .6)
    assert budget.allocation_bytes + GIB == budget.ceiling_bytes
    assert training_memory_budget(24 * GIB, recommended_bytes=10 * GIB).ceiling_bytes == 10 * GIB
    assert training_memory_budget(24 * GIB, limit_bytes=8 * GIB).allocation_bytes == 7 * GIB
    with pytest.raises(ValueError, match='headroom'):
        training_memory_budget(GIB)


def test_largest_fitting_batch_preserves_effective_batch():
    budget = training_memory_budget(24 * GIB)
    plan = plan_training_batch(49_166_848, MODEL, 256, budget)
    assert plan['estimated_peak_bytes'] + GIB <= budget.ceiling_bytes
    assert plan['batch_size'] * plan['grad_accum'] == 256
    larger = [b for b in [1, 2, 4, 8, 16, 32, 64, 128, 256] if b > plan['batch_size']]
    assert all(estimate_training_bytes(49_166_848, MODEL, b) > budget.allocation_bytes for b in larger)
    assert plan['grad_accum'] > 1


def test_full_effective_batch_only_when_it_fits_and_handles_nonpowers():
    tiny = dict(d_model=64, n_layers=1, n_heads=1, seq_len=16, vocab_size=128)
    plan = plan_training_batch(10000, tiny, 30, training_memory_budget(24 * GIB))
    assert (plan['batch_size'], plan['grad_accum']) == (30, 1)


def test_checkpoint_fallback_and_untrainable_model():
    budget = training_memory_budget(4 * GIB)
    long = {**MODEL, 'seq_len': 2048}
    plan = plan_training_batch(10_000_000, long, 8, budget)
    assert plan['checkpointing']
    assert plan['grad_accum'] > 1
    with pytest.raises(RuntimeError, match='microbatch 1'):
        plan_training_batch(10_000_000_000, MODEL, 256, budget)


def test_explicit_checkpointing_stays_enabled_and_invalid_batches_fail():
    budget = training_memory_budget(24 * GIB)
    assert plan_training_batch(10000, MODEL, 7, budget, checkpointing=True)['checkpointing']
    with pytest.raises(ValueError):
        plan_training_batch(10000, MODEL, 0, budget)


def test_trainer_applies_plan_without_changing_effective_batch():
    mx = pytest.importorskip('mlx.core')
    if not mx.metal.is_available():
        pytest.skip('Metal required')
    from telos.models.mlx_transformer import MLXTelosTransformer
    from telos.training.trainer_mlx import UnifiedMLXTrainer

    cfg = dict(model=dict(vocab_size=128, d_model=64, n_layers=1, n_heads=1,
                          seq_len=16, is_causal=True),
               training=dict(batch_size=2, gradient_accumulation=4))
    trainer = UnifiedMLXTrainer('ar', MLXTelosTransformer(**cfg['model']), cfg)
    assert cfg['training']['batch_size'] == 8
    assert cfg['training']['gradient_accumulation'] == 1
    assert trainer.memory_plan['effective_batch'] == 8
    assert trainer.memory_plan['estimated_peak_bytes'] + GIB <= trainer.memory_budget.ceiling_bytes


def test_measured_training_saves_selected_memory_plan(tmp_path, monkeypatch):
    import json
    mx = pytest.importorskip('mlx.core')
    if not mx.metal.is_available():
        pytest.skip('Metal required')
    from telos.models.mlx_transformer import MLXTelosTransformer
    from telos.training.trainer_mlx import UnifiedMLXTrainer

    monkeypatch.chdir(tmp_path)
    cfg = dict(model=dict(vocab_size=128, d_model=64, n_layers=1, n_heads=1,
                          seq_len=16, is_causal=True),
               data=dict(synthetic=True),
               training=dict(batch_size=1, gradient_accumulation=2, max_steps=2,
                             max_lr=1e-3, weight_decay=.1, autotune=True,
                             autotune_candidates=[1, 2], autotune_checkpointing=False),
               checkpoint=dict(checkpoint_dir=str(tmp_path / 'checkpoint')))
    trainer = UnifiedMLXTrainer('ar', MLXTelosTransformer(**cfg['model']), cfg)
    trainer.train()
    saved = json.loads((tmp_path / 'checkpoint' / 'config.json').read_text())
    plan = saved['training']['memory_plan']
    assert plan['effective_batch'] == plan['batch_size'] * plan['grad_accum'] == 2
    assert plan['batch_size'] == 2
    assert plan['measured_peak_bytes'] + GIB <= plan['memory_ceiling_bytes']
    assert plan == trainer.memory_plan
