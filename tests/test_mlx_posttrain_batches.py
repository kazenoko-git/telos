"""Native collation and frozen-reference reuse preserve post-training semantics."""
import json

import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core")
from telos.posttrain.mlx_batches import collate_mlx, ReferenceScores
from telos.posttrain.trainer import _collate, _prepare_records, _response_rows, posttrain
from telos.data.tokenizer import load_tokenizer


@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("response", ["chosen", "rejected"])
def test_native_collation_preserves_tokens_labels_and_padding(fixed, response):
    tok = load_tokenizer()
    records = [dict(prompt="long prompt " * 30, chosen="A", rejected=""),
               dict(prompt="Q", chosen="answer " * 30, rejected="B C")]
    prepared = _prepare_records(records, tok, 16, "dpo")
    rows = _response_rows(prepared, response)
    expected = _collate(rows, tok, 16, torch.device("cpu"), fixed=fixed)
    actual = collate_mlx(rows, 16, fixed=fixed)
    for a, b in zip(actual, expected):
        assert a.dtype == mx.int32
        np.testing.assert_array_equal(np.array(a), b.numpy())


def test_reference_cache_reuses_reordered_rows_and_bounds_storage():
    cache = ReferenceScores(1, 2)
    rows = [{"_reference_key": key} for key in ("a", "b", "a", "c")]
    calls = []

    def evaluate():
        calls.append(1)
        return mx.array([1., 2., 1., 3.])

    assert cache.get(rows, 8, evaluate).tolist() == [1., 2., 1., 3.]
    reordered = [rows[i] for i in (3, 2, 1, 0)]
    assert cache.get(reordered, 8, evaluate).tolist() == [3., 1., 2., 1.]
    assert len(calls) == 1
    # Another padding width cannot reuse scores from a different forward shape.
    cache.get(rows, 16, evaluate)
    cache.get(rows, 16, evaluate)
    assert len(calls) == 3
    assert len(cache.scores) == cache.capacity == 4
    assert all(isinstance(value, float) for value in cache.scores.values())


def test_disabled_reference_cache_always_evaluates():
    cache = ReferenceScores(0, 2)
    calls = []
    for _ in range(2):
        cache.get([{}], 8, lambda: calls.append(1) or mx.array([1.]))
    assert len(calls) == 2
    assert not cache.scores


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_reordered_reference_scores_preserve_dpo_loss_and_gradients(dtype):
    if not mx.metal.is_available():
        pytest.skip("Metal required")
    import copy
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from telos.models import MLXTelosTransformer
    from telos.posttrain.trainer import _mlx_logps

    mx.random.seed(19)
    model = MLXTelosTransformer(vocab_size=64, d_model=32, n_layers=1, n_heads=2,
                                precision="float32", is_causal=True)
    model.set_dtype(dtype)
    reference = copy.deepcopy(model)
    reference.eval()
    ids = mx.array([[1, 2, 3, 4], [1, 5, 6, 7], [1, 8, 9, 10], [1, 11, 12, 13]])
    labels = mx.where(ids < 3, -100, ids)
    cache = ReferenceScores(1, 2)
    rows = [{"_reference_key": i} for i in range(4)]
    cache.get(rows, 4, lambda: _mlx_logps(reference, ids, labels))
    order = mx.array([1, 0, 3, 2])
    ids, labels = ids[order], labels[order]
    cached = cache.get([rows[i] for i in (1, 0, 3, 2)], 4,
                       lambda: pytest.fail("reordered responses should hit the cache"))
    uncached = _mlx_logps(reference, ids, labels)
    assert mx.array_equal(cached, uncached).item()

    def loss(policy, scores):
        c, r = mx.split(_mlx_logps(policy, ids, labels), 2)
        rc, rr = mx.split(scores, 2)
        return mx.mean(mx.logaddexp(0, -0.1 * ((c - r) - (rc - rr))))

    grad = nn.value_and_grad(model, loss)
    a, ga = grad(model, cached)
    b, gb = grad(model, uncached)
    assert mx.array_equal(a, b).item()
    for (ka, va), (kb, vb) in zip(tree_flatten(ga), tree_flatten(gb)):
        assert ka == kb
        assert mx.allclose(va, vb, rtol=1e-5, atol=1e-6).item(), ka


@pytest.mark.parametrize("finetune,precision,padding,compile_step", [
    ("full", "float32", "fixed", True), ("full", "bf16", "fixed", True),
    ("lora", "bf16", "fixed", True), ("qlora", "bf16", "fixed", True),
    ("full", "float32", "batch", False)])
def test_dpo_cache_preserves_updates_and_resume_state(tmp_path, finetune, precision, padding, compile_step):
    if not mx.metal.is_available():
        pytest.skip("Metal required")
    from telos.models import TelosConfig, TelosTransformer
    from telos.training.checkpoint import load_mlx_state
    from mlx.utils import tree_flatten

    tok = load_tokenizer()
    cfg = TelosConfig(vocab_size=len(tok.get_vocab()), d_model=64, n_layers=1,
                      n_heads=2, max_seq_len=16, is_causal=True)
    torch.manual_seed(13)
    source = tmp_path / "base.pt"
    torch.save({"model_state_dict": TelosTransformer(cfg).state_dict(),
                "config": {"paradigm": "ar", "model": vars(cfg)}}, source)
    data = tmp_path / "rows.jsonl"
    data.write_text("\n".join(json.dumps(dict(prompt=f"Q{i}", chosen="A " * (i + 1), rejected=f"B{i}"))
                              for i in range(3)))
    options = dict(method="dpo", hardware="mlx", finetune=finetune, precision=precision,
                   lora_rank=4, lora_dropout=.1, max_steps=6, batch_size=2,
                   max_length=12, save_every=3, seed=7, padding=padding, compile_step=compile_step)
    for enabled in (False, True):
        posttrain(source, data, tmp_path / str(enabled), cache_reference=enabled, **options)
    a, b = [load_mlx_state(tmp_path / str(enabled) / "model.resume.npz")
            for enabled in (False, True)]
    assert any(mx.any(value != 0).item() for key, value in tree_flatten(b["optimizer"])
               if key.endswith(".m")), "DPO must produce nonzero optimizer moments"
    # Check the full saved state: weights, FP32 masters, moments and all RNGs.
    def compare(left, right):
        if isinstance(left, mx.array):
            assert mx.allclose(left, right, rtol=1e-5, atol=1e-6).item()
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                compare(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            assert len(left) == len(right)
            for x, y in zip(left, right):
                compare(x, y)
        else:
            assert left == right
    # Options intentionally differ in cache_reference; training state must not.
    for key in ("model", "optimizer", "master", "reference", "rng", "sampler", "host_rng"):
        compare(a[key], b[key])
    posttrain(source, data, tmp_path / "resumed", cache_reference=True,
              resume=tmp_path / "True" / "checkpoint_step_3.resume.npz", **options)
    resumed = load_mlx_state(tmp_path / "resumed" / "model.resume.npz")
    for key in ("model", "optimizer", "master", "reference", "rng", "sampler", "host_rng"):
        compare(b[key], resumed[key])
