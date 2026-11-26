"""Tiny sharded fixtures for Inkling's fixed-loop layout and packed experts."""

import json
import math
from dataclasses import replace

import pytest

from telos.models.inkling2 import InklingTextLayout
from telos.models.ssd_experts import SSDExpertCache
from telos.training.core import metal_usable

pytestmark = pytest.mark.skipif(not metal_usable(), reason="MLX Metal required")


def _write_fixture(path, mx):
    config = {
        "model_type": "inkling2_loop",
        "text_config": {
            "num_hidden_layers": 2,
            "hidden_size": 32,
            "intermediate_size": 32,
            "n_routed_experts": 2,
            "num_experts_per_tok": 1,
            "route_scale": 8.0,
            "use_gate_bias": True,
            "mlp_gate_softcap": 4.0,
            "mlp_up_softcap": 25.0,
            "loop_config": {"core_start": 0, "core_end": 1, "num_passes": 3, "is_varloop": False},
        },
    }
    (path / "config.json").write_text(json.dumps(config))
    index = {}
    for layer, (one, two, columns, scales, unit_scale) in enumerate((
        (0x22222222, 0x44444444, 4, 2, 0x38),  # E2M1 codes, with E4M3 scale 1
        (0x38383838, 0x40404040, 8, 1, 127),  # E4M3 codes, with E8M0 scale 1
    )):
        prefix = f"model.language_model.layers.{layer}.mlp."
        gate_up = mx.tile(mx.array([[one], [two]], dtype=mx.uint32), (2, 32, columns))
        tensors = {
            "gate_up_proj.weight": gate_up,
            "gate_up_proj.scales": mx.full((2, 64, scales), unit_scale, dtype=mx.uint8),
            "down_proj.weight": mx.full((2, 32, columns), one, dtype=mx.uint32),
            "down_proj.scales": mx.full((2, 32, scales), unit_scale, dtype=mx.uint8),
        }
        if layer == 0:
            tensors["gate_up_proj.global_scale"] = mx.array([2688.0, 1344.0], dtype=mx.float32)
            tensors["down_proj.global_scale"] = mx.array([2688.0, 1344.0], dtype=mx.float32)
        shard = f"model-0000{layer + 1}-of-00002.safetensors"
        weights = {prefix + "experts." + key: value for key, value in tensors.items()}
        weights[prefix + "gate.weight"] = mx.broadcast_to(mx.array([[0.0], [1.0]]), (2, 32))
        weights[prefix + "gate.e_score_correction_bias"] = mx.array([1.0, 0.0])
        mx.save_safetensors(str(path / shard), weights)
        index.update(dict.fromkeys(weights, shard))
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_inkling2_fixed_loop_expert_projection_parity(tmp_path, dtype):
    import mlx.core as mx

    class FixtureGuard:
        def check(self):
            pass

    _write_fixture(tmp_path, mx)
    layout = InklingTextLayout.from_checkpoint(tmp_path)
    assert layout.visits == ((0, 0), (0, 2), (0, 3), (1, 1))
    assert tuple(bank.mode for bank in layout.expert_banks) == ("nvfp4", "mxfp8")
    first, second = layout.expert_banks
    first_bytes = sum(entry.size // 2 for entry in first.ranges.values())
    second_bytes = sum(entry.size // 2 for entry in second.ranges.values())
    cache = SSDExpertCache(max(first_bytes, second_bytes), FixtureGuard())
    x = mx.full((1, 32), 1 / 32, dtype=getattr(mx, dtype))

    for bank in (first, second):
        resident_tensors = mx.load(str(next(iter(bank.ranges.values())).path))
        for expert in (0, 1):
            prefix = f"model.language_model.layers.{bank.layer}.mlp.experts."
            resident = {key: resident_tensors[prefix + key][expert] for key in bank.ranges}
            streamed = cache.get(bank.layer, expert, bank.ranges)
            for key, array in resident.items():
                assert mx.array_equal(array, streamed[key]).item()
            projected = bank.project(x, streamed)
            assert mx.array_equal(projected, bank.project(x, resident)).item()
            outer = 0.5 if bank.mode == "nvfp4" and expert == 1 else 1.0
            expected = (32 * outer * 4 * math.tanh(outer / 4) / (1 + math.exp(-outer))
                        * 25 * math.tanh(2 * outer / 25))
            tolerance = 1e-5 if dtype == "float32" else 0.01
            assert mx.allclose(projected, mx.full((1, 32), expected), rtol=tolerance, atol=1e-5).item()
    assert cache.bytes_read == 2 * (first_bytes + second_bytes)
    assert cache.resident_bytes <= cache.byte_limit


def test_inkling2_rejects_invalid_loop_before_opening_weight_shards(tmp_path):
    config = {
        "model_type": "inkling2_loop",
        "text_config": {
            "num_hidden_layers": 2,
            "loop_config": {"core_start": 0, "core_end": 3, "num_passes": 2, "is_varloop": False},
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors.index.json").touch()
    with pytest.raises(ValueError, match="loop bounds"):
        InklingTextLayout.from_checkpoint(tmp_path)


def test_inkling2_router_bias_affects_selection_but_not_mixture(tmp_path):
    import mlx.core as mx

    class FixtureGuard:
        def check(self):
            pass

    _write_fixture(tmp_path, mx)
    bank = InklingTextLayout.from_checkpoint(tmp_path).expert_banks[0]
    router = bank.load_router()
    x = mx.full((1, 32), 1 / 32, dtype=mx.float32)
    indices, weights = bank.route(x, router)
    assert indices.tolist() == [[0]]
    assert weights.tolist() == [[8.0]]

    bank = replace(bank, top_k=2)
    indices, weights = bank.route(x, router)
    scores = mx.array([0.5, 1 / (1 + math.exp(-1))])
    expected_weights = 8 * scores / scores.sum()
    assert indices.tolist() == [[0, 1]]
    assert mx.allclose(weights[0], expected_weights).item()

    one_expert = sum(entry.size // 2 for entry in bank.ranges.values())
    cache = SSDExpertCache(one_expert, FixtureGuard())
    expected = mx.zeros_like(x)
    for expert in (0, 1):
        resident = {key: entry.load()[expert] for key, entry in bank.ranges.items()}
        expected = expected + bank.project(x, resident) * weights[0, expert]
    assert mx.array_equal(bank.routed(x, router, cache), expected).item()
    assert cache.bytes_read == 2 * one_expert
    assert cache.resident_bytes == one_expert
