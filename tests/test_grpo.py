"""GRPO rollout, loss and checkpoint correctness."""

import json

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers

from telos.models import TelosConfig, TelosTransformer
from telos.posttrain import grpo
from telos.posttrain.trainer import posttrain


def _tokenizer():
    tokens = ["[PAD]", "[MASK]", "[BOS]", "[EOS]", "[UNK]", "Q", "A", "B", "C", "D"]
    tokenizer = Tokenizer(models.WordLevel({token: i for i, token in enumerate(tokens)}, unk_token="[UNK]"))
    tokenizer.add_special_tokens(tokens[:5])
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return tokenizer


def varied_reward(record, completion):
    return sum(completion.encode()) / 100.0 + float(completion == record["answer"])


def test_rollouts_keep_individual_eos_targets_and_group_advantages():
    tokenizer = _tokenizer()
    records = [{"prompt": "Q", "answer": "A"}, {"prompt": "Q Q", "answer": ""}]
    prompts, eos, pad = grpo.prepare_prompts(records, tokenizer, max_length=5, max_new_tokens=2)
    assert prompts[0][1] == [2, 5]
    assert prompts[1][1] == [2, 5, 5]

    calls = iter([([1, 1, 2, 2], [6, 7, eos, eos]), ([2, 2], [eos, 8])])

    def sample(ids, positions):
        expected, tokens = next(calls)
        assert positions == expected
        assert ids.shape == (len(expected), 3)
        return tokens, [-0.25 * token for token in tokens]

    rows, action_logps, advantages, reward, uniform = grpo.collect_rollouts(
        prompts, tokenizer, eos, pad, 2, 2, sample, grpo.reward_function("exact"),
    )
    assert [row["_encoded"][1] for row in rows] == [
        [-100, -100, 6, eos], [-100, -100, 7, 8], [-100, -100, -100, eos], [-100, -100, -100, eos],
    ]
    np.testing.assert_allclose(grpo.collate_old_logps(rows, action_logps, 5), [
        [0.0, -1.5, -0.75, 0.0], [0.0, -1.75, -2.0, 0.0],
        [0.0, 0.0, -0.75, 0.0], [0.0, 0.0, -0.75, 0.0],
    ])
    np.testing.assert_allclose(advantages, [1, -1, 0, 0])
    assert reward == 0.75
    assert uniform == 0.5


def test_grpo_clipping_masks_and_reference_kl():
    labels = torch.tensor([[-100, -100, 1, 2], [-100, 3, -100, -100]])
    old = torch.full((2, 3), -3.0)
    logps = (old + torch.log(torch.tensor([[4.0, 1.5, 1.0], [0.5, 4.0, 4.0]]))).requires_grad_()
    advantages = torch.tensor([1.0, -1.0])
    loss = grpo.torch_loss(logps, old, old, labels, advantages, 0.0, 0.2)
    assert loss.item() == pytest.approx(-0.15)
    loss.backward()
    torch.testing.assert_close(logps.grad, torch.tensor([[0., 0., -.25], [0., 0., 0.]]))
    kl = grpo.torch_loss(logps, old, old, labels, advantages, 0.3, 0.2)
    assert kl > loss

    mx = pytest.importorskip("mlx.core")
    args = [mx.array(t.detach().numpy()) for t in (logps, old, old, labels, advantages)]
    actual = grpo.mlx_loss(*args, 0.3, 0.2)
    assert actual.item() == pytest.approx(kl.item(), abs=5e-7)


@pytest.mark.parametrize("hardware,precision,finetune", [
    ("cpu", "float32", "full"), ("mlx", "bf16", "full"), ("mlx", "float32", "qlora"),
])
def test_grpo_rollout_updates_and_resume(tmp_path, hardware, precision, finetune):
    if hardware == "mlx":
        from telos.training.core import metal_usable
        if not metal_usable():
            pytest.skip("Metal required")
    tokenizer = _tokenizer()
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    config = TelosConfig(vocab_size=tokenizer.get_vocab_size(), d_model=64, n_layers=1,
                        n_heads=2, n_kv_heads=2, max_seq_len=12, is_causal=True)
    torch.manual_seed(31)
    model = TelosTransformer(config)
    source = tmp_path / "base.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": {
        "paradigm": "ar", "model": vars(config), "data": {"tokenizer": str(tokenizer_path)},
    }}, source)
    data = tmp_path / "rollouts.jsonl"
    data.write_text('{"prompt":"Q","answer":"A"}\n{"prompt":"Q Q","answer":"B"}\n')
    opts = dict(method="grpo", hardware=hardware, precision=precision, finetune=finetune,
                max_steps=3, batch_size=2, max_new_tokens=3, group_size=4, policy_updates=2,
                reward=f"{__name__}:varied_reward", save_every=1, learning_rate=0.003)
    final = posttrain(source, data, tmp_path / "full", **opts)
    suffix = ".resume.npz" if hardware == "mlx" else ".resume.pt"
    resume = tmp_path / "full" / ("checkpoint_step_1" + suffix)
    replay = posttrain(source, data, tmp_path / "resumed", resume=resume, **opts)

    if hardware == "mlx":
        import mlx.core as mx
        from mlx.utils import tree_flatten
        from telos.training.checkpoint import load_mlx_state
        a, b = [load_mlx_state(path.with_suffix(suffix)) for path in (final, replay)]
        for group in ("model", "optimizer", "reference", "rng"):
            for (ka, va), (kb, vb) in zip(tree_flatten(a[group]), tree_flatten(b[group]), strict=True):
                assert ka == kb and mx.array_equal(va, vb).item(), (group, ka)
        assert a["optimizer"]["step"].item() == 6
        initial = load_mlx_state(resume)
        assert any(not mx.array_equal(v, iv).item() for (_, v), (_, iv) in zip(
            tree_flatten(a["model"]), tree_flatten(initial["model"]), strict=True))
    else:
        a, b = [torch.load(path.with_suffix(suffix), weights_only=True) for path in (final, replay)]
        assert torch.equal(a["rng"], b["rng"])
        for key in a["model"]:
            assert torch.equal(a["model"][key], b["model"][key]), key
        for key in a["optimizer"]["state"]:
            assert a["optimizer"]["state"][key]["step"].item() == 6
            for name, value in a["optimizer"]["state"][key].items():
                assert torch.equal(value, b["optimizer"]["state"][key][name])
        assert any(not torch.equal(a["model"][key], a["reference"][key]) for key in a["model"])

    with pytest.raises(ValueError, match="configuration"):
        posttrain(source, data, tmp_path / "mismatch", resume=resume, **(opts | {"group_size": 3}))
