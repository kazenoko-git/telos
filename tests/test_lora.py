from copy import deepcopy
import pytest
import torch
from telos.models import TelosTransformer
from telos.posttrain.adapters import apply_lora, merged_state_dict, NF4Linear


def model():
    return TelosTransformer(vocab_size=64, d_model=32, n_layers=1, n_heads=2,
                            n_kv_heads=1, max_seq_len=16, is_causal=True)


@pytest.mark.parametrize('quantized', [False, True])
def test_lora_only_adapters_update_and_merge_reloads(quantized):
    torch.manual_seed(7)
    active = model()
    frozen = deepcopy(active.state_dict())
    apply_lora(active, rank=4, alpha=8, quantized=quantized)
    trainable = [name for name, p in active.named_parameters() if p.requires_grad]
    assert trainable and all(name.endswith(('lora_a', 'lora_b')) for name in trainable)
    tokens = torch.randint(0, 64, (2, 8))
    optim = torch.optim.AdamW((p for p in active.parameters() if p.requires_grad), lr=.01)
    before = deepcopy(active.state_dict())
    for _ in range(2):
        optim.zero_grad()
        active(tokens).square().mean().backward()
        optim.step()
    for name, value in active.state_dict().items():
        if not name.endswith(('lora_a', 'lora_b')):
            assert torch.equal(value, before[name])
    assert any(not torch.equal(value, before[name]) for name, value in active.state_dict().items()
               if name.endswith('lora_b'))
    exported = model()
    exported.load_state_dict(merged_state_dict(active), strict=True)
    torch.testing.assert_close(exported(tokens), active(tokens), atol=2e-6, rtol=2e-5)
    assert torch.equal(exported.tok_embeddings.weight, frozen['tok_embeddings.weight'])
    assert exported.tok_embeddings.weight is exported.output_projection.weight


def test_nf4_storage_and_error():
    layer = torch.nn.Linear(128, 64, bias=False)
    quantized = NF4Linear(layer)
    assert quantized.packed_weight.numel() == layer.weight.numel() // 2
    assert not list(quantized.parameters())
    error = (quantized.dequantize() - layer.weight).square().mean().sqrt()
    assert error < layer.weight.square().mean().sqrt() * .15
    zero = torch.nn.Linear(64, 4, bias=False)
    torch.nn.init.zeros_(zero.weight)
    assert torch.equal(NF4Linear(zero).dequantize(), zero.weight)


def test_lora_zero_initialization_preserves_model():
    active = model()
    tokens = torch.randint(0, 64, (2, 8))
    expected = active(tokens)
    apply_lora(active, rank=4)
    torch.testing.assert_close(active(tokens), expected, rtol=0, atol=0)
