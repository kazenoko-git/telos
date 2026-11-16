from copy import deepcopy
import pytest
import torch
from telos.models import TelosTransformer
from telos.models.packed import pack_projections


@pytest.mark.parametrize('kv_heads', [1, 2])
@pytest.mark.parametrize('causal', [False, True])
def test_packed_outputs_gradients_and_checkpoint_compatibility(kv_heads, causal):
    torch.manual_seed(3)
    reference = TelosTransformer(vocab_size=64, d_model=32, n_layers=1,
                                n_heads=2, n_kv_heads=kv_heads, is_causal=causal)
    packed = deepcopy(reference)
    pack_projections(packed)
    tokens = torch.randint(0, 64, (2, 8))
    expected, actual = reference(tokens), packed(tokens)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    expected.square().mean().backward(); actual.square().mean().backward()
    torch.testing.assert_close(reference.tok_embeddings.weight.grad, packed.tok_embeddings.weight.grad)
    qkv = packed.layers[0].attn.qkv_proj.weight.grad.split(packed.layers[0].attn.qkv_sizes, 0)
    for name, gradient in zip(('q_proj','k_proj','v_proj'),qkv):
        torch.testing.assert_close(getattr(reference.layers[0].attn,name).weight.grad, gradient)
    weights = packed.state_dict()
    assert weights.keys() == reference.state_dict().keys()
    reference.load_state_dict(weights, strict=True)
    packed.load_state_dict(reference.state_dict(), strict=True)
    torch.testing.assert_close(reference(tokens), packed(tokens))


def test_packed_trainer_resume(tmp_path):
    from telos.training.trainer_pytorch import UnifiedPyTorchTrainer
    cfg={'paradigm':'ar','model':{'vocab_size':64,'seq_len':8,'d_model':32,'n_layers':1,'n_heads':2},
         'training':{'max_steps':4,'batch_size':2,'gradient_accumulation':2,'packed_projections':True,
                     'precision':'float32','seed':17,'max_lr':.001,'warmup_steps':1},
         'data':{'synthetic':True},'checkpoint':{'checkpoint_dir':str(tmp_path/'out'),'save_every_steps':2}}
    def make():
        torch.manual_seed(19)
        return UnifiedPyTorchTrainer('ar',TelosTransformer(vocab_size=64,d_model=32,n_layers=1,n_heads=2,is_causal=True),deepcopy(cfg),device_type='cpu')
    reference=make();reference.train()
    expected=deepcopy(reference.model.state_dict())
    resumed=make();resumed.load_checkpoint(tmp_path/'out/checkpoint_step_2.pt');resumed.train()
    for name,value in expected.items():
        torch.testing.assert_close(value,resumed.model.state_dict()[name],atol=0,rtol=0)
