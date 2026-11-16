from copy import deepcopy
import pytest
import torch
from telos.models import TelosTransformer
from telos.models.vocab_loss import tiled_cross_entropy


@pytest.mark.parametrize('chunk', [7, 32, 128])
def test_tiled_ce_loss_and_gradients(chunk):
    torch.manual_seed(23)
    hidden=torch.randn(2,7,16,requires_grad=True)
    weight=torch.randn(67,16,requires_grad=True)
    labels=torch.randint(0,67,(2,7))
    expected=torch.nn.functional.cross_entropy((hidden@weight.T).reshape(-1,67),labels.flatten(),reduction='none').reshape(2,7)
    actual=tiled_cross_entropy(hidden,weight,labels,chunk)
    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=2e-6)
    gradients=torch.autograd.grad(expected.mean(),(hidden,weight),retain_graph=True)
    actual_gradients=torch.autograd.grad(actual.mean(),(hidden,weight))
    for a,b in zip(gradients,actual_gradients):
        torch.testing.assert_close(a,b,rtol=2e-5,atol=1e-6)


def test_model_next_token_projection_and_tiled_gradients():
    model=TelosTransformer(vocab_size=67,d_model=32,n_layers=1,n_heads=2,is_causal=True)
    tokens=torch.randint(0,67,(2,9))
    logits=model(tokens)
    trimmed=model(tokens,next_token_only=True)
    torch.testing.assert_close(trimmed,logits[:,:-1])
    loss=model(tokens,next_token_only=True,loss_targets=tokens[:,1:],vocab_chunk_size=13)
    expected=torch.nn.functional.cross_entropy(logits[:,:-1].reshape(-1,67),tokens[:,1:].flatten(),reduction='none')
    torch.testing.assert_close(loss.flatten(),expected)
    loss.mean().backward()
    assert model.tok_embeddings.weight.grad is not None
