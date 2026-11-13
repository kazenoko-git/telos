import json
from pathlib import Path
import pytest
import torch
from telos.data.tokenizer import load_tokenizer
from telos.models import TelosConfig, TelosTransformer
from telos.posttrain.trainer import posttrain


@pytest.mark.parametrize('method', ['sft', 'dpo'])
@pytest.mark.parametrize('hardware', ['cpu', 'mlx'])
def test_posttrain_resume_replays(tmp_path, method, hardware):
    if hardware == 'mlx':
        from telos.training.core import metal_usable
        if not metal_usable():
            pytest.skip('Metal required')
    tok = load_tokenizer()
    cfg = TelosConfig(vocab_size=len(tok.get_vocab()), d_model=16, n_layers=1,
                      n_heads=2, n_kv_heads=2, max_seq_len=16, is_causal=True)
    model = TelosTransformer(cfg)
    source = tmp_path / 'source.pt'
    torch.save({'model_state_dict': model.state_dict(), 'config': {'paradigm': 'ar', 'model':
        dict(vocab_size=cfg.vocab_size,d_model=16,n_layers=1,n_heads=2,n_kv_heads=2,max_seq_len=16,is_causal=True)}}, source)
    row = {'prompt': '1+1?', 'response': '2'} if method == 'sft' else {'prompt': '1+1?', 'chosen': '2', 'rejected': '4'}
    data = tmp_path / 'data.jsonl'
    rows = [row, dict(row, prompt='2+2?'), dict(row, prompt='3+3?')]
    data.write_text(''.join(json.dumps(item)+'\n' for item in rows))
    options = dict(method=method,hardware=hardware,devices=1,max_steps=4,batch_size=1,max_length=8,save_every=2)
    first = posttrain(source,data,tmp_path/'first',**options)
    suffix = '.resume.npz' if hardware == 'mlx' else '.resume.pt'
    second = posttrain(source,data,tmp_path/'second',resume=tmp_path/'first'/('checkpoint_step_2'+suffix),**options)
    if hardware == 'cpu':
        a,b = [torch.load(p,weights_only=True)['model_state_dict'] for p in (first,second)]
        for key in a:
            assert torch.equal(a[key],b[key])
    else:
        import mlx.core as mx
        a,b = mx.load(str(first)),mx.load(str(second))
        for key in a:
            assert mx.allclose(a[key],b[key],rtol=1e-5,atol=1e-6).item()
