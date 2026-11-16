"""CPU simulation of XLA control flow; never presented as TPU verification."""
import json
import sys
import types
import pytest
import torch
from telos.data.tokenizer import load_tokenizer
from telos.models import TelosTransformer
from telos.posttrain.trainer import posttrain, _xla_posttrain_worker


@pytest.mark.parametrize('method', ['sft', 'dpo'])
@pytest.mark.parametrize('finetune', ['full', 'lora', 'qlora'])
def test_xla_worker_fixed_shapes_and_logging_collectives(tmp_path, monkeypatch, method, finetune):
    tok = load_tokenizer()
    model = TelosTransformer(vocab_size=len(tok.get_vocab()), d_model=32, n_layers=1,
                            n_heads=2, max_seq_len=32, is_causal=True)
    source = tmp_path/'base.pt'
    torch.save({'model_state_dict': model.state_dict(), 'config': {'paradigm':'ar',
                'model':vars(model.config)}},source)
    row = {'prompt':'Q?', 'response':'A'} if method=='sft' else {'prompt':'Q?', 'chosen':'A', 'rejected':'B'}
    data = tmp_path/'data.jsonl'
    data.write_text(json.dumps(row)+'\n'+json.dumps(dict(row,prompt='A longer question?')))
    xla, core, xm, xr = [types.ModuleType(name) for name in
        ('torch_xla','torch_xla.core','torch_xla.core.xla_model','torch_xla.runtime')]
    xla.core, xla.runtime, core.xla_model = core, xr, xm
    xr.global_ordinal = lambda:0
    xr.global_runtime_device_count = lambda:1
    xr.world_size = lambda:1
    xm.set_rng_state = lambda seed:None
    xm.get_rng_state = lambda:42
    xm.xla_device = lambda:torch.device('cpu')
    xm.mark_step = lambda:None
    xm.rendezvous = lambda name:None
    xm.mesh_reduce = lambda name,value,fn:fn([value])
    xm.reduce_gradients = lambda optimizer:None
    reductions = []
    xm.REDUCE_SUM = 'sum'
    def reduce(operation,value):
        reductions.append(value.shape)
        return value
    xm.all_reduce = reduce
    for module in (xla,core,xm,xr):
        monkeypatch.setitem(sys.modules,module.__name__,module)
    import telos.posttrain.trainer as trainer
    monkeypatch.setattr(trainer,'_posttrain_xla',lambda arguments:_xla_posttrain_worker(0,arguments))
    widths=[]
    original=trainer._collate
    def collate(*args,**kwargs):
        assert kwargs.get('fixed') is True
        result=original(*args,**kwargs)
        widths.append(result[0].shape[1])
        return result
    monkeypatch.setattr(trainer,'_collate',collate)
    posttrain(source,data,tmp_path/'out',method=method,hardware='xla',finetune=finetune,
              precision='float32',max_steps=3,batch_size=1,max_length=16,save_every=0,lora_rank=4)
    assert widths==[16,16,16]
    assert len(reductions)==2  # First and last update, not every update.
    assert (tmp_path/'out/checkpoint_final.pt').is_file()
