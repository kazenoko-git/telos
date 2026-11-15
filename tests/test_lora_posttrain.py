"""Exercise real native adapter training, exports and full interrupted resume."""
import json
import pytest
import torch
from telos.data.tokenizer import load_tokenizer
from telos.eval.runner import load_model_from_checkpoint
from telos.models import TelosConfig, TelosTransformer
from telos.posttrain.trainer import posttrain


@pytest.mark.parametrize('hardware', ['cpu', 'mlx'])
@pytest.mark.parametrize('finetune', ['lora', 'qlora'])
@pytest.mark.parametrize('method', ['sft', 'dpo'])
def test_adapter_posttrain_and_resume(tmp_path, hardware, finetune, method):
    if hardware == 'mlx':
        from telos.training.core import metal_usable
        if not metal_usable():
            pytest.skip('Metal required')
    tok = load_tokenizer()
    cfg = TelosConfig(vocab_size=len(tok.get_vocab()), d_model=64, n_layers=1,
                      n_heads=2, n_kv_heads=2, max_seq_len=16, is_causal=True)
    torch.manual_seed(5)
    source_model = TelosTransformer(cfg)
    source = tmp_path/'base.pt'
    torch.save({'model_state_dict': source_model.state_dict(), 'config': {'paradigm':'ar',
                'model': vars(cfg)}}, source)
    row = {'prompt':'1+1?', 'response':'2'} if method=='sft' else {
        'prompt':'1+1?', 'chosen':'2', 'rejected':'4'}
    data = tmp_path/'data.jsonl'
    data.write_text('\n'.join(json.dumps(dict(row, prompt=prompt)) for prompt in ['1+1?', '2+2?']))
    opts = dict(method=method, hardware=hardware, finetune=finetune, lora_rank=4,
                max_steps=3, batch_size=1, max_length=12, save_every=1, learning_rate=.002,
                lora_dropout=.1, precision="bf16" if hardware == "mlx" else "auto")
    final = posttrain(source, data, tmp_path/'first', **opts)
    if hardware == 'mlx':
        import mlx.core as mx
        expected_mlx = mx.load(str(final))
        mx.eval(expected_mlx)
    suffix = '.resume.npz' if hardware == 'mlx' else '.resume.pt'
    resumed = posttrain(source, data, (tmp_path/'first' if hardware == 'mlx' else tmp_path/'resumed'),
        resume=tmp_path/'first'/('checkpoint_step_1'+suffix), **opts)
    loaded, backend, _ = load_model_from_checkpoint(final)
    assert backend == ('mlx' if hardware=='mlx' else 'pytorch')
    if hardware == 'cpu':
        a,b = [torch.load(path,weights_only=True)['model_state_dict'] for path in (final,resumed)]
        for key in a:
            torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
        from telos.posttrain.adapters import load_adapter
        restored, _, _ = load_model_from_checkpoint(source)
        load_adapter(restored, final.with_suffix('.adapter.pt'), base_checkpoint=source)
        restored.eval(); loaded.eval()
        tokens = torch.tensor([[4, 5, 6, 7]])
        torch.testing.assert_close(restored(tokens), loaded(tokens), rtol=1e-4, atol=1e-6)
        adapter = torch.load(final.with_suffix('.adapter.pt'),weights_only=True)
        assert adapter['adapter_state_dict']
        assert all(key.endswith(('lora_a','lora_b')) for key in adapter['adapter_state_dict'])
        assert any(value.count_nonzero() for key,value in adapter['adapter_state_dict'].items() if key.endswith('lora_b'))
    else:
        import mlx.core as mx
        a,b = expected_mlx,mx.load(str(resumed))
        for key in a:
            assert mx.allclose(a[key],b[key],rtol=1e-5,atol=1e-6).item(), key
        from telos.models import MLXTelosTransformer
        from telos.posttrain.trainer import _mlx_parameters_from_pytorch
        from telos.posttrain.mlx_adapters import load_adapter, merged_weights
        restored = MLXTelosTransformer(vocab_size=cfg.vocab_size, d_model=64, n_layers=1,
                                       n_heads=2, n_kv_heads=2, is_causal=True, precision='float32')
        restored.load_weights(list(_mlx_parameters_from_pytorch(source_model.state_dict()).items()), strict=False)
        restored.set_dtype(mx.bfloat16)
        load_adapter(restored, final.with_suffix('.adapter.safetensors'), base_checkpoint=source)
        restored_export = dict(merged_weights(restored))
        for key in a:
            assert mx.array_equal(a[key], restored_export[key]).item(), key
        adapter = mx.load(str(final.with_suffix('.adapter.safetensors')))
        assert adapter and all(key.endswith(('lora_a','lora_b')) for key in adapter)
