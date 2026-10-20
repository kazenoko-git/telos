"""Evidence inventory: successful TPU versus eligible older unified runs."""
from pathlib import Path
import json
import hashlib
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
LABELS = ['100m_r50_tpu_v5e8_seed42_cad1', '100m_5b', '100m_tpu_2',
          '100m_5b_alpha075', '100m_schedstep1', '50m_python',
          '50m_tpu_schedstep1', '50m_tpu_lrcadence_schedstep1']

def clean(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return {'tensor_shape': list(value.shape), 'dtype': str(value.dtype)}
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return str(type(value))

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(8 * 1024**2):
            h.update(block)
    return h.hexdigest()

records = {}
for label in LABELS:
    directory = ROOT / 'checkpoints/corosred/unified' / label
    path = directory / 'checkpoint_final.pt'
    c = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    config = c['config']
    cr = config.get('corosred', {})
    assert cr.get('unified') and cr.get('phase') == 'unified', label
    state = c['model_state_dict']
    record = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest(path),
              'checkpoint_keys': list(c), 'config': clean(config),
              'non_weight_state': clean({k: v for k, v in c.items() if k not in ['model_state_dict', 'optimizer_state_dict', 'config']}),
              'optimizer_saved': 'optimizer_state_dict' in c,
              'parameter_dtypes': sorted({str(v.dtype) for v in state.values()}),
              'embedding_output_equal': torch.equal(state['tok_embeddings.weight'], state['output_projection.weight']),
              'sidecars': {name: json.loads((directory / name).read_text()) for name in
                  ['runtime.json', 'launch_command.json', 'dataset_source.json'] if (directory / name).exists()}}
    records[label] = record
    print(label, json.dumps(record['non_weight_state']), flush=True)
    del c, state
(OUT / 'checkpoint_inventory.json').write_text(json.dumps(records, indent=2) + '\n')
