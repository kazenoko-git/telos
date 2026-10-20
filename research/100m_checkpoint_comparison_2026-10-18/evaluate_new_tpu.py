"""Score the completed cadence-1 TPU endpoint using the preserved fixture."""
import json
import time
import torch
import evaluate as baseline

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
torch.manual_seed(42)
label = '100m_r50_tpu_v5e8_seed42_cad1'
path = baseline.ROOT / 'checkpoints/corosred/unified' / label / 'checkpoint_final.pt'
fixture = torch.load(baseline.OUT / 'fixture.pt', weights_only=True)
started = time.perf_counter()
model, metadata = baseline.model_metadata(path)
print('Loaded final checkpoint at step', metadata['step'], flush=True)
result = baseline.score(model, fixture['tokens'], fixture['infill_mask'])
result['metadata'] = metadata
result['seconds'] = time.perf_counter() - started
report = {
    'hf_revision': 'c93b4446a5d939391f652619265a58299fb80d6c',
    'label': label,
    'fixture_sha256': baseline.digest_file(baseline.OUT / 'fixture.pt'),
    'evaluator_sha256': baseline.digest_file(baseline.__file__),
    'device': 'cpu', 'dtype': 'float32', 'torch': str(torch.__version__),
    'result': result,
}
output = baseline.OUT / 'new_tpu_results.json'
output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: result[k] for k in ['causal', 'infill', 'lrh_correctness_auroc', 'seconds']}, indent=2), flush=True)
print(output, flush=True)
