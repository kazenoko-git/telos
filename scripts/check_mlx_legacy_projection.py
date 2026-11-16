"""Reproduce BF16 failures using the unchanged historical tiled method."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import mlx.core as mx
from telos.training.mlx_losses import cross_entropy
from telos.training.mlx_vocab_loss import tiled_projection_cross_entropy

commit = 'f5581469adba661eee8e06d534de563ae7b0a2ae'
source = subprocess.check_output(['git', 'show', f'{commit}:telos/models/mlx_transformer.py'], text=True)
model = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'MLXTelosTransformer')
method = next(n for n in model.body if isinstance(n, ast.FunctionDef) and n.name == 'output_cross_entropy')
scope = {'mx': mx}
exec(compile(ast.Module(body=[method], type_ignores=[]), '<historical tiled method>', 'exec'), scope)
historical = scope['output_cross_entropy']
targets = mx.array([[0]])
input_ids = mx.zeros((1, 1), dtype=mx.int32)

def old(hidden, weight):
    projection = SimpleNamespace(tied_embeddings=True, d_model=2, vocab_size=257,
                                 emb=SimpleNamespace(weight=weight),
                                 hidden_states=lambda *args, **kwargs: hidden)
    return historical(projection, input_ids, targets, chunk_size=32)

cases = []
for name, scale, target_logit in (('rounded_target', 1.0625, 37.75), ('confident_target', 1., 14.)):
    hidden = mx.array([[[scale, 0.]]], dtype=mx.bfloat16)
    weight = mx.zeros((257, 2), dtype=mx.bfloat16).at[0, 0].add(target_logit)
    functions = {
        'full': lambda w: mx.sum(cross_entropy(hidden @ w.T, targets)),
        'historical': lambda w: mx.sum(old(hidden, w)),
        'corrected': lambda w: mx.sum(tiled_projection_cross_entropy(hidden, w, targets, chunk_size=32)),
    }
    row = {'name': name}
    for mode, function in functions.items():
        loss, gradient = mx.value_and_grad(function)(weight)
        row[mode] = {'loss': loss.item(), 'target_weight_gradient': gradient[0, 0].item()}
    cases.append(row)
result = dict(historical_commit=commit, source_sha256=hashlib.sha256(source.encode()).hexdigest(),
              mlx_version=mx.__version__, cases=cases)
Path('reports/mlx_fusion_2026-11-16/legacy_projection_gate.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
