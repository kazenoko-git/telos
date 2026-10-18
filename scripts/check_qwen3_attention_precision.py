#!/usr/bin/env python3
"""Exploratory FP32 output-head check after the fixed BF16 attention experiment.

Reuses all 48 paired examples, only query_only/zero. Backbone stays BF16;
the tied output projection is recomputed in FP32 to resolve rounded score ties.
This is not a new held-out experiment and is not full-FP32 attention validation.
"""
import hashlib
import json
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm.utils import load_model
import numpy as np
from tokenizers import Tokenizer

from probe_qwen3_attention import CACHE, REVISION, ROOT, attention_mask


def main():
    folder = ROOT / 'research/qwen3_attention_probe_2026-10-18/full'
    destination = folder / 'fp32_head_check.json'
    if destination.exists():
        raise FileExistsError(destination)
    cases = [c for c in json.loads((folder/'cases.json').read_text()) if c['pair']]
    old = {(r['id'], r['readout']): r for line in (folder/'results.jsonl').read_text().splitlines()
           if (r := json.loads(line))['mode'] == 'query_only' and r['placeholder'] == 'zero'}
    model_dir = CACHE / 'models--Qwen--Qwen3-4B/snapshots' / REVISION
    tok = Tokenizer.from_file(str(model_dir/'tokenizer.json'))
    model, config = load_model(model_dir, strict=True)
    model.eval()
    model.freeze()
    assert config['tie_word_embeddings'] and not tree_flatten(model.trainable_parameters())
    weight = model.model.embed_tokens.weight.astype(mx.float32)
    mx.eval(weight)
    placeholder = tok.encode('_', add_special_tokens=False).ids[0]
    rows = []
    for c in cases:
        hole = c['hole']
        ids = list(c['ids'])
        ids[hole] = placeholder
        h = model.model.embed_tokens(mx.array([ids]))
        keep = mx.array([j != hole for j in range(len(ids))])[None, :, None]
        h = mx.where(keep, h, mx.zeros_like(h))
        for index, layer in enumerate(model.model.layers):
            mask = mx.array(attention_mask(len(ids), hole, 'query_only', index, len(model.model.layers)), dtype=h.dtype)
            h = layer(h, mask=mask, cache=None)
        selected = model.model.norm(h[:, [hole-1, hole], :])
        bf16 = model.model.embed_tokens.as_linear(selected)
        fp32 = selected.astype(mx.float32) @ weight.T
        mx.eval(bf16, fp32)
        bf16 = np.asarray(bf16.astype(mx.float32))[0]
        fp32 = np.asarray(fp32)[0]
        for i, readout in enumerate(('shifted', 'same_position')):
            target, alt = c['target'], c['alternative']
            previous = old[c['id'], readout]
            bf_margin = float(bf16[i, target]-bf16[i, alt])
            assert bf_margin == previous['candidate_margin'], (c['id'], readout, bf_margin, previous['candidate_margin'])
            pred = int(fp32[i].argmax())
            rows.append(dict(id=c['id'], pair=c['pair'], readout=readout,
                target_text=c['target_text'], bf16_margin=bf_margin,
                fp32_margin=float(fp32[i,target]-fp32[i,alt]), prediction=pred,
                prediction_text=tok.decode([pred]), correct=pred == target))
    summary = {}
    for readout in ('shifted', 'same_position'):
        pairs = {}
        for r in rows:
            if r['readout'] == readout:
                pairs.setdefault(r['pair'], []).append(r)
        summary[readout] = dict(pairs=len(pairs),
            both_top1=sum(all(r['correct'] for r in p) for p in pairs.values()),
            both_prefer_required=sum(all(r['fp32_margin']>0 for r in p) for p in pairs.values()),
            top1=sum(r['correct'] for r in rows if r['readout'] == readout))
    result = dict(status='complete', revision=REVISION, exploratory=True,
        reason='Resolve BF16 output-score ties in query_only condition',
        backbone_precision='BF16', output_head_precision='FP32',
        bf16_candidate_margins_reproduced_exactly=True,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        summary=summary, rows=rows)
    destination.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summary,indent=2), flush=True)


if __name__ == '__main__':
    main()
