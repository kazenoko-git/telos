"""
Empirical Attention Weight Distribution Analysis:
Compares 100M AR vs 100M COROSred (alpha=0.50) vs 50M COROSred
on Task 12 prompt in pure causal mode.
Measures attention mass allocated to:
1. Docstring region (e.g. '(val * 13 + 169) % 10007')
2. Immediate last 3-5 tokens (recency sink)
3. Signature / BOS tokens
"""

import sys
import json
import math
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from telos.eval.runner import load_model_from_checkpoint
from telos.data.tokenizer import load_tokenizer
from telos.models.components import apply_rope

def extract_layer_attention_weights(model, token_ids: list[int]):
    """
    Runs a forward pass and extracts the explicit attention weights (B, H, T, T)
    for all layers using the model's Q, K projections and RoPE.
    """
    model.eval()
    device = next(model.parameters()).device
    x = torch.tensor([token_ids], dtype=torch.long, device=device)
    batch, seq_len = x.shape

    h = model.tok_embeddings(x)
    h = model.dropout(h)
    cos, sin = model.rope(h, seq_len)

    all_layer_attentions = []

    for idx, layer in enumerate(model.layers):
        norm_h = layer.attn_norm(h)
        attn_mod = layer.attn

        q = attn_mod.q_proj(norm_h).view(batch, seq_len, attn_mod.n_heads, attn_mod.head_dim).permute(0, 2, 1, 3)
        k = attn_mod.k_proj(norm_h).view(batch, seq_len, attn_mod.n_kv_heads, attn_mod.head_dim).permute(0, 2, 1, 3)
        v = attn_mod.v_proj(norm_h).view(batch, seq_len, attn_mod.n_kv_heads, attn_mod.head_dim).permute(0, 2, 1, 3)

        q, k = apply_rope(q, k, cos, sin)
        if attn_mod.num_queries_per_kv > 1:
            k = k.repeat_interleave(attn_mod.num_queries_per_kv, dim=1)
            v = v.repeat_interleave(attn_mod.num_queries_per_kv, dim=1)

        # Compute explicit causal attention matrix
        # (B, H, T, d_k) x (B, H, d_k, T) -> (B, H, T, T)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(attn_mod.head_dim)
        
        # Causal mask: upper triangle is -inf
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)
        scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)
        
        attn_weights = F.softmax(scores, dim=-1)  # (1, n_heads, seq_len, seq_len)
        all_layer_attentions.append(attn_weights[0].detach().cpu())

        # Forward through layer to continue activations
        h = layer(h, cos, sin, mask_override=True)

    return all_layer_attentions

def analyze_task_attention():
    tok = load_tokenizer(str(PROJECT_ROOT / "configs/tokenizer_mac.json"))
    with open(PROJECT_ROOT / "evals/benchmarks/private_unseen_suite.json") as f:
        suite = json.load(f)

    # Task 12 prompt
    task = suite[12]
    prompt = task["prompt"]
    token_ids = tok.encode(prompt).ids
    
    print("=" * 80)
    print("PROMPT FOR TASK 12:")
    print(prompt.strip())
    print("=" * 80)
    print(f"Total prompt tokens: {len(token_ids)}")
    
    # Identify token index regions
    # Find where the docstring starts and ends
    tokens_text = [tok.decode([tid]) for tid in token_ids]
    
    docstring_start = None
    docstring_end = None
    for i, t in enumerate(tokens_text):
        if '"""' in t or "Computes" in t:
            if docstring_start is None:
                docstring_start = i
        if '"""' in t and docstring_start is not None and i > docstring_start:
            docstring_end = i + 1
            break
            
    if docstring_start is None:
        docstring_start = 12
    if docstring_end is None:
        docstring_end = len(token_ids) - 1

    last_n = 4  # last 4 tokens
    recency_start = len(token_ids) - last_n
    
    print(f"Docstring token range: [{docstring_start} : {docstring_end}] -> {repr(tok.decode(token_ids[docstring_start:docstring_end]))}")
    print(f"Recency token range (last {last_n}): [{recency_start} : {len(token_ids)}] -> {repr(tok.decode(token_ids[recency_start:]))}")
    print("-" * 80)

    models = [
        ("100M AR (5B)", PROJECT_ROOT / "checkpoints/ar/100m_5b/checkpoint_final.pt"),
        ("100M COROSred (alpha=0.50)", PROJECT_ROOT / "checkpoints/corosred/unified/100m_5b/checkpoint_final.pt"),
        ("50M COROSred (Lightning)", PROJECT_ROOT / "checkpoints/corosred/50m_lightning/checkpoint_final.pt"),
    ]

    for model_name, ckpt_path in models:
        if not ckpt_path.exists():
            print(f"Skipping {model_name}, not found.")
            continue

        model, _, _ = load_model_from_checkpoint(ckpt_path)
        attn_by_layer = extract_layer_attention_weights(model, token_ids)
        num_layers = len(attn_by_layer)
        
        print(f"\n>>> Model: {model_name} (Layers: {num_layers})")
        print(f"{'Layer':<10} | {'Docstring Mass (%)':<20} | {'Last 4 Toks Recency (%)':<24} | {'BOS / Prefix (%)':<18}")
        print("-" * 75)

        # Inspect the last 4 layers
        last_layers = list(range(max(0, num_layers - 4), num_layers))
        
        for l_idx in last_layers:
            attn = attn_by_layer[l_idx]  # (n_heads, seq_len, seq_len)
            
            # Query at the very last token (generating the first token of function body)
            query_idx = len(token_ids) - 1
            query_attn = attn[:, query_idx, :]  # (n_heads, seq_len)
            mean_query_attn = query_attn.mean(dim=0)  # average across heads: (seq_len,)
            
            doc_mass = mean_query_attn[docstring_start:docstring_end].sum().item() * 100.0
            rec_mass = mean_query_attn[recency_start:].sum().item() * 100.0
            bos_mass = mean_query_attn[:docstring_start].sum().item() * 100.0
            
            print(f"Layer {l_idx:<4} | {doc_mass:<20.2f}% | {rec_mass:<24.2f}% | {bos_mass:<18.2f}%")

if __name__ == "__main__":
    analyze_task_attention()
