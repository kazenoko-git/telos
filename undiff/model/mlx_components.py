import math
import mlx.core as mx
import mlx.nn as nn
import yaml

class MLXRMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((d_model,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, self.weight, self.eps)

class MLXSwiGLU(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        hidden = int(d_model * 8 / 3)
        hidden = ((hidden + 63) // 64) * 64
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(d_model, hidden, bias=False)
        self.w3 = nn.Linear(hidden, d_model, bias=False)

    def __call__(self, x):
        return self.w3(nn.silu(self.w1(x)) * self.w2(x))

class MLXBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        self.norm1 = MLXRMSNorm(d_model)
        self.norm2 = MLXRMSNorm(d_model)
        self.head_dim = d_model // n_heads
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.kv_groups = n_heads // n_kv_heads

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.mlp = MLXSwiGLU(d_model)

    def __call__(self, x):
        B, T, D = x.shape
        h = self.norm1(x)

        q = self.q_proj(h).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(h).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(h).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Apply hardware-accelerated RoPE Metal kernel to Q and K
        q = mx.fast.rope(q, self.head_dim, traditional=False, base=10000.0, scale=1.0, offset=0)
        k = mx.fast.rope(k, self.head_dim, traditional=False, base=10000.0, scale=1.0, offset=0)

        scale = 1.0 / (self.head_dim ** 0.5)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, D)

        x = x + self.out(out)
        x = x + self.mlp(self.norm2(x))
        return x

class MLXTelosTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        n_kv_heads: int,
        tied_embeddings: bool = True,
        use_grad_checkpoint: bool = False,
        **kwargs
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.tied_embeddings = tied_embeddings
        self.use_grad_checkpoint = use_grad_checkpoint
        self.emb = nn.Embedding(vocab_size, d_model)
        self.layers = [MLXBlock(d_model, n_heads, n_kv_heads) for _ in range(n_layers)]
        self.norm = MLXRMSNorm(d_model)
        if not tied_embeddings:
            self.head = nn.Linear(d_model, vocab_size, bias=False)

    def hidden_states(self, x):
        x = self.emb(x)
        for layer in self.layers:
            if self.use_grad_checkpoint:
                x = mx.checkpoint(layer)(x)
            else:
                x = layer(x)
        return self.norm(x)

    def logits_from_hidden(self, hidden):
        if self.tied_embeddings:
            return self.emb.as_linear(hidden)
        return self.head(hidden)

    def __call__(self, x):
        return self.logits_from_hidden(self.hidden_states(x))

def get_upscaled_layer_mapping(src_layers, tgt_layers):
    mapping = []
    ratio = src_layers / tgt_layers
    for i in range(tgt_layers):
        mapping.append(min(int(i * ratio), src_layers - 1))
    return mapping

def pad_weight(src_weight, tgt_shape, is_norm=False):
    """
    Pads weight tensors from source shape to target shape.
    For RMSNorm (is_norm=True), scales the active elements by sqrt(d_old / d_new)
    so that activation variance across the expanded dimension remains invariant.
    """
    if src_weight.shape == tgt_shape: return src_weight
    if is_norm and len(tgt_shape) == 1:
        scale = math.sqrt(src_weight.shape[0] / tgt_shape[0])
        padded = mx.ones(tgt_shape, dtype=src_weight.dtype)
        padded[:src_weight.shape[0]] = src_weight * scale
        return padded
    padded = mx.zeros(tgt_shape, dtype=src_weight.dtype)
    if len(tgt_shape) == 1:
        padded[:src_weight.shape[0]] = src_weight
    elif len(tgt_shape) == 2:
        padded[:src_weight.shape[0], :src_weight.shape[1]] = src_weight
    return padded

def load_upscaled_weights(tgt_model, tgt_cfg, src_ckpt_path, src_cfg_path):
    print(f"  [Upscaling] Loading source config: {src_cfg_path}")
    with open(src_cfg_path, "r") as f:
        src_cfg = yaml.safe_load(f)["model"]
        
    print(f"  [Upscaling] Loading source weights: {src_ckpt_path}")
    src_weights = mx.load(src_ckpt_path)
    tgt_weights = {}
    
    tgt_weights["emb.weight"] = pad_weight(src_weights["emb.weight"], (tgt_cfg["vocab_size"], tgt_cfg["d_model"]))
    if not tgt_cfg.get("tied_embeddings", True):
        head_w = src_weights.get("head.weight", src_weights["emb.weight"])
        tgt_weights["head.weight"] = pad_weight(head_w, (tgt_cfg["vocab_size"], tgt_cfg["d_model"]))
    tgt_weights["norm.weight"] = pad_weight(src_weights["norm.weight"], (tgt_cfg["d_model"],), is_norm=True)
    
    layer_map = get_upscaled_layer_mapping(src_cfg["n_layers"], tgt_cfg["n_layers"])
    print(f"  [Upscaling] Depth mapping (Target <- Source): {layer_map}")
    
    for tgt_i, src_i in enumerate(layer_map):
        prefix_src = f"layers.{src_i}."
        prefix_tgt = f"layers.{tgt_i}."
        
        tgt_weights[prefix_tgt + "norm1.weight"] = pad_weight(src_weights[prefix_src + "norm1.weight"], (tgt_cfg["d_model"],), is_norm=True)
        tgt_weights[prefix_tgt + "norm2.weight"] = pad_weight(src_weights[prefix_src + "norm2.weight"], (tgt_cfg["d_model"],), is_norm=True)
        
        tgt_weights[prefix_tgt + "q_proj.weight"] = pad_weight(src_weights[prefix_src + "q_proj.weight"], (tgt_cfg["n_heads"] * (tgt_cfg["d_model"] // tgt_cfg["n_heads"]), tgt_cfg["d_model"]))
        tgt_weights[prefix_tgt + "k_proj.weight"] = pad_weight(src_weights[prefix_src + "k_proj.weight"], (tgt_cfg["n_kv_heads"] * (tgt_cfg["d_model"] // tgt_cfg["n_heads"]), tgt_cfg["d_model"]))
        tgt_weights[prefix_tgt + "v_proj.weight"] = pad_weight(src_weights[prefix_src + "v_proj.weight"], (tgt_cfg["n_kv_heads"] * (tgt_cfg["d_model"] // tgt_cfg["n_heads"]), tgt_cfg["d_model"]))
        tgt_weights[prefix_tgt + "out.weight"] = pad_weight(src_weights[prefix_src + "out.weight"], (tgt_cfg["d_model"], tgt_cfg["d_model"]))
        
        hidden_tgt = int(tgt_cfg["d_model"] * 8 / 3)
        hidden_tgt = ((hidden_tgt + 63) // 64) * 64
        hidden_src = int(src_cfg["d_model"] * 8 / 3)
        hidden_src = ((hidden_src + 63) // 64) * 64
        
        tgt_weights[prefix_tgt + "mlp.w1.weight"] = pad_weight(src_weights[prefix_src + "mlp.w1.weight"], (hidden_tgt, tgt_cfg["d_model"]))
        tgt_weights[prefix_tgt + "mlp.w2.weight"] = pad_weight(src_weights[prefix_src + "mlp.w2.weight"], (hidden_tgt, tgt_cfg["d_model"]))
        tgt_weights[prefix_tgt + "mlp.w3.weight"] = pad_weight(src_weights[prefix_src + "mlp.w3.weight"], (tgt_cfg["d_model"], hidden_tgt))

    tgt_model.load_weights(list(tgt_weights.items()), strict=False)
    print("  [Upscaling] Success: Initialized model with upscaled weights.")
