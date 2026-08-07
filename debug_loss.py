import torch
from telos.hub.inference import TelosModel

model_obj = TelosModel.from_pretrained("checkpoints/phase_c_tpu_125m/checkpoint_tpu_125M_final_step_238.pt")
model = model_obj.model

print("Vocab size:", model.config.vocab_size)
print("Layers:", model.config.n_layers)

emb_w = model.tok_embeddings.weight
print("Embedding Var:", emb_w.var().item(), "Mean:", emb_w.mean().item())
print("Has NaNs:", torch.isnan(emb_w).any().item())

ckpt = torch.load("checkpoints/phase_c_tpu_125m/checkpoint_tpu_125M_final_step_238.pt", map_location="cpu")
print("Loss in checkpoint:", ckpt["loss"])
print("Step in checkpoint:", ckpt["step"])

