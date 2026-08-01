"""standalone high-level inference api for télos Masked Diffusion Language Model.

provides TelosModel.from_pretrained() API. since standard transformers.generate()
is hardcoded for causal autoregressive models, this class encapsulates the iterative
denoising sampler directly alongside the weights and tokenizer.
"""

from pathlib import Path
import json
import torch
from tokenizers import Tokenizer
from telos.model.transformer import TelosTransformer, TelosConfig
from telos.diffusion.sampler import MDLMSampler


class TelosModel:
    """high-level standalone inference wrapper for télos models."""

    def __init__(self, model: TelosTransformer, tokenizer: Tokenizer, config: TelosConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_pretrained(cls, model_path_or_id: str | Path = "checkpoints") -> "TelosModel":
        """Loads model weights, tokenizer, and config from local folder or HF Hub."""
        model_path = Path(model_path_or_id)

        # 1. Locate checkpoint weights file
        weights_file = None
        for name in ["checkpoint_final.pt", "model.pt", "model.safetensors"]:
            if (model_path / name).exists():
                weights_file = model_path / name
                break
        if weights_file is None and model_path.is_file():
            weights_file = model_path
        elif weights_file is None:
            pt_files = list(model_path.glob("*.pt"))
            if pt_files:
                weights_file = pt_files[-1]

        assert weights_file is not None, f"No model weights (.pt / .safetensors) found in {model_path}"

        # Load weights state_dict and optional embedded config
        state_dict = torch.load(weights_file, map_location="cpu")
        embedded_cfg = None
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            embedded_cfg = state_dict.get("config")
            state_dict = state_dict["model_state_dict"]

        # 2. Locate / parse Config
        if (model_path / "config.json").exists():
            with open(model_path / "config.json", "r") as f:
                cfg_dict = json.load(f)
            config = TelosConfig(**cfg_dict)
        elif embedded_cfg and "model" in embedded_cfg:
            config = TelosConfig(**embedded_cfg["model"])
        elif Path("configs/phase_a.yaml").exists():
            import yaml
            with open("configs/phase_a.yaml", "r") as f:
                cfg_dict = yaml.safe_load(f)["model"]
            config = TelosConfig(**cfg_dict)
        else:
            config = TelosConfig()

        # 3. Locate Tokenizer
        tokenizer_file = None
        for tok_path in [model_path / "tokenizer.json", Path("configs/tokenizer.json")]:
            if tok_path.exists():
                tokenizer_file = tok_path
                break

        assert tokenizer_file is not None, f"Tokenizer file (tokenizer.json) missing"
        tokenizer = Tokenizer.from_file(str(tokenizer_file))

        # Convert legacy qkv_proj state_dict keys to GQA q_proj/k_proj/v_proj format
        new_state_dict = {}
        for key, val in state_dict.items():
            if "attn.qkv_proj.weight" in key:
                prefix = key.rsplit("qkv_proj.weight", 1)[0]
                d_model = val.shape[0] // 3
                new_state_dict[prefix + "q_proj.weight"] = val[:d_model]
                new_state_dict[prefix + "k_proj.weight"] = val[d_model:2*d_model]
                new_state_dict[prefix + "v_proj.weight"] = val[2*d_model:]
            else:
                new_state_dict[key] = val

        # 4. Instantiate model and load state_dict
        model = TelosTransformer(config)
        model.load_state_dict(new_state_dict, strict=False)
        return cls(model, tokenizer, config)

    @torch.no_grad()
    def complete(
        self,
        prompt: str,
        max_tokens: int = 128,
        num_steps: int = 64,
        temperature: float = 0.8,
        schedule: str = "linear"
    ) -> str:
        """completes code given a prompt using masked diffusion iterative unmasking.

        args:
            prompt: input string (e.g., function signature + docstring).
            max_tokens: total target sequence length.
            num_steps: denoising steps (16-128).
            temperature: sampling temperature.
            schedule: unmasking schedule ("linear" or "cosine").

        returns:
            completion: generated Python code text.
        """
        encoded = self.tokenizer.encode(prompt)
        prompt_ids = torch.tensor([encoded.ids], dtype=torch.long, device=self.device)

        sampler = MDLMSampler(
            self.model,
            mask_token_id=1,
            num_steps=num_steps,
            temperature=temperature,
            schedule=schedule
        )

        sampled_ids = sampler.sample(seq_len=max_tokens, prompt_ids=prompt_ids, device=self.device)
        return self.tokenizer.decode(sampled_ids[0].tolist())
