"""standalone high-level inference api for télos Masked Diffusion Language Model.

provides TelosModel.from_pretrained() API. since standard transformers.generate()
is hardcoded for causal autoregressive models, this class encapsulates the iterative
denoising sampler directly alongside the weights and tokenizer.
"""

from pathlib import Path
import json
import numpy as np
import torch
from tokenizers import Tokenizer
from mdiff.model.transformer import TelosTransformer, TelosConfig
from mdiff.diffusion.sampler import MDLMSampler


class TelosModel:
    """High-level standalone inference wrapper for télos models.
    
    This class orchestrates the complete iterative denoising inference pipeline 
    for Masked Diffusion Language Models (MDLMs). It manages the initialization 
    of the underlying Transformer architecture, the loading of tokenizer 
    assets, and the execution of the diffusion sampling loop. Unlike 
    autoregressive models which predict tokens monotonically from left to 
    right, this model begins with a fully masked sequence and iteratively 
    predicts all tokens in parallel, solidifying predictions based on the 
    defined unmasking schedule and temperature dynamics.
    """

    def __init__(self, model: TelosTransformer, tokenizer: Tokenizer, config: TelosConfig):
        """Initializes the inference wrapper with pre-loaded model components."""
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _load_weights_dict(model_path: Path):
        """Locates and loads raw state dictionary from safetensors or pt."""
        weights_file = None
        for name in ["checkpoint_final.pt", "model.pt", "model.safetensors"]:
            if (model_path / name).exists():
                weights_file = model_path / name
                break
        if weights_file is None and model_path.is_file():
            weights_file = model_path
        elif weights_file is None:
            st_files = sorted(list(model_path.glob("*.safetensors")), key=lambda p: p.stat().st_mtime)
            pt_files = sorted(list(model_path.glob("*.pt")), key=lambda p: p.stat().st_mtime)
            if st_files:
                weights_file = st_files[-1]
            elif pt_files:
                weights_file = pt_files[-1]

        assert weights_file is not None, f"No model weights found in {model_path}"

        if str(weights_file).endswith(".safetensors"):
            try:
                from safetensors.torch import load_file
                state_dict = load_file(str(weights_file))
            except Exception:
                state_dict = torch.load(str(weights_file), map_location="cpu", weights_only=False)
        else:
            state_dict = torch.load(str(weights_file), map_location="cpu", weights_only=False)
            
        return state_dict, weights_file

    @staticmethod
    def _parse_config(model_path: Path, embedded_cfg):
        """Resolves configuration from embedded dictionary, json file, or fallbacks."""
        if isinstance(embedded_cfg, TelosConfig):
            return embedded_cfg
        elif (model_path / "config.json").exists():
            with open(model_path / "config.json", "r") as f:
                cfg_dict = json.load(f)
            return TelosConfig(**cfg_dict)
        elif embedded_cfg and isinstance(embedded_cfg, dict) and "model" in embedded_cfg:
            return TelosConfig(**embedded_cfg["model"])
        elif embedded_cfg and isinstance(embedded_cfg, dict):
            return TelosConfig(**embedded_cfg)
        elif Path("configs/phase_b_25m_mlx.yaml").exists():
            import yaml
            with open("configs/phase_b_25m_mlx.yaml", "r") as f:
                cfg_dict = yaml.safe_load(f)["model"]
            return TelosConfig(**cfg_dict)
        elif Path("configs/phase_a.yaml").exists():
            import yaml
            with open("configs/phase_a.yaml", "r") as f:
                cfg_dict = yaml.safe_load(f)["model"]
            return TelosConfig(**cfg_dict)
        return TelosConfig()

    @staticmethod
    def _load_tokenizer(model_path: Path):
        """Locates and instantiates tokenizer from local paths."""
        tokenizer_file = None
        for tok_path in [model_path / "tokenizer.json", Path("configs/tokenizer_mac.json"), Path("configs/tokenizer.json")]:
            if tok_path.exists():
                tokenizer_file = tok_path
                break
        assert tokenizer_file is not None, "Tokenizer file missing"
        return Tokenizer.from_file(str(tokenizer_file))

    @staticmethod
    def _convert_state_dict(state_dict: dict, is_safetensors: bool) -> dict:
        """Converts legacy and framework-specific weights to expected schema."""
        new_state_dict = {}
        for key, val in state_dict.items():
            if not isinstance(val, torch.Tensor):
                val = torch.from_numpy(np.array(val)) if hasattr(val, "__array__") else torch.tensor(val)

            if "attn.qkv_proj.weight" in key:
                prefix = key.rsplit("qkv_proj.weight", 1)[0]
                d_model = val.shape[0] // 3
                new_state_dict[prefix + "q_proj.weight"] = val[:d_model]
                new_state_dict[prefix + "k_proj.weight"] = val[d_model:2*d_model]
                new_state_dict[prefix + "v_proj.weight"] = val[2*d_model:]
            elif key.startswith("emb."):
                new_state_dict["tok_embeddings.weight"] = val
            elif key.startswith("layers."):
                k_clean = key.replace(".norm1.", ".attn_norm.").replace(".norm2.", ".mlp_norm.")
                k_clean = k_clean.replace(".out.", ".attn.o_proj.")
                # Transpose MLX weights for safetensors.
                if is_safetensors and k_clean.endswith(".weight") and ("attn.q_proj" in k_clean or "attn.k_proj" in k_clean or "attn.v_proj" in k_clean or "attn.o_proj" in k_clean or "mlp.w2" in k_clean or "mlp.w3" in k_clean):
                    val = val.T
                new_state_dict[k_clean] = val
            elif key.startswith("norm."):
                new_state_dict["final_norm.weight"] = val
            elif key.startswith("head."):
                new_state_dict["output_projection.weight"] = val
            else:
                new_state_dict[key] = val
        return new_state_dict

    @classmethod
    def from_pretrained(cls, model_path_or_id: str | Path = "checkpoints") -> "TelosModel":
        """Instantiates a complete TelosModel pipeline from a designated directory.
        
        This factory method orchestrates the discovery and resolution of 
        model weights, configuration schemas, and tokenizer definitions. It 
        gracefully handles weight transposition and key mapping for weights 
        originally trained in external frameworks such as MLX. The resulting 
        instance is fully primed for device placement and inference.
        
        Args:
            model_path_or_id: Path pointing to the directory containing weights.
            
        Returns:
            An instantiated TelosModel ready for sampling.
        """
        model_path = Path(model_path_or_id)

        raw_state_dict, weights_file = cls._load_weights_dict(model_path)

        embedded_cfg = None
        if isinstance(raw_state_dict, dict) and "model_state_dict" in raw_state_dict:
            embedded_cfg = raw_state_dict.get("config")
            raw_state_dict = raw_state_dict["model_state_dict"]

        config = cls._parse_config(model_path, embedded_cfg)
        tokenizer = cls._load_tokenizer(model_path)

        is_st = str(weights_file).endswith(".safetensors")
        new_state_dict = cls._convert_state_dict(raw_state_dict, is_st)

        model = TelosTransformer(config)
        model.load_state_dict(new_state_dict, strict=False)
        return cls(model, tokenizer, config)

    @torch.no_grad()
    def complete(
        self,
        prompt: str,
        max_tokens: int = 128,
        num_steps: int = 64,
        temperature: float = 0.3,
        repetition_penalty: float = 1.2,
        schedule: str = "linear"
    ) -> str:
        """Executes full masked diffusion iterative unmasking generation.
        
        This method translates the provided prompt string into token ids, 
        concatenates them with `max_tokens` masked positions, and initiates 
        the reverse diffusion sampler. Through exactly `num_steps` iterations, 
        the model parallelizes prediction across all currently masked positions, 
        selecting and freezing the highest confidence tokens as dictated by 
        the designated mathematical schedule. The ultimate generated token 
        sequence is then detokenized and optionally truncated.
        
        Args:
            prompt: The string prefix acting as the conditioned context.
            max_tokens: The exact number of mask tokens to append and solve.
            num_steps: Total number of denoising iteration steps to execute.
            temperature: Softmax scaling factor applied prior to token sampling.
            repetition_penalty: Logit suppression factor targeting previous tokens.
            schedule: The functional curve governing the unmasking rate.
            
        Returns:
            The raw text string representing the final generated code.
        """
        encoded = self.tokenizer.encode(prompt)
        prompt_ids = torch.tensor([encoded.ids], dtype=torch.long, device=self.device)

        mask_token_id = self.tokenizer.token_to_id("[MASK]") or 4
        total_seq_len = len(encoded.ids) + max_tokens

        sampler = MDLMSampler(
            self.model,
            mask_token_id=mask_token_id,
            num_steps=num_steps,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            schedule=schedule
        )

        sampled_ids = sampler.sample(seq_len=total_seq_len, prompt_ids=prompt_ids, device=self.device)
        
        prompt_len = prompt_ids.shape[1]
        completion_ids = sampled_ids[0, prompt_len:]
        full_text = self.tokenizer.decode(completion_ids.tolist(), skip_special_tokens=True)

        for stop_str in ["[EOS]", "[PAD]", "<|endoftext|>"]:
            if stop_str in full_text:
                full_text = full_text.split(stop_str)[0]

        return full_text.rstrip()
