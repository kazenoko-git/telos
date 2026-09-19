"""
Hugging Face Transformers Model Adapter.

Supports evaluation of open-weights models:
- Gemma 4 e4b / Gemma 4 12B
- Ternary Bonsai 27B
- Any causal LM from Hugging Face Hub or local path
- Features: device_map="auto", bfloat16, 4-bit/8-bit quantization
"""

from typing import List, Dict, Any, Optional
from .base import BaseModelAdapter


class HuggingFaceAdapter(BaseModelAdapter):
    """Adapter for Hugging Face AutoModelForCausalLM."""

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "auto",
        torch_dtype: str = "bfloat16",
        quantization: Optional[str] = None,  # '4bit', '8bit', None
        trust_remote_code: bool = True,
        **kwargs
    ):
        super().__init__(model_name=model_name_or_path, backend_name="huggingface")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "auto": "auto",
        }
        selected_dtype = dtype_map.get(torch_dtype, torch.bfloat16)

        model_kwargs: Dict[str, Any] = {
            "torch_dtype": selected_dtype,
            "trust_remote_code": trust_remote_code,
        }

        if device == "auto":
            model_kwargs["device_map"] = "auto"
        elif device != "cpu":
            model_kwargs["device_map"] = {"": device}

        if quantization == "4bit":
            model_kwargs["load_in_4bit"] = True
        elif quantization == "8bit":
            model_kwargs["load_in_8bit"] = True

        print(f"Loading Hugging Face model: {model_name_or_path} (dtype={torch_dtype}, quant={quantization})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
        self.model.eval()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """Generates code completion using Hugging Face model."""
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if temperature > 0.0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        continuation_ids = output_ids[0, input_len:]
        continuation_text = self.tokenizer.decode(continuation_ids, skip_special_tokens=True)

        if stop:
            for s in stop:
                if s in continuation_text:
                    continuation_text = continuation_text[:continuation_text.index(s)]

        return continuation_text
