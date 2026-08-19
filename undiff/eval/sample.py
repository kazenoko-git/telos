"""
Qualitative Sampling and Code Structure Evaluation Script.
"""

import ast
import torch
from tokenizers import Tokenizer
from mdiff.diffusion.sampler import MDLMSampler


def check_syntax_validity(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def run_qualitative_evaluation(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    prompts: list[str] | None = None,
    num_steps: int = 128,
    device: str | torch.device = "cpu"
):
    """Runs qualitative code generation tests on sample prompts."""
    if prompts is None:
        prompts = [
            "def add(a: int, b: int) -> int:\n    return a +",
            "import os\n\ndef get_current_dir():\n    return os.path.dirname(",
            "class User:\n    def __init__(self, username: str):\n        self.username = username\n\n    def get_name(self):\n        return self."
        ]

    mask_token_id = tokenizer.token_to_id("[MASK]")
    if mask_token_id is None:
        mask_token_id = 4
    
    print(f"\n--- Qualitative Sampling Evaluation (Steps={num_steps}) ---")

    for temp in [0.0, 0.5, 1.0]:
        print(f"\n================ TEMPERATURE = {temp} ================")
        sampler = MDLMSampler(model, mask_token_id=mask_token_id, num_steps=num_steps, schedule="linear", temperature=temp)
        
        for idx, prompt_text in enumerate(prompts):
            encoded = tokenizer.encode(prompt_text)
            prompt_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)

            seq_len = prompt_ids.shape[1] + 8
            sampled_ids = sampler.sample(seq_len=seq_len, prompt_ids=prompt_ids, device=device)

            completed_text = tokenizer.decode(sampled_ids[0].tolist())

            print(f"\n[Prompt #{idx + 1}]:\n{prompt_text}")
            print(f"[Generated Completion]:\n{completed_text}")
