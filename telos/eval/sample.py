"""
Qualitative Sampling and Code Structure Evaluation Script.
"""

import ast
import torch
from tokenizers import Tokenizer
from telos.diffusion.sampler import MDLMSampler


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
    num_steps: int = 64,
    device: str | torch.device = "cpu"
):
    """Runs qualitative code generation tests on sample prompts."""
    if prompts is None:
        prompts = [
            "def add(a: int, b: int) -> int:\n    \"\"\"Add two numbers.\"\"\"\n    return",
            "def fibonacci(n: int) -> int:\n    \"\"\"Calculate nth Fibonacci number.\"\"\"\n",
            "def reverse_list(items: list) -> list:\n    \"\"\"Return reversed list.\"\"\"\n"
        ]

    sampler = MDLMSampler(model, mask_token_id=1, num_steps=num_steps, schedule="linear")
    print(f"\n--- Qualitative Sampling Evaluation (Steps={num_steps}) ---")

    for idx, prompt_text in enumerate(prompts):
        encoded = tokenizer.encode(prompt_text)
        prompt_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)

        sampled_ids = sampler.sample(seq_len=128, prompt_ids=prompt_ids, device=device)

        completed_text = tokenizer.decode(sampled_ids[0].tolist())

        valid_syntax = check_syntax_validity(completed_text)

        print(f"\n[Prompt #{idx + 1}]:\n{prompt_text}")
        print(f"[Generated Completion]:\n{completed_text}")
        print(f"[AST Valid Python]: {valid_syntax}")
