"""Gradio Interactive Web UI for HuggingFace Spaces.

Renders an interactive web interface for télos Masked Diffusion Code Autocomplete,
featuring step-by-step unmasking speed controls and syntax-highlighted code output.
"""

import gradio as gr
import torch
from telos.hub.inference import TelosModel


def build_app(model_path_or_id: str = "checkpoints"):
    """Constructs Gradio Interface."""
    try:
        model = TelosModel.from_pretrained(model_path_or_id)
    except Exception:
        model = None

    def predict(prompt: str, max_tokens: int, num_steps: int, temperature: float):
        if model is None:
            return "# Model not loaded yet. Train Phase A or Phase B model first."

        completion = model.complete(
            prompt=prompt,
            max_tokens=int(max_tokens),
            num_steps=int(num_steps),
            temperature=float(temperature)
        )
        return completion

    demo = gr.Interface(
        fn=predict,
        inputs=[
            gr.Code(
                value="def fibonacci(n: int) -> int:\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n",
                language="python",
                label="Input Prompt (Function Signature + Docstring)"
            ),
            gr.Slider(minimum=32, maximum=512, value=128, step=32, label="Max Target Tokens"),
            gr.Slider(minimum=16, maximum=128, value=64, step=8, label="Denoising Steps (Speed vs Quality Knob)"),
            gr.Slider(minimum=0.1, maximum=1.5, value=0.8, step=0.1, label="Sampling Temperature"),
        ],
        outputs=gr.Code(language="python", label="Télos Masked Diffusion Completion"),
        title="τέλος (télos) — Masked Diffusion Language Model for Code Autocomplete",
        description="Iterative discrete diffusion code completion model trained from scratch with full bidirectional self-attention."
    )
    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
