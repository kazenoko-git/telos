"""Gradio Interactive Web UI for télos MDLM.

Renders an interactive web interface for télos Masked Diffusion Code Autocomplete,
featuring checkpoint selection, step-by-step unmasking speed controls, temperature adjustment,
and syntax-highlighted code output.
"""

from pathlib import Path
import gradio as gr
from telos.hub.inference import TelosModel

# Model cache for dynamic switching
LOADED_MODELS = {}


def get_available_checkpoints() -> list[str]:
    """Scans checkpoints directory for available model directories or weights."""
    ckpt_base = Path("checkpoints")
    options = []
    if ckpt_base.exists():
        for item in ckpt_base.iterdir():
            if item.is_dir():
                # Check if dir contains weights
                if list(item.glob("*.safetensors")) or list(item.glob("*.pt")):
                    options.append(str(item))
            elif item.suffix in [".pt", ".safetensors"]:
                options.append(str(item))

    if not options:
        options = ["checkpoints/phase_b_25m_mlx"]
    return sorted(options)


def load_model(checkpoint_path: str) -> TelosModel | None:
    """Loads and caches TelosModel from checkpoint path."""
    if checkpoint_path in LOADED_MODELS:
        return LOADED_MODELS[checkpoint_path]

    try:
        model = TelosModel.from_pretrained(checkpoint_path)
        LOADED_MODELS[checkpoint_path] = model
        return model
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_path}: {e}")
        return None


def build_app(default_checkpoint: str = "checkpoints/phase_b_25m_mlx"):
    """Constructs enhanced Gradio Blocks Interface."""
    available_ckpts = get_available_checkpoints()
    if default_checkpoint not in available_ckpts and available_ckpts:
        default_checkpoint = available_ckpts[0]

    # Pre-warm default model
    load_model(default_checkpoint)

    def predict(checkpoint: str, prompt: str, max_tokens: int, num_steps: int, temperature: float):
        model = load_model(checkpoint)
        if model is None:
            return f"# Error: Failed to load checkpoint from '{checkpoint}'. Make sure trained weights exist in that directory."

        completion = model.complete(
            prompt=prompt,
            max_tokens=int(max_tokens),
            num_steps=int(num_steps),
            temperature=float(temperature)
        )
        return completion

    with gr.Blocks(title="τέλος — Masked Diffusion Language Model for Code") as demo:
        gr.Markdown(
            """
            # τέλος (télos) — Discrete Masked Diffusion Code Completion
            *Iterative discrete diffusion language model trained from scratch with full bidirectional self-attention.*
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                checkpoint_dropdown = gr.Dropdown(
                    choices=available_ckpts,
                    value=default_checkpoint,
                    label="Model Checkpoint",
                    info="Select trained model checkpoint"
                )
                prompt_input = gr.Code(
                    value="def fibonacci(n: int) -> int:\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n",
                    language="python",
                    label="Input Prompt (Function Signature + Docstring)",
                    lines=6
                )
                max_tokens_slider = gr.Slider(
                    minimum=32, maximum=512, value=128, step=32,
                    label="Max Target Tokens"
                )
                num_steps_slider = gr.Slider(
                    minimum=16, maximum=128, value=64, step=8,
                    label="Denoising Steps (Speed vs Quality Knob)"
                )
                temp_slider = gr.Slider(
                    minimum=0.1, maximum=1.5, value=0.3, step=0.05,
                    label="Sampling Temperature (Lower = Deterministic Code)"
                )
                submit_btn = gr.Button("Generate Code Completion", variant="primary")

            with gr.Column(scale=1):
                output_code = gr.Code(
                    language="python",
                    label="Télos Masked Diffusion Completion",
                    lines=16
                )

        submit_btn.click(
            fn=predict,
            inputs=[checkpoint_dropdown, prompt_input, max_tokens_slider, num_steps_slider, temp_slider],
            outputs=output_code
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
